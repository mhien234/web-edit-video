"""EditorCore: the single door for every change (Vidlab's key idea).

The web UI, the MCP agent and the automatic features all send *commands*
here. Each command makes one new version, so each one is one undo step,
and everyone sees the same project.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from .models import TRACK_KINDS, Clip, Project, Track, new_id
from .store import Store


class CommandError(ValueError):
    pass


class VersionConflict(CommandError):
    def __init__(self, current: int):
        super().__init__(f"project changed; current version is {current}")
        self.current = current


class Locked(CommandError):
    def __init__(self, owner: str):
        super().__init__(f"project is being edited by {owner}")
        self.owner = owner


CLIP_FIELDS = {"start", "in_point", "out_point", "media_id", "text", "volume", "look"}
PROJECT_FIELDS = {"name", "width", "height", "fps",
                  "caption_font", "caption_size", "caption_color", "caption_bold", "caption_preset",
                  "mask_on", "mask_mode", "mask_x", "mask_y", "mask_w", "mask_h"}


def _clip_from(spec: dict[str, Any]) -> Clip:
    unknown = set(spec) - CLIP_FIELDS - {"id"}
    if unknown:
        raise CommandError(f"unknown clip fields: {sorted(unknown)}")
    clip = Clip(id=spec.get("id") or new_id("clip"), **{k: v for k, v in spec.items() if k in CLIP_FIELDS})
    _check_clip(clip)
    return clip


def _check_clip(clip: Clip) -> None:
    if clip.start < 0 or clip.in_point < 0:
        raise CommandError("start and in_point must be >= 0")
    if clip.out_point <= clip.in_point:
        raise CommandError("out_point must be greater than in_point")


# Each handler changes a *copy* of the project and returns a small result.
def _add_track(p: Project, kind: str, name: str = "", id: str | None = None) -> dict:
    if kind not in TRACK_KINDS:
        raise CommandError(f"kind must be one of {TRACK_KINDS}")
    t = Track(id=id or new_id("trk"), kind=kind, name=name or f"{kind} {len(p.tracks) + 1}")
    p.tracks.append(t)
    return {"track_id": t.id}


def _update_track(p: Project, track_id: str, fields: dict) -> dict:
    bad = set(fields) - {"name", "muted"}
    if bad:
        raise CommandError(f"unknown track fields: {sorted(bad)}")
    t = p.find_track(track_id)
    for k, v in fields.items():
        setattr(t, k, v)
    return {}


def _remove_track(p: Project, track_id: str) -> dict:
    p.tracks.remove(p.find_track(track_id))
    return {}


def _add_clip(p: Project, track_id: str, clip: dict) -> dict:
    c = _clip_from(clip)
    p.find_track(track_id).clips.append(c)
    return {"clip_id": c.id}


def _update_clip(p: Project, clip_id: str, fields: dict) -> dict:
    bad = set(fields) - CLIP_FIELDS
    if bad:
        raise CommandError(f"unknown clip fields: {sorted(bad)}")
    _, c = p.find_clip(clip_id)
    for k, v in fields.items():
        setattr(c, k, v)
    _check_clip(c)
    return {}


def _remove_clip(p: Project, clip_id: str) -> dict:
    t, c = p.find_clip(clip_id)
    t.clips.remove(c)
    return {}


def _move_clip(p: Project, clip_id: str, track_id: str, start: float | None = None) -> dict:
    t, c = p.find_clip(clip_id)
    target = p.find_track(track_id)
    t.clips.remove(c)
    if start is not None:
        c.start = float(start)
    target.clips.append(c)
    return {}


def _split_clip(p: Project, clip_id: str, at: float) -> dict:
    """Cut a clip in two at timeline time `at` (the 'blade' tool)."""
    t, c = p.find_clip(clip_id)
    if not (c.start < at < c.end):
        raise CommandError("split point must be inside the clip")
    cut = c.in_point + (at - c.start)
    right = copy.deepcopy(c)
    right.id, right.start, right.in_point = new_id("clip"), at, cut
    c.out_point = cut
    t.clips.insert(t.clips.index(c) + 1, right)
    return {"left_id": c.id, "right_id": right.id}


def _set_project(p: Project, **fields: Any) -> dict:
    bad = set(fields) - PROJECT_FIELDS
    if bad:
        raise CommandError(f"unknown project fields: {sorted(bad)}")
    for k, v in fields.items():
        setattr(p, k, v)
    return {}


def _replace_clips(p: Project, track_id: str, clips: list[dict]) -> dict:
    """Put a whole generated edit (e.g. a beat-cut plan) on a track as ONE undo step."""
    t = p.find_track(track_id)
    t.clips = [_clip_from(c) for c in clips]
    return {"clip_ids": [c.id for c in t.clips]}


HANDLERS: dict[str, Callable[..., dict]] = {
    "add_track": _add_track,
    "remove_track": _remove_track,
    "update_track": _update_track,
    "add_clip": _add_clip,
    "update_clip": _update_clip,
    "remove_clip": _remove_clip,
    "move_clip": _move_clip,
    "split_clip": _split_clip,
    "set_project": _set_project,
    "replace_clips": _replace_clips,
}


class EditorCore:
    def __init__(self, store: Store, lock_ttl: float = 120):
        self.store = store
        self.lock_ttl = lock_ttl

    def create_project(self, name: str, width: int = 1080, height: int = 1920, fps: int = 30,
                       actor: str = "user") -> Project:
        p = Project(id=new_id("prj"), name=name, width=width, height=height, fps=fps)
        self.store.create_project(p, actor)
        return p

    def get(self, project_id: str) -> tuple[Project, int]:
        return self.store.load(project_id)

    def apply(self, project_id: str, command: dict, actor: str = "user",
              expected_version: int | None = None) -> dict:
        """Run one command. Returns {"version": n, "result": {...}}.

        `expected_version` protects against editing a stale copy: if someone
        changed the project since you read it, the command is refused and
        you get the current version instead of silently overwriting.
        """
        # The lock is between agents. The human ("user") can always edit,
        # so they can step in and fix things while an agent works.
        owner = self.store.lock_owner(project_id)
        if owner and owner != actor and actor != "user":
            raise Locked(owner)
        project, head = self.store.load(project_id)
        if expected_version is not None and expected_version != head:
            raise VersionConflict(head)
        cmd = dict(command)
        ctype = cmd.pop("type", None)
        if ctype not in HANDLERS and ctype != "batch":
            raise CommandError(f"unknown command type: {ctype}. Known: {sorted(HANDLERS) + ['batch']}")
        working = copy.deepcopy(project)
        if ctype == "batch":
            # Several commands, ONE version, ONE undo step (e.g. a whole beat cut).
            result = {"results": [self._run(working, dict(c)) for c in cmd.get("commands", [])]}
        else:
            result = self._run(working, {"type": ctype, **cmd})
        version = self.store.commit(working, actor, command)
        return {"version": version, "result": result}

    @staticmethod
    def _run(working: Project, cmd: dict) -> dict:
        ctype = cmd.pop("type", None)
        if ctype not in HANDLERS:
            raise CommandError(f"unknown command type: {ctype}")
        try:
            return HANDLERS[ctype](working, **cmd)
        except KeyError as e:
            raise CommandError(str(e)) from e
        except TypeError as e:
            raise CommandError(f"bad arguments for {ctype}: {e}") from e

    def undo(self, project_id: str) -> int:
        head, _ = self.store.head(project_id)
        if head <= 1:
            raise CommandError("nothing to undo")
        self.store.move_head(project_id, head - 1)
        return head - 1

    def redo(self, project_id: str) -> int:
        head, latest = self.store.head(project_id)
        if head >= latest:
            raise CommandError("nothing to redo")
        self.store.move_head(project_id, head + 1)
        return head + 1

    # Locks expire on their own, so an agent that crashes cannot block the
    # project for the next one (Vidlab pitfall P8).
    def lock(self, project_id: str, owner: str) -> bool:
        return self.store.acquire_lock(project_id, owner, self.lock_ttl)

    def unlock(self, project_id: str, owner: str) -> None:
        self.store.release_lock(project_id, owner)
