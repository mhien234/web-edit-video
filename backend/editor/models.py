"""Project data model: a project holds tracks, a track holds clips.

Times are in seconds on the timeline. A clip shows media from `in_point`
to `out_point`, starting at `start` on the timeline.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

TRACK_KINDS = ("video", "audio", "text")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class Clip:
    id: str
    start: float
    in_point: float = 0.0
    out_point: float = 0.0
    media_id: str | None = None
    text: str | None = None          # text clips (captions, titles)
    volume: float = 1.0
    look: str | None = None          # film look name, see app/features/looks.py

    @property
    def duration(self) -> float:
        return max(0.0, self.out_point - self.in_point)

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Track:
    id: str
    kind: str
    name: str = ""
    muted: bool = False
    clips: list[Clip] = field(default_factory=list)


@dataclass
class Project:
    id: str
    name: str
    width: int = 1080
    height: int = 1920
    fps: int = 30
    tracks: list[Track] = field(default_factory=list)
    caption_font: str = "Arial"
    caption_size: int = 0          # 0 = tự co theo khung hình (xem render.py)
    caption_color: str = "#FFFFFF"
    caption_bold: bool = True
    caption_preset: str = "classic"   # kiểu viền/nền chữ, xem CAPTION_PRESETS ở render.py
    # Che phụ đề cháy sẵn trong video gốc. Vùng che tính theo % khung hình để đổi
    # độ phân giải vẫn đúng chỗ. Mặc định tắt: không bật thì video xuất ra y như cũ.
    mask_on: bool = False
    mask_mode: str = "blur"           # "blur" = làm mờ, "box" = hộp đen đặc
    mask_x: float = 5.0
    mask_y: float = 78.0
    mask_w: float = 90.0
    mask_h: float = 16.0

    @property
    def duration(self) -> float:
        ends = [c.end for t in self.tracks for c in t.clips]
        return max(ends) if ends else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Project":
        tracks = [
            Track(id=t["id"], kind=t["kind"], name=t.get("name", ""), muted=t.get("muted", False),
                  clips=[Clip(**c) for c in t.get("clips", [])])
            for t in d.get("tracks", [])
        ]
        return Project(id=d["id"], name=d["name"], width=d.get("width", 1080), height=d.get("height", 1920),
                       fps=d.get("fps", 30), tracks=tracks,
                       caption_font=d.get("caption_font", "Arial"), caption_size=d.get("caption_size", 0),
                       caption_color=d.get("caption_color", "#FFFFFF"), caption_bold=d.get("caption_bold", True),
                       caption_preset=d.get("caption_preset", "classic"),
                       mask_on=d.get("mask_on", False), mask_mode=d.get("mask_mode", "blur"),
                       mask_x=d.get("mask_x", 5.0), mask_y=d.get("mask_y", 78.0),
                       mask_w=d.get("mask_w", 90.0), mask_h=d.get("mask_h", 16.0))

    def find_track(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(f"track not found: {track_id}")

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for t in self.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        raise KeyError(f"clip not found: {clip_id}")


@dataclass
class Media:
    id: str
    path: str
    original_name: str
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    variable_fps: bool = False
