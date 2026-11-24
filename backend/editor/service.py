"""Các việc của trình dựng: thư viện media, project, cắt theo beat, nối với các worker AI có sẵn.

API web (api.py) và cổng MCP (mcp_server.py, qua HTTP) đều gọi vào đây, và mọi thay đổi
timeline đều đi qua EditorCore, nên người và agent luôn sửa cùng một project.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import media
from ..jobs import DATA, JobContext, JobQueue
from ..workers.beat_sync import SR, detect_beats
from . import render
from .core import EditorCore
from .models import Media, new_id
from .store import Store

EDITOR_DIR = DATA / "editor"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
# Các worker sẵn có nhận file video/âm thanh qua tên "video_..."; nối được với trình dựng.
TOOL_KINDS = {"denoise", "subtitles", "voiceover", "remove_bg"}


def load_templates() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(TEMPLATE_DIR.glob("*.json"))}


class EditorService:
    def __init__(self, jobs: JobQueue, data_dir: Path = EDITOR_DIR, lock_ttl: float = 120):
        self.jobs = jobs
        self.dir = data_dir
        self.media_dir = data_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(data_dir / "editor.db")
        self.core = EditorCore(self.store, lock_ttl=lock_ttl)

    # ------------------------------------------------------------ media
    def import_file(self, source: Path, original_name: str | None = None, move: bool = False) -> Media:
        """Chép file vào thư mục dữ liệu dưới tên ASCII ngắn (tránh lỗi đường dẫn có dấu)."""
        source = Path(source)
        ext = source.suffix.lower()
        if ext not in VIDEO_EXT | AUDIO_EXT:
            raise media.MediaError(f"Không hỗ trợ định dạng {ext or '(không rõ)'}.")
        mid = new_id("med")
        dest = self.media_dir / f"{mid}{ext}"
        (shutil.move if move else shutil.copy2)(str(source), str(dest))
        try:
            info = describe(dest)
            if info["variable_fps"]:
                # Video quay điện thoại/game hay có tốc độ khung hình thay đổi: điểm cắt sẽ trôi khỏi beat.
                cfr = self.media_dir / f"{mid}_cfr.mp4"
                media.run_ffmpeg(["-i", str(dest), "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast",
                                  "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(cfr)])
                dest.unlink(missing_ok=True)
                dest = cfr
                info = describe(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        m = Media(id=mid, path=str(dest), original_name=original_name or source.name, **info)
        self.store.add_media(m)
        return m

    def delete_media(self, media_id: str, force: bool = False) -> dict:
        """Xóa một media khỏi thư viện: bỏ bản ghi, xóa file và ảnh thu nhỏ trên đĩa.

        Mặc định từ chối nếu còn dự án đang dùng, để người dùng không vô tình làm hỏng
        một bản dựng. Truyền force=True khi họ đã xác nhận.
        """
        m = self.store.get_media(media_id)          # không có thì KeyError -> 400
        dang_dung = self.store.media_in_use(media_id)
        if dang_dung and not force:
            raise ValueError("Media đang được dùng trong: " + ", ".join(dang_dung))
        self.store.delete_media(media_id)
        da_xoa = []
        for p in (Path(m.path), self.media_dir / "thumbs" / f"{media_id}.jpg"):
            try:
                if p.is_file():
                    p.unlink()
                    da_xoa.append(p.name)
            except OSError:
                pass        # file đang bị khóa: bản ghi đã bỏ rồi, không chặn người dùng
        return {"deleted": media_id, "files": da_xoa, "was_used_by": dang_dung}

    def thumbnail(self, media_id: str) -> Path:
        """Ảnh nhỏ của video (khung ở giây thứ 1), tạo một lần rồi dùng lại."""
        m = self.store.get_media(media_id)
        if not m.has_video:
            raise ValueError("File âm thanh không có ảnh thu nhỏ.")
        out = self.media_dir / "thumbs" / f"{media_id}.jpg"
        if not out.exists():
            out.parent.mkdir(exist_ok=True)
            at = min(1.0, m.duration / 2)
            media.run_ffmpeg(["-ss", f"{at:.2f}", "-i", m.path, "-frames:v", "1",
                              "-vf", "scale=320:-2", "-q:v", "4", str(out)])
        return out

    def submit_tool(self, kind: str, media_id: str, options: dict) -> str:
        """Chạy một worker AI có sẵn (khử ồn, phụ đề, lồng tiếng) trên file trong thư viện."""
        if kind not in TOOL_KINDS:
            raise ValueError(f"Công cụ không hỗ trợ: {kind}. Dùng: {sorted(TOOL_KINDS)}")
        m = self.store.get_media(media_id)
        job_id, job_dir = self.jobs.new_job_dir()
        shutil.copy2(m.path, job_dir / f"video_00_{Path(m.path).name}")
        self.jobs.submit(job_id, kind, {**options, "editor_media_id": media_id})
        return job_id

    def import_job_result(self, job_id: str) -> Media:
        job = self.jobs.get(job_id)
        if not job or job["status"] != "done" or not job["result_file"]:
            raise ValueError("Việc chưa xong hoặc không có file kết quả.")
        path = self.jobs.jobs_dir / job_id / job["result_file"]
        return self.import_file(path, f"{job['kind']}-{job_id}{path.suffix}")

    # ------------------------------------------------------------ AI trên timeline
    def captions_from_job(self, project_id: str, job_id: str, lang: str | None = None, offset: float = 0.0,
                          actor: str = "user") -> dict:
        """Đưa kết quả của việc 'subtitles' (chế độ transcribe) lên một track chữ - một bước hoàn tác."""
        job = self.jobs.get(job_id)
        if not job or job["status"] != "done" or not job.get("extra"):
            raise ValueError("Việc phụ đề chưa xong.")
        extra = job["extra"]
        segs = extra.get("tracks", {}).get(lang) if lang else None
        segs = segs or extra.get("original") or extra.get("segments") or []
        if not segs:
            raise ValueError("Việc phụ đề không có câu nào.")
        name = f"Phụ đề ({lang or extra.get('source_lang', 'gốc')})"
        tid = new_id("trk")
        clips = [{"start": round(float(a) + offset, 3), "in_point": 0.0, "out_point": round(float(b) - float(a), 3),
                  "text": str(t).strip()} for a, b, t in segs if float(b) > float(a)]
        res = self.core.apply(project_id, {"type": "batch", "commands": [
            {"type": "add_track", "kind": "text", "name": name, "id": tid},
            {"type": "replace_clips", "track_id": tid, "clips": clips}]}, actor)
        return {"track_id": tid, "lines": len(clips), "version": res["version"]}

    def beat_cut(self, project_id: str, music_id: str, source_ids: list[str], beats_per_cut: int = 2,
                 min_clip: float = 0.3, look: str | None = None, max_seconds: float = 60,
                 sensitivity: int = 60, actor: str = "user") -> dict:
        """Dò beat bằng bộ dò có sẵn của dự án, rồi dựng thành timeline (sửa tay tiếp được)."""
        music = self.store.get_media(music_id)
        total = min(music.duration, float(max_seconds))
        info = detect_beats(media.decode_mono(Path(music.path), SR, max_seconds=total), sensitivity)
        sources = [{"media_id": s, "duration": self.store.get_media(s).duration} for s in source_ids]
        clips = plan_cuts(info["beats"], sources, beats_per_cut, min_clip, total, look)
        for c in clips:
            c["volume"] = 0.0   # tắt tiếng clip gốc để nghe rõ nhạc
        p, _ = self.core.get(project_id)
        video = next((t for t in p.tracks if t.kind == "video"), None)
        cmds = []
        vid = video.id if video else new_id("trk")
        if not video:
            cmds.append({"type": "add_track", "kind": "video", "name": "Video", "id": vid})
        mus = new_id("trk")
        cmds += [{"type": "replace_clips", "track_id": vid, "clips": clips},
                 {"type": "add_track", "kind": "audio", "name": "Nhạc", "id": mus},
                 {"type": "add_clip", "track_id": mus, "clip": {"media_id": music_id, "start": 0.0,
                                                                "in_point": 0.0, "out_point": round(total, 3)}}]
        res = self.core.apply(project_id, {"type": "batch", "commands": cmds}, actor)
        return {"bpm": info["bpm"], "beats": len(info["beats"]), "clips": len(clips),
                "duration": round(total, 2), "version": res["version"]}

    def apply_template(self, project_id: str, template: str, source_ids: list[str], music_id: str | None = None,
                       actor: str = "user") -> dict:
        tpl = load_templates()[template]
        self.core.apply(project_id, {"type": "set_project", "width": tpl["width"], "height": tpl["height"],
                                     "fps": tpl["fps"]}, actor)
        if tpl["mode"] == "beat":
            if not music_id:
                raise ValueError("Mẫu này cần một bài nhạc (music_id).")
            return self.beat_cut(project_id, music_id, source_ids, tpl["beats_per_cut"], tpl["min_clip"],
                                 tpl.get("look"), actor=actor)
        cursor, clips = 0.0, []
        for sid in source_ids:
            m = self.store.get_media(sid)
            clips.append({"media_id": sid, "start": round(cursor, 3), "in_point": 0.0,
                          "out_point": round(m.duration, 3), "look": tpl.get("look")})
            cursor += m.duration
        tid = new_id("trk")
        res = self.core.apply(project_id, {"type": "batch", "commands": [
            {"type": "add_track", "kind": "video", "name": "Video", "id": tid},
            {"type": "replace_clips", "track_id": tid, "clips": clips}]}, actor)
        out = {"clips": len(clips), "duration": round(cursor, 2), "version": res["version"]}
        if tpl.get("captions"):
            out["next_step"] = ("Chạy công cụ 'subtitles' (mode=transcribe) cho từng clip, "
                                "rồi gọi captions-from-job với offset = start của clip.")
        return out

    # ------------------------------------------------------------ xuất video (chạy trong hàng đợi việc)
    def render_job(self, ctx: JobContext) -> dict:
        p, version = self.core.get(ctx.params["project_id"])
        out, report = render.export(p, self.store.get_media, ctx.dir, ctx.report)
        if not report["ok"]:
            raise media.MediaError("File xuất ra có vấn đề: " + "; ".join(report["problems"]))
        return {"file": out, "extra": {**report, "project_id": p.id, "version": version}}


def describe(path: Path) -> dict:
    from fractions import Fraction
    info = media.probe(path)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def rate(t):
        try:
            return float(Fraction(t)) if t and t != "0/0" else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    r = a = 0.0
    if video:
        r, a = rate(video.get("r_frame_rate")), rate(video.get("avg_frame_rate"))
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, ValueError):
        duration = media.duration_of(path)
    return {"duration": round(duration, 3), "has_video": video is not None, "has_audio": audio is not None,
            "width": int(video["width"]) if video else 0, "height": int(video["height"]) if video else 0,
            "fps": round(a or r, 3), "variable_fps": bool(video and r and a and abs(r - a) > 0.5)}


def plan_cuts(beats: list[float], sources: list[dict], beats_per_cut: int = 2, min_clip: float = 0.3,
              total: float | None = None, look: str | None = None) -> list[dict]:
    """Đổi cảnh mỗi `beats_per_cut` beat; xoay vòng các video nguồn và đi tiếp trong từng video."""
    if not sources:
        raise ValueError("Cần ít nhất một video nguồn.")
    points = [b for b in beats if total is None or b < total][::max(1, int(beats_per_cut))]
    merged = [0.0]
    for p in points:
        if p - merged[-1] >= min_clip:
            merged.append(p)
    if total:
        if total - merged[-1] < min_clip and len(merged) > 1:
            merged[-1] = total      # đoạn cuối quá ngắn -> gộp vào đoạn trước
        elif merged[-1] < total:
            merged.append(total)
    used = {s["media_id"]: 0.0 for s in sources}
    clips, k = [], 0
    for a, b in zip(merged, merged[1:]):
        length = b - a
        for _ in range(len(sources)):
            src = sources[k % len(sources)]
            k += 1
            pos = used[src["media_id"]]
            if pos + length > src["duration"]:
                pos = 0.0 if length <= src["duration"] else None
            if pos is not None:
                break
        else:
            raise ValueError(f"Không video nào đủ dài cho một cảnh {length:.2f}s.")
        used[src["media_id"]] = pos + length
        clip = {"media_id": src["media_id"], "start": round(a, 3), "in_point": round(pos, 3),
                "out_point": round(pos + length, 3)}
        if look:
            clip["look"] = look
        clips.append(clip)
    return clips
