"""API của trình dựng: /api/editor/...

Mọi yêu cầu thay đổi dữ liệu cần header X-CML: 1 (giống /api/jobs), nên trang web lạ
không gửi được. Mở /docs trên máy chủ để bấm thử từng API.
"""
from __future__ import annotations

import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..media import MediaError
from .core import CommandError, Locked, VersionConflict
from .looks import LOOKS
from .service import EditorService, load_templates

MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
        ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac"}


def ranged_file(path: Path, range_header: str | None):
    """Phát file có hỗ trợ tua (HTTP Range) để khung xem trước nhảy tới giữa clip được."""
    size = path.stat().st_size
    mime = MIME.get(path.suffix.lower(), "application/octet-stream")
    if not range_header or not range_header.startswith("bytes="):
        return FileResponse(path, media_type=mime, headers={"Accept-Ranges": "bytes"})
    first = range_header[6:].split(",")[0].strip()
    a, _, b = first.partition("-")
    try:
        if a:
            start, end = int(a), (int(b) if b else size - 1)
        else:
            start, end = max(0, size - int(b)), size - 1
    except ValueError:
        start, end = 0, size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                data = f.read(min(1024 * 1024, left))
                if not data:
                    break
                left -= len(data)
                yield data

    return StreamingResponse(chunks(), status_code=206, media_type=mime, headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1)})


def require_cml(x_cml: str | None = Header(default=None)) -> None:
    if x_cml != "1":
        raise HTTPException(403, "Thiếu header X-CML: 1.")


# Chạy ở localhost (máy của chính chủ, kể cả agent MCP gọi vào) thì ai cũng là "local" -
# dùng một mình nên không cần ngăn cách. Chỉ khi mở công khai qua một tên miền khác
# (ví dụ deploy lên Render) mới phát cho mỗi trình duyệt một "uid" riêng qua cookie,
# để người lạ không thấy được video/dự án của nhau.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def get_uid(request: Request, response: Response) -> str:
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if not host or host in _LOCAL_HOSTS:
        return "local"
    uid = request.cookies.get("cml_uid")
    if not uid:
        uid = secrets.token_urlsafe(24)
        response.set_cookie("cml_uid", uid, max_age=60 * 60 * 24 * 365,
                            httponly=True, samesite="lax", secure=True, path="/")
    return uid


def _owns(stored_uid: str | None, uid: str) -> bool:
    """Chưa gắn chủ (tạo từ trước khi có tính năng này, hoặc dùng ở localhost) -> ai cũng xem được."""
    return stored_uid is None or stored_uid == uid


class NewProject(BaseModel):
    name: str
    width: int = 1080
    height: int = 1920
    fps: int = 30


class CommandIn(BaseModel):
    command: dict[str, Any]
    actor: str = "user"
    expected_version: int | None = None


class Owner(BaseModel):
    owner: str


class PathIn(BaseModel):
    path: str


class ToolIn(BaseModel):
    kind: str
    options: dict[str, Any] = {}


class CaptionsIn(BaseModel):
    job_id: str
    lang: str | None = None
    offset: float = 0.0
    actor: str = "user"


class BeatCutIn(BaseModel):
    music_id: str
    source_ids: list[str]
    beats_per_cut: int = 2
    min_clip: float = 0.3
    look: str | None = None
    max_seconds: float = 60
    sensitivity: int = 60
    actor: str = "user"


class TemplateIn(BaseModel):
    template: str
    source_ids: list[str]
    music_id: str | None = None
    actor: str = "user"


def make_router(svc: EditorService) -> APIRouter:
    r = APIRouter(prefix="/api/editor", tags=["editor"])
    write = [Depends(require_cml)]

    def fail(e: Exception):
        if isinstance(e, VersionConflict):
            raise HTTPException(409, {"error": str(e), "current_version": e.current})
        if isinstance(e, Locked):
            raise HTTPException(423, {"error": str(e), "owner": e.owner})
        if isinstance(e, (CommandError, KeyError, ValueError, MediaError)):
            raise HTTPException(400, {"error": str(e).strip("'\"")})
        raise e

    # ---------- media ----------
    @r.get("/media")
    def list_media(uid: str = Depends(get_uid)):
        return [m.__dict__ for m in svc.store.list_media() if _owns(svc.store.media_uid(m.id), uid)]

    @r.get("/media/{media_id}/file")
    def media_file(media_id: str, request: Request, uid: str = Depends(get_uid)):
        if not _owns(svc.store.media_uid(media_id), uid):
            raise HTTPException(404, "Không có file này.")
        try:
            m = svc.store.get_media(media_id)
        except KeyError:
            raise HTTPException(404, "Không có file này.")
        return ranged_file(Path(m.path), request.headers.get("range"))

    @r.get("/media/{media_id}/thumb")
    def media_thumb(media_id: str, uid: str = Depends(get_uid)):
        if not _owns(svc.store.media_uid(media_id), uid):
            raise HTTPException(404, "Không có file này.")
        try:
            return FileResponse(svc.thumbnail(media_id), media_type="image/jpeg",
                                headers={"Cache-Control": "max-age=86400"})
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/media/upload", dependencies=write)
    def upload(file: UploadFile = File(...), uid: str = Depends(get_uid)):
        suffix = Path(file.filename or "file").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=svc.media_dir) as tmp:
            shutil.copyfileobj(file.file, tmp)
        try:
            m = svc.import_file(Path(tmp.name), file.filename, move=True)
            svc.store.set_media_uid(m.id, uid)
            return m.__dict__
        except Exception as e:  # noqa: BLE001
            Path(tmp.name).unlink(missing_ok=True)
            fail(e)

    @r.post("/media/import-path", dependencies=write)
    def import_path(body: PathIn, uid: str = Depends(get_uid)):
        """Cho agent: nhập một file đang có trên máy này theo đường dẫn đầy đủ."""
        try:
            m = svc.import_file(Path(body.path))
            svc.store.set_media_uid(m.id, uid)
            return m.__dict__
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/media/{media_id}/delete", dependencies=write)
    def delete_media(media_id: str, force: bool = False, uid: str = Depends(get_uid)):
        """Xóa media khỏi thư viện và khỏi đĩa. POST chứ không DELETE: CORS chỉ mở GET/POST."""
        if not _owns(svc.store.media_uid(media_id), uid):
            raise HTTPException(404, "Không có file này.")
        try:
            return svc.delete_media(media_id, force=force)
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/media/{media_id}/tool", dependencies=write)
    def run_tool(media_id: str, body: ToolIn, uid: str = Depends(get_uid)):
        """Chạy worker có sẵn trên file: denoise, subtitles (mode=transcribe), voiceover. Trả về mã việc."""
        if not _owns(svc.store.media_uid(media_id), uid):
            raise HTTPException(404, "Không có file này.")
        try:
            return {"job_id": svc.submit_tool(body.kind, media_id, body.options)}
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/media/from-job/{job_id}", dependencies=write)
    def from_job(job_id: str, uid: str = Depends(get_uid)):
        try:
            m = svc.import_job_result(job_id)
            svc.store.set_media_uid(m.id, uid)
            return m.__dict__
        except Exception as e:  # noqa: BLE001
            fail(e)

    # ---------- project ----------
    def _guard_project(pid: str, uid: str) -> None:
        """Chặn truy cập dự án của người khác kể cả khi biết/đoán đúng id."""
        if not _owns(svc.store.project_uid(pid), uid):
            raise HTTPException(404, "Không tìm thấy dự án này.")

    @r.get("/projects")
    def projects(uid: str = Depends(get_uid)):
        return [p for p in svc.store.list_projects() if _owns(svc.store.project_uid(p["id"]), uid)]

    @r.post("/projects", dependencies=write)
    def create(body: NewProject, uid: str = Depends(get_uid)):
        p = svc.core.create_project(body.name, body.width, body.height, body.fps)
        svc.store.set_project_uid(p.id, uid)
        return {"project": p.to_dict(), "version": 1}

    @r.get("/projects/{pid}")
    def get_project(pid: str, since_version: int | None = None, uid: str = Depends(get_uid)):
        """Giao diện hỏi mỗi 1 giây; có since_version thì trả {changed:false} rất nhẹ."""
        _guard_project(pid, uid)
        try:
            head, latest = svc.store.head(pid)
            if since_version is not None and since_version == head:
                return {"changed": False, "version": head}
            p, v = svc.core.get(pid)
            return {"changed": True, "project": p.to_dict(), "version": v, "can_undo": v > 1,
                    "can_redo": v < latest, "duration": round(p.duration, 3),
                    "locked_by": svc.store.lock_owner(pid)}
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/delete", dependencies=write)
    def delete_project(pid: str, uid: str = Depends(get_uid)):
        """Xóa một dự án. Dùng POST chứ không dùng DELETE vì CORS của máy chủ
        chỉ mở cho GET và POST (xem allow_methods trong backend/main.py)."""
        _guard_project(pid, uid)
        try:
            svc.store.head(pid)                 # không có thì báo lỗi rõ ràng
            svc.store.delete_project(pid)
            return {"deleted": pid}
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.get("/projects/{pid}/history")
    def history(pid: str, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        return svc.store.history(pid)

    @r.post("/projects/{pid}/commands", dependencies=write)
    def command(pid: str, body: CommandIn, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return svc.core.apply(pid, body.command, body.actor, body.expected_version)
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/undo", dependencies=write)
    def undo(pid: str, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return {"version": svc.core.undo(pid)}
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/redo", dependencies=write)
    def redo(pid: str, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return {"version": svc.core.redo(pid)}
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/lock", dependencies=write)
    def lock(pid: str, body: Owner, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        if not svc.core.lock(pid, body.owner):
            raise HTTPException(423, {"error": "Project đang được agent khác sửa.", "owner": svc.store.lock_owner(pid)})
        return {"ok": True, "ttl_seconds": svc.core.lock_ttl}

    @r.post("/projects/{pid}/unlock", dependencies=write)
    def unlock(pid: str, body: Owner, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        svc.core.unlock(pid, body.owner)
        return {"ok": True}

    # ---------- AI trên timeline ----------
    @r.post("/projects/{pid}/beat-cut", dependencies=write)
    def beat_cut(pid: str, body: BeatCutIn, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return svc.beat_cut(pid, **body.model_dump())
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/template", dependencies=write)
    def template(pid: str, body: TemplateIn, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return svc.apply_template(pid, **body.model_dump())
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/captions-from-job", dependencies=write)
    def captions(pid: str, body: CaptionsIn, uid: str = Depends(get_uid)):
        _guard_project(pid, uid)
        try:
            return svc.captions_from_job(pid, **body.model_dump())
        except Exception as e:  # noqa: BLE001
            fail(e)

    @r.post("/projects/{pid}/export", dependencies=write)
    def export(pid: str, uid: str = Depends(get_uid)):
        """Xuất MP4 trong hàng đợi việc chung. Theo dõi bằng GET /api/jobs/{job_id}."""
        _guard_project(pid, uid)
        try:
            svc.core.get(pid)
        except Exception as e:  # noqa: BLE001
            fail(e)
        job_id, _ = svc.jobs.new_job_dir()
        svc.jobs.submit(job_id, "render", {"project_id": pid})
        return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}

    @r.get("/templates")
    def templates():
        return load_templates()

    @r.get("/looks")
    def looks():
        return sorted(LOOKS)

    return r
