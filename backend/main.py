"""Máy chủ Creative Motion Lab.

Chạy:  python -m uvicorn backend.main:app --port 8000
Rồi mở http://localhost:8000 – máy chủ phát luôn website trong thư mục frontend/.

API:
  GET  /api/health              kiểm tra FFmpeg, AI, số việc đang chờ
  POST /api/jobs                gửi việc (form: kind, options, video[], music)
  GET  /api/jobs/{id}           trạng thái + tiến độ
  GET  /api/jobs/{id}/result    tải file kết quả
  /api/editor/...               trình dựng nhiều track, hoàn tác, cho cả người và AI agent
                                (xem backend/editor/api.py, hoặc mở /docs)
"""
from __future__ import annotations

import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import media
from .editor.api import make_router
from .editor.service import EditorService
from .jobs import ROOT, JobQueue
from .workers import HANDLERS
from .workers.remove_bg import rembg_status
from .workers.subtitles import ai_status
from .workers.voiceover import VOICES, tts_status

FRONTEND = ROOT / "frontend"
MAX_UPLOAD_MB = 1024
ALLOWED = {
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
    "music": {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"},
}
RESULT_NAMES = {"setup_ai": "cai-dat-ai", "beat_sync": "creative-motion-lab-beat", "denoise": "creative-motion-lab-khu-on",
                "subtitles": "creative-motion-lab-phu-de",
                "voiceover": "creative-motion-lab-long-tieng", "setup_tts": "thu-giong-doc",
                "remove_bg": "creative-motion-lab-xoa-nen", "setup_rembg": "cai-dat-xoa-nen",
                "render": "creative-motion-lab-xuat-video"}
# Chỉ nhận yêu cầu gửi tới đúng tên máy này. Chặn kiểu tấn công "DNS rebinding": một trang web lạ
# trỏ tên miền của nó về 127.0.0.1 để gọi vào máy chủ trên máy bạn. Thêm tên khác bằng CML_ALLOWED_HOSTS.
# RENDER_EXTERNAL_HOSTNAME: Render tự đặt biến này thành tên miền công khai của web (xxx.onrender.com)
# nên khi deploy lên Render, không cần tự tay khai CML_ALLOWED_HOSTS.
ALLOWED_HOSTS = {"localhost", "127.0.0.1",
                 *(h.strip().lower() for h in os.environ.get("CML_ALLOWED_HOSTS", "").split(",") if h.strip())}
if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
    ALLOWED_HOSTS.add(os.environ["RENDER_EXTERNAL_HOSTNAME"].strip().lower())

jobs = JobQueue()
for kind, handler in HANDLERS.items():
    jobs.register(kind, handler)
editor = EditorService(jobs)
jobs.register("render", editor.render_job)   # xuất video từ timeline, chạy chung hàng đợi


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.start()          # bật luồng worker chạy ngầm
    yield


app = FastAPI(title="Creative Motion Lab API", lifespan=lifespan)
# Chỉ cho chính website này gọi API: mở qua localhost, hoặc mở thẳng file index.html (origin "null").
# Trang web lạ không gửi được việc vì thiếu header X-CML (trình duyệt chặn ở bước kiểm tra CORS).
app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:8000", "http://127.0.0.1:8000"],
                   allow_methods=["GET", "POST"], allow_headers=["X-CML", "Content-Type"])


@app.middleware("http")
async def host_allowlist(request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": f"Tên máy '{host}' không được phép."}, status_code=403)
    return await call_next(request)


app.include_router(make_router(editor))


@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": media.refresh_ffmpeg(), "pending": jobs.pending(),
            "kinds": sorted(jobs.handlers), **ai_status(), **tts_status(), **rembg_status(), "voices": VOICES}


def _save(upload: UploadFile, dest: Path, allowed: set[str]) -> None:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Không hỗ trợ định dạng {ext or '(không rõ)'} – dùng {', '.join(sorted(allowed))}")
    size = 0
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File quá lớn (tối đa {MAX_UPLOAD_MB} MB).")
            f.write(chunk)


@app.post("/api/jobs")
def create_job(kind: str = Form(...), options: str = Form("{}"),
               video: list[UploadFile] = File(default=[]), music: UploadFile | None = File(default=None),
               x_cml: str | None = Header(default=None)):
    if x_cml != "1":
        raise HTTPException(403, "Chỉ website Creative Motion Lab mới được gửi việc.")
    if kind not in HANDLERS:
        raise HTTPException(400, f"Loại việc không hỗ trợ: {kind}")
    try:
        params = json.loads(options or "{}")
        assert isinstance(params, dict)
    except (ValueError, AssertionError):
        raise HTTPException(400, "options phải là JSON object")
    job_id, job_dir = jobs.new_job_dir()
    try:
        for i, v in enumerate(video):
            _save(v, job_dir / f"video_{i:02d}_{media.safe_name(v.filename, 'video')}", ALLOWED["video"])
        if music is not None:
            _save(music, job_dir / f"music_{media.safe_name(music.filename, 'music')}", ALLOWED["music"])
        jobs.submit(job_id, kind, params)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return {"id": job_id, "status": "queued", "position": jobs.position(job_id)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy việc này (có thể đã bị dọn sau 3 ngày).")
    return {k: job[k] for k in ("id", "kind", "status", "progress", "message", "error", "extra", "position")} | {
        "result_url": f"/api/jobs/{job_id}/result" if job["status"] == "done" and job["result_file"] else None}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done" or not job["result_file"]:
        raise HTTPException(404, "Kết quả chưa sẵn sàng.")
    path = (jobs.jobs_dir / job_id / job["result_file"]).resolve()
    if not path.is_file() or jobs.jobs_dir.resolve() not in path.parents:
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    return FileResponse(path, filename=RESULT_NAMES.get(job["kind"], "ket-qua") + path.suffix)


# Phát website (đặt SAU các route /api để không che mất API).
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
