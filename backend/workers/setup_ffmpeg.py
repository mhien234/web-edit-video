"""Việc "Cài FFmpeg": tải bản FFmpeg dựng sẵn cho Windows vào <dự án>/tools/ffmpeg/bin.

Không cần quyền quản trị, không sửa PATH của máy – chỉ dự án này dùng.
Nguồn: bản "release essentials" của gyan.dev (được trang ffmpeg.org giới thiệu cho Windows);
nếu không tải được thì thử bản dựng của BtbN trên GitHub.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile

from ..jobs import JobContext
from ..media import TOOLS_FFMPEG, MediaError, refresh_ffmpeg

SOURCES = [
    ("gyan.dev", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"),
    ("GitHub BtbN", "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"),
]


def _download(url: str, dest, ctx: JobContext, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "CreativeMotionLab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1024 * 256):
            f.write(chunk)
            done += len(chunk)
            if total:
                ctx.report(0.02 + 0.85 * done / total,
                           f"Đang tải FFmpeg từ {label}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
            else:
                ctx.report(0.4, f"Đang tải FFmpeg từ {label}: {done / 1e6:.0f} MB")


def run(ctx: JobContext) -> dict:
    if os.name != "nt":
        raise MediaError("Nút này dành cho Windows. Trên máy khác hãy cài FFmpeg bằng trình quản lý gói.")
    if refresh_ffmpeg():
        return {"file": None, "extra": {"already": True}}
    zip_path = ctx.dir / "ffmpeg.zip"
    errors = []
    for label, url in SOURCES:
        try:
            _download(url, zip_path, ctx, label)
            break
        except Exception as e:  # thử nguồn tiếp theo
            errors.append(f"{label}: {e}")
    else:
        raise MediaError("Không tải được FFmpeg (máy có mạng không?). " + " | ".join(errors)[:300])

    ctx.report(0.9, "Đang giải nén ffmpeg.exe và ffprobe.exe")
    TOOLS_FFMPEG.mkdir(parents=True, exist_ok=True)
    found = set()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1].lower()
            if base in ("ffmpeg.exe", "ffprobe.exe") and "/bin/" in name.lower():
                with z.open(name) as src, open(TOOLS_FFMPEG / base, "wb") as out:
                    shutil.copyfileobj(src, out)
                found.add(base)
    zip_path.unlink(missing_ok=True)
    if found != {"ffmpeg.exe", "ffprobe.exe"} or not refresh_ffmpeg():
        raise MediaError("File tải về không có đủ ffmpeg.exe và ffprobe.exe.")
    return {"file": None, "extra": {"path": str(TOOLS_FFMPEG)}}
