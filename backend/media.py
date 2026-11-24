"""Các hàm tiện ích gọi FFmpeg / FFprobe.

Mọi xử lý video và âm thanh nặng đều đi qua FFmpeg chạy như một tiến trình riêng,
nên máy chủ web không bị treo trong lúc dựng video.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Thư mục FFmpeg do website tự cài (nút "Cài FFmpeg"): <dự án>/tools/ffmpeg/bin
TOOLS_FFMPEG = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin"
FFMPEG: str | None = None
FFPROBE: str | None = None


def refresh_ffmpeg() -> bool:
    """Tìm FFmpeg: biến môi trường -> thư mục tools/ của dự án -> PATH của máy."""
    global FFMPEG, FFPROBE
    exe = ".exe" if os.name == "nt" else ""
    local_ff, local_fp = TOOLS_FFMPEG / f"ffmpeg{exe}", TOOLS_FFMPEG / f"ffprobe{exe}"
    FFMPEG = os.environ.get("FFMPEG_PATH") or (str(local_ff) if local_ff.is_file() else None) or shutil.which("ffmpeg")
    FFPROBE = os.environ.get("FFPROBE_PATH") or (str(local_fp) if local_fp.is_file() else None) or shutil.which("ffprobe")
    return bool(FFMPEG and FFPROBE)


refresh_ffmpeg()

# Trên Windows, không bật cửa sổ console đen cho mỗi lần gọi FFmpeg.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

ProgressFn = Callable[[float], None]


class MediaError(RuntimeError):
    """Lỗi khi xử lý file media – thông điệp sẽ được hiển thị cho người dùng."""


def require_ffmpeg() -> None:
    if not FFMPEG or not FFPROBE:
        if refresh_ffmpeg():
            return
        raise MediaError("Chưa có FFmpeg. Bấm nút “Cài FFmpeg” trên website để máy chủ tự cài.")


def run_ffmpeg(args: list[str], *, duration: float | None = None,
               on_progress: Optional[ProgressFn] = None, cwd: Path | None = None) -> None:
    """Chạy FFmpeg; nếu biết thời lượng đầu ra thì báo tiến độ 0..1."""
    require_ffmpeg()
    cmd = [FFMPEG, "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
           "-progress", "pipe:1", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=str(cwd) if cwd else None, creationflags=_NO_WINDOW)
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_progress and duration and line.startswith("out_time_us="):
            try:
                us = int(line.split("=", 1)[1])
                on_progress(min(1.0, us / 1e6 / duration))
            except ValueError:
                pass
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        last = err.strip().splitlines()[-3:] if err.strip() else ["(không có thông báo)"]
        raise MediaError("FFmpeg báo lỗi: " + " | ".join(last))


def probe(path: Path) -> dict:
    require_ffmpeg()
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )
    if out.returncode != 0:
        raise MediaError(f"Không đọc được file {path.name}. File có thể bị hỏng hoặc không phải video/âm thanh.")
    return json.loads(out.stdout or "{}")


def duration_of(path: Path) -> float:
    info = probe(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError):
        durs = [float(s["duration"]) for s in info.get("streams", []) if "duration" in s]
        if not durs:
            raise MediaError(f"Không xác định được thời lượng của {path.name}.")
        return max(durs)


def has_stream(path: Path, kind: str) -> bool:
    return any(s.get("codec_type") == kind for s in probe(path).get("streams", []))


def decode_mono(path: Path, sr: int = 22050, max_seconds: float | None = None) -> np.ndarray:
    """Giải mã âm thanh thành mảng float32 mono – không cần librosa hay soundfile."""
    require_ffmpeg()
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if out.returncode != 0 or not out.stdout:
        raise MediaError(f"Không giải mã được âm thanh trong {path.name}.")
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def safe_name(name: str, default: str = "file") -> str:
    """Tên file an toàn để lưu trên đĩa (bỏ dấu cách, ký tự lạ, đường dẫn)."""
    name = Path(name or default).name
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or default
    ext = re.sub(r"[^A-Za-z0-9.]", "", ext)[:8]
    return f"{stem[:60]}{ext.lower()}"
