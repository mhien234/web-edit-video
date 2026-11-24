"""Worker "Khử tiếng ồn".

Mặc định dùng bộ lọc afftdn (khử nhiễu theo phổ) có sẵn trong FFmpeg – không cần cài thêm gì.
Muốn chất lượng cao hơn cho giọng nói, có thể thay bằng DeepFilterNet (xem README).
"""
from __future__ import annotations

from ..jobs import JobContext
from ..media import MediaError, duration_of, has_stream, run_ffmpeg

LEVELS = {
    0: ("Nhẹ nhàng", "highpass=f=60,afftdn=nr=8:nf=-35"),
    1: ("Tiêu chuẩn", "highpass=f=80,afftdn=nr=15:nf=-30:tn=1,lowpass=f=12000"),
    2: ("Mạnh mẽ", "highpass=f=100,afftdn=nr=25:nf=-25:tn=1,lowpass=f=9000,dynaudnorm=f=200"),
}


def run(ctx: JobContext) -> dict:
    src = ctx.input_files("video_")
    if not src:
        raise MediaError("Chưa có video. Hãy bấm “Đổi video” để chọn video cần khử tiếng ồn.")
    src = src[0]
    if not has_stream(src, "audio"):
        raise MediaError("Video này không có âm thanh nên không cần khử tiếng ồn.")
    level = int(ctx.params.get("level", 1))
    name, chain = LEVELS.get(level, LEVELS[1])
    dur = duration_of(src)
    out = ctx.dir / "denoised.mp4"
    has_video = has_stream(src, "video")
    args = ["-i", str(src), "-af", chain]
    args += ["-map", "0:v:0?", "-c:v", "copy"] if has_video else []
    args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    ctx.report(0.05, f"Đang khử tiếng ồn ({name})")
    run_ffmpeg(args, duration=dur, on_progress=lambda f: ctx.report(0.05 + 0.9 * f, f"Đang khử tiếng ồn ({name})"))
    return {"file": out, "extra": {"level": name, "duration": round(dur, 2)}}
