"""Worker "Edit theo nhạc": dò beat của bài nhạc rồi cắt – ghép video đúng nhịp.

Ý tưởng lấy từ các repo đã đọc:
- composer.py (SaarD00/AI-Youtube-Shorts-Generator): cắt từng đoạn bằng FFmpeg rồi ghép,
  xuất yuv420p + faststart để mở được trên mọi trình phát.
- Phần dò beat tự viết bằng numpy (không cần librosa) để dễ cài trên Windows.
"""
from __future__ import annotations

import numpy as np

from ..jobs import JobContext
from ..media import MediaError, decode_mono, duration_of, run_ffmpeg

SR = 22050
HOP = 512
FPS_ONSET = SR / HOP


# ------------------------------------------------------------------ dò beat
def onset_envelope(y: np.ndarray) -> np.ndarray:
    """Spectral flux: độ tăng năng lượng phổ giữa hai khung liên tiếp."""
    n_fft = 1024
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // HOP
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * np.hanning(n_fft)[None, :]
    mag = np.log1p(10 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    # trừ nền trượt để bỏ phần năng lượng kéo dài
    k = 16
    base = np.convolve(flux, np.ones(k) / k, mode="same")
    env = np.maximum(0.0, flux - base)
    return env / (env.max() + 1e-9)


def estimate_period(env: np.ndarray, bpm_min: float = 70, bpm_max: float = 180) -> float:
    """Ước lượng chu kỳ beat (đơn vị: khung) bằng tự tương quan, ưu tiên quanh 120 BPM."""
    x = env - env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo = int(FPS_ONSET * 60 / bpm_max)
    hi = int(FPS_ONSET * 60 / bpm_min) + 1
    lags = np.arange(lo, min(hi, len(ac) - 1))
    if len(lags) == 0:
        return FPS_ONSET * 0.5
    bpms = 60 * FPS_ONSET / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    score = ac[lags] * prior
    i = int(np.argmax(score))
    # nội suy parabol cho chu kỳ lẻ khung
    if 0 < i < len(lags) - 1:
        a, b, c = score[i - 1], score[i], score[i + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(lags[i] + np.clip(shift, -0.5, 0.5))
    return float(lags[i])


def detect_beats(y: np.ndarray, sensitivity: int = 60) -> dict:
    env = onset_envelope(y)
    period = estimate_period(env)
    n = len(env)
    # chọn pha cho lưới beat khớp nhiều đỉnh nhất
    best_off, best_score = 0.0, -1.0
    for off in np.arange(0, period, 0.5):
        pos = np.round(np.arange(off, n, period)).astype(int)
        pos = pos[pos < n]
        s = env[pos].sum()
        if s > best_score:
            best_off, best_score = off, s
    win = max(1, int(round(period * 0.12)))
    beats, strengths = [], []
    for p in np.arange(best_off, n, period):
        c = int(round(p))
        a, b = max(0, c - win), min(n, c + win + 1)
        if a >= b:
            continue
        j = a + int(np.argmax(env[a:b]))
        beats.append(j / FPS_ONSET)
        strengths.append(float(env[j]))
    beats_arr, strengths_arr = np.array(beats), np.array(strengths)
    if len(beats_arr):
        # độ nhạy cao -> giữ nhiều beat (cả nhịp yếu); thấp -> chỉ giữ beat mạnh
        sens = int(np.clip(sensitivity, 1, 100))
        cutoff = np.percentile(strengths_arr, (100 - sens) * 0.6)
        keep = strengths_arr >= cutoff
        beats_arr, strengths_arr = beats_arr[keep], strengths_arr[keep]
    return {"bpm": round(60 * FPS_ONSET / period, 1), "beats": beats_arr.tolist(),
            "strengths": strengths_arr.tolist()}


def cut_points(beats: list[float], total: float, every: int, min_len: float = 0.3) -> list[float]:
    """Từ danh sách beat -> các mốc cắt [0, ..., total]."""
    every = max(1, int(every))
    out = [0.0]
    for i, b in enumerate(beats):
        if i % every == 0 and b - out[-1] >= min_len and total - b >= min_len:
            out.append(float(b))
    out.append(float(total))
    return out


# ------------------------------------------------------------------ dựng video
def frame_size(ratio: str, quality: str) -> tuple[int, int]:
    h = {"720p": 720, "1080p": 1080, "4K": 2160}.get(quality, 1080)
    if ratio == "9:16":
        w, h = h, int(round(h * 16 / 9))
    elif ratio == "1:1":
        w = h
    else:
        w = int(round(h * 16 / 9))
    return w - w % 2, h - h % 2


def segment_filter(W: int, H: int, L: float, fx: set[str], transition: str) -> str:
    parts = []
    if "velocity" in fx:
        parts.append("setpts=PTS/1.5")
    if "shake" in fx:
        sw, sh = int(W * 1.04) // 2 * 2, int(H * 1.04) // 2 * 2
        parts.append(f"scale={sw}:{sh}:force_original_aspect_ratio=increase")
        parts.append(f"crop={W}:{H}:x='(iw-ow)/2+10*sin(t*70)*max(0,1-t/0.3)':y='(ih-oh)/2'")
    else:
        parts.append(f"scale={W}:{H}:force_original_aspect_ratio=increase")
        parts.append(f"crop={W}:{H}")
    parts.append("setsar=1")
    parts.append(f"tpad=stop_mode=clone:stop_duration={L:.3f}")
    parts.append("fps=30")
    if "zoom" in fx:
        parts.append(f"zoompan=z='1+0.12*max(0,1-it/0.25)':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={W}x{H}:fps=30")
    if "flash" in fx:
        parts.append("eq=brightness='0.35*max(0,1-t/0.12)':eval=frame")
    if "glitch" in fx:
        parts.append("rgbashift=rh=8:bh=-8:enable='lt(t,0.15)'")
    if transition == "fade":
        parts.append("fade=t=in:st=0:d=0.15")
    parts.append("format=yuv420p")
    return ",".join(parts)


def run(ctx: JobContext) -> dict:
    p = ctx.params
    videos = ctx.input_files("video_")
    music = ctx.input_files("music_")
    if not videos:
        raise MediaError("Chưa có video. Hãy chọn ít nhất một video .mp4.")
    if not music:
        raise MediaError("Chưa có nhạc. Hãy chọn một file nhạc .mp3 hoặc .wav.")
    music = music[0]

    max_seconds = float(min(max(5, p.get("max_seconds", 60)), 300))
    ctx.report(0.02, "Đang đọc bài nhạc")
    total = min(duration_of(music), max_seconds)
    y = decode_mono(music, SR, max_seconds=total)

    ctx.report(0.06, "Đang dò beat")
    info = detect_beats(y, int(p.get("sensitivity", 60)))
    if p.get("cut_times"):
        # Website đã dò beat và người dùng có thể đã xóa bớt điểm cắt -> làm đúng như họ thấy.
        user = sorted(float(t) for t in p["cut_times"] if 0 < float(t) < total)
        cuts = [0.0]
        for t in user:
            if t - cuts[-1] >= 0.3 and total - t >= 0.3:
                cuts.append(t)
        cuts.append(total)
    else:
        cuts = cut_points(info["beats"], total, int(p.get("every", 2)))
    if len(cuts) < 2:
        raise MediaError("Không dò được nhịp trong bài nhạc này. Thử bài có tiếng trống rõ hơn.")

    W, H = frame_size(p.get("ratio", "16:9"), p.get("quality", "1080p"))
    fx = set(p.get("fx", []))
    transition = p.get("transition", "cut")
    durs = [duration_of(v) for v in videos]
    cursors = [0.0] * len(videos)
    seg_paths = []
    n_seg = len(cuts) - 1

    for i in range(n_seg):
        L = cuts[i + 1] - cuts[i]
        k = i % len(videos)
        need = L * (1.5 if "velocity" in fx else 1.0)
        start = cursors[k]
        if start + need > durs[k]:
            start = 0.0
        cursors[k] = start + need
        out = ctx.dir / f"seg_{i:04d}.mp4"
        run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{need + 0.05:.3f}", "-i", str(videos[k]), "-an",
                    "-vf", segment_filter(W, H, L, fx, transition),
                    "-t", f"{L:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", str(out)])
        seg_paths.append(out)
        ctx.report(0.1 + 0.8 * (i + 1) / n_seg, f"Đang dựng đoạn {i + 1}/{n_seg}")

    ctx.report(0.92, "Đang ghép các đoạn và chèn nhạc")
    list_file = ctx.dir / "segments.txt"
    list_file.write_text("".join(f"file '{s.name}'\n" for s in seg_paths), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", "video_only.mp4"], cwd=ctx.dir)
    result = ctx.dir / "beatcut.mp4"
    run_ffmpeg(["-i", "video_only.mp4", "-t", f"{total:.3f}", "-i", str(music),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", result.name], cwd=ctx.dir)
    for s in seg_paths:
        s.unlink(missing_ok=True)
    (ctx.dir / "video_only.mp4").unlink(missing_ok=True)

    return {"file": result, "extra": {
        "bpm": info["bpm"], "beats": len(info["beats"]), "cuts": n_seg,
        "duration": round(total, 2), "size": f"{W}x{H}",
        "cut_times": [round(c, 3) for c in cuts[1:-1]],
    }}
