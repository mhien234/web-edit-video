"""Worker "Giọng đọc AI" (lồng tiếng theo phụ đề).

Nhận danh sách phụ đề [[bắt đầu, kết thúc, chữ], ...] của MỘT ngôn ngữ, đọc từng câu bằng
giọng AI rồi đặt đúng thời điểm trên dòng thời gian. Câu nào đọc dài hơn chỗ trống thì được
tăng tốc nhẹ (tối đa 1.6 lần). Cuối cùng trộn với âm thanh gốc: giọng gốc được vặn nhỏ và tự
"nhường" (ducking) mỗi khi giọng AI cất lên.

Bộ máy đọc (engine) tách riêng để sau này thêm VieNeu (offline) hay ElevenLabs mà không phải sửa
phần ghép âm thanh. Hiện có: "edge" – giọng Microsoft Edge, miễn phí, không cần khóa, cần mạng.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import media
from ..jobs import JobContext
from ..media import MediaError, duration_of, has_stream, run_ffmpeg

SR = 24000            # tần số mẫu của giọng đọc
MAX_SPEEDUP = 1.6     # nhanh hơn nữa sẽ khó nghe
PARALLEL = 3          # số câu đọc cùng lúc (nhiều hơn dễ bị Microsoft chặn)
MAX_FAIL_RATIO = 0.3  # quá 30% số câu đọc lỗi thì báo lỗi cả việc

# 10 ngôn ngữ, mỗi ngôn ngữ một giọng nữ và một giọng nam (giọng đầu tiên là mặc định).
VOICES: dict[str, list[dict]] = {
    "vi": [{"id": "vi-VN-HoaiMyNeural", "name": "Hoài My", "gender": "Nữ"},
           {"id": "vi-VN-NamMinhNeural", "name": "Nam Minh", "gender": "Nam"}],
    "en": [{"id": "en-US-JennyNeural", "name": "Jenny (Mỹ)", "gender": "Nữ"},
           {"id": "en-US-GuyNeural", "name": "Guy (Mỹ)", "gender": "Nam"}],
    "ja": [{"id": "ja-JP-NanamiNeural", "name": "Nanami", "gender": "Nữ"},
           {"id": "ja-JP-KeitaNeural", "name": "Keita", "gender": "Nam"}],
    "ko": [{"id": "ko-KR-SunHiNeural", "name": "SunHi", "gender": "Nữ"},
           {"id": "ko-KR-InJoonNeural", "name": "InJoon", "gender": "Nam"}],
    "zh-CN": [{"id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao", "gender": "Nữ"},
              {"id": "zh-CN-YunxiNeural", "name": "Yunxi", "gender": "Nam"}],
    "th": [{"id": "th-TH-PremwadeeNeural", "name": "Premwadee", "gender": "Nữ"},
           {"id": "th-TH-NiwatNeural", "name": "Niwat", "gender": "Nam"}],
    "fr": [{"id": "fr-FR-DeniseNeural", "name": "Denise", "gender": "Nữ"},
           {"id": "fr-FR-HenriNeural", "name": "Henri", "gender": "Nam"}],
    "es": [{"id": "es-ES-ElviraNeural", "name": "Elvira", "gender": "Nữ"},
           {"id": "es-ES-AlvaroNeural", "name": "Álvaro", "gender": "Nam"}],
    "de": [{"id": "de-DE-KatjaNeural", "name": "Katja", "gender": "Nữ"},
           {"id": "de-DE-ConradNeural", "name": "Conrad", "gender": "Nam"}],
    "ru": [{"id": "ru-RU-SvetlanaNeural", "name": "Svetlana", "gender": "Nữ"},
           {"id": "ru-RU-DmitryNeural", "name": "Dmitry", "gender": "Nam"}],
}
LANG_ALIAS = {"zh": "zh-CN", "zh-cn": "zh-CN", "en-us": "en", "en-gb": "en"}


def tts_status() -> dict:
    importlib.invalidate_caches()
    return {"tts": importlib.util.find_spec("edge_tts") is not None,
            "tts_languages": list(VOICES)}


def pick_voice(lang: str, voice: str | None) -> str:
    lang = LANG_ALIAS.get((lang or "").lower(), lang)
    options = VOICES.get(lang)
    if not options:
        raise MediaError(f"Giọng đọc AI chưa hỗ trợ ngôn ngữ “{lang}”. Hiện có: {', '.join(VOICES)}.")
    ids = [v["id"] for v in options]
    return voice if voice in ids else ids[0]


# ---------------------------------------------------------------- engines
class EdgeEngine:
    name = "edge"

    def __init__(self):
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise MediaError("Chưa cài giọng đọc AI. Bấm “Cài giọng đọc” trong mục Dịch phụ đề.") from e

    def synth(self, text: str, voice: str, out: Path) -> None:
        import edge_tts

        async def go():
            await edge_tts.Communicate(text, voice).save(str(out))

        last = None
        # Dịch vụ của Microsoft thỉnh thoảng trả về rỗng ("No audio was received") -> thử lại, chờ lâu dần
        for attempt in range(5):
            try:
                asyncio.run(go())
                if out.exists() and out.stat().st_size > 0:
                    return
            except Exception as e:
                last = e
            time.sleep(min(8.0, 1.5 * 2 ** attempt))
        raise MediaError(f"Không tạo được giọng đọc (máy có mạng không?): {last or 'file rỗng'}")


ENGINES = {"edge": EdgeEngine}


def get_engine(name: str | None):
    cls = ENGINES.get(name or "edge")
    if cls is None:
        raise MediaError(f"Không có bộ máy đọc “{name}”.")
    return cls()


# ---------------------------------------------------------------- audio helpers
def decode_clip(path: Path, tempo: float = 1.0) -> np.ndarray:
    media.require_ffmpeg()
    cmd = [media.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path)]
    if abs(tempo - 1.0) > 0.01:
        cmd += ["-af", f"atempo={tempo:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0 or not out.stdout:
        raise MediaError(f"Không đọc được file giọng {path.name}.")
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def trim_silence(y: np.ndarray, thr: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(y) > thr)
    if idx.size == 0:
        return y[:0]
    pad = int(0.03 * SR)
    return y[max(0, idx[0] - pad): idx[-1] + pad]


def write_wav(path: Path, y: np.ndarray) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def clean_text(text: str) -> str:
    # Phụ đề song ngữ có dạng "gốc\nbản dịch": chỉ đọc dòng cuối (bản dịch)
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------- main
def build_track(ctx: JobContext, segments: list, voice: str, engine, total: float | None) -> tuple[Path, dict]:
    segs = sorted(([float(a), float(b), clean_text(t)] for a, b, t in segments), key=lambda s: s[0])
    segs = [s for s in segs if s[2]]
    if not segs:
        raise MediaError("Phụ đề không có câu nào để đọc.")
    end_all = max(total or 0.0, segs[-1][1] + 1.0)
    clips_dir = ctx.dir / "voice"
    clips_dir.mkdir(exist_ok=True)
    placed: list[tuple[int, np.ndarray]] = []
    sped = 0
    failed: list[int] = []
    done = 0

    def synth_one(i: int):
        clip = clips_dir / f"{i:04d}.mp3"
        try:
            engine.synth(segs[i][2], voice, clip)
            return i, clip, None
        except MediaError as e:
            return i, clip, e

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        results = {}
        for i, clip, err in pool.map(synth_one, range(len(segs))):
            done += 1
            ctx.report(0.05 + 0.75 * done / len(segs), f"Đang đọc câu {done}/{len(segs)}"
                       + (f" ({len(failed) + (err is not None)} câu lỗi, sẽ bỏ qua)" if failed or err else ""))
            if err is not None:
                failed.append(i)
                continue
            results[i] = clip
    if not results or len(failed) > max(1, int(len(segs) * MAX_FAIL_RATIO)):
        raise MediaError(f"Dịch vụ giọng đọc lỗi ở {len(failed)}/{len(segs)} câu – kiểm tra mạng rồi thử lại sau ít phút.")

    for i in sorted(results):
        start, end, _ = segs[i]
        clip = results[i]
        y = trim_silence(decode_clip(clip))
        if y.size == 0:
            continue
        nxt = segs[i + 1][0] if i + 1 < len(segs) else end_all
        slot = max(0.3, max(end, nxt) - start - 0.05)   # được đọc lấn sang khoảng lặng trước câu sau
        if y.size / SR > slot:
            y = trim_silence(decode_clip(clip, min(MAX_SPEEDUP, y.size / SR / slot)))
            sped += 1
        placed.append((int(start * SR), y))
    ctx.report(0.82, "Đang ghép các câu theo thời gian")
    length = max(int(end_all * SR), max(p + y.size for p, y in placed) if placed else 0)
    track = np.zeros(length, dtype=np.float32)
    for pos, y in placed:
        track[pos:pos + y.size] += y[: max(0, length - pos)]
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track *= 0.98 / peak
    out = ctx.dir / "voiceover.wav"
    write_wav(out, track)
    return out, {"lines": len(segs), "spoken": len(placed), "sped_up": sped,
                 "skipped": [[segs[i][0], segs[i][2]] for i in failed]}


def run(ctx: JobContext) -> dict:
    p = ctx.params
    voice = pick_voice(p.get("lang", "vi"), p.get("voice"))
    engine = get_engine(p.get("engine"))
    orig = float(p.get("orig_volume", 0.25))
    orig = min(1.0, max(0.0, orig))

    src = ctx.input_files("video_")
    src = src[0] if src else None
    total = duration_of(src) if src else None
    voice_wav, info = build_track(ctx, p.get("segments") or [], voice, engine, total)
    info.update({"voice": voice, "engine": engine.name, "orig_volume": orig})

    if src is None:     # không có video -> trả về file giọng đọc riêng
        out = ctx.dir / "giong-doc.m4a"
        run_ffmpeg(["-i", voice_wav.name, "-c:a", "aac", "-b:a", "160k", out.name], cwd=ctx.dir)
        return {"file": out, "extra": info}

    out = ctx.dir / "long-tieng.mp4"
    if has_stream(src, "audio") and orig > 0:
        # giọng gốc nhỏ lại + tự nhường khi giọng AI cất lên, rồi trộn với giọng AI
        fc = (f"[1:a]aresample=48000,asplit=2[vo][key];"
              f"[0:a]aresample=48000,volume={orig:.2f}[bg];"
              f"[bg][key]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=350[duck];"
              f"[duck][vo]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        fc = "[1:a]aresample=48000[a]"
    args = ["-i", str(src), "-i", voice_wav.name, "-filter_complex", fc,
            "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", out.name]
    if not has_stream(src, "video"):
        raise MediaError("File này không có hình ảnh để lồng tiếng.")
    run_ffmpeg(args, duration=total, cwd=ctx.dir,
               on_progress=lambda f: ctx.report(0.85 + 0.14 * f, "Đang trộn giọng đọc vào video"))
    return {"file": out, "extra": info}


def setup(ctx: JobContext) -> dict:
    """Cài thư viện giọng đọc (edge-tts, khoảng 1 MB) rồi thử đọc một câu."""
    ctx.report(0.05, "Đang cài thư viện giọng đọc edge-tts...")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                           "edge-tts>=6.1"], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (ctx.dir / "pip.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["không rõ lỗi"]
        raise MediaError(f"Cài edge-tts thất bại (máy có mạng không?): {tail[0][:300]}")
    importlib.invalidate_caches()
    ctx.report(0.6, "Đang thử đọc một câu tiếng Việt")
    test = ctx.dir / "thu-giong.mp3"
    get_engine("edge").synth("Xin chào, đây là Creative Motion Lab.", VOICES["vi"][0]["id"], test)
    return {"file": test, "extra": tts_status()}
