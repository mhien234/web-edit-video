"""Worker "Dịch phụ đề tự động".

Hai bước, gọi riêng từ website:
1. mode="transcribe": nghe video bằng faster-whisper -> danh sách câu có thời gian,
   dịch sang ngôn ngữ đích (Google Dịch, hoặc Argos Translate offline nếu đã cài gói).
2. mode="apply": nhận phụ đề người dùng đã sửa -> gắn vào video (in chữ lên hình hoặc phụ đề rời).

faster-whisper và deep-translator là TÙY CHỌN. Website có nút “Cài ngay” gọi việc setup_ai để
máy chủ tự cài chúng (hàm setup bên dưới). Thiếu chúng thì bước 1 báo lỗi rõ ràng, còn bước 2 vẫn
chạy được vì chỉ cần FFmpeg. Argos Translate (dịch offline) vẫn được dùng nếu bạn tự cài thêm.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys

from ..jobs import ROOT, JobContext
from ..media import MediaError, decode_mono, duration_of, has_stream, run_ffmpeg

# Model nhỏ cho máy RAM 8GB. Đổi bằng biến môi trường WHISPER_MODEL=base / medium ...
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
_model = None


def _has(module: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def _argos_langs() -> list[str]:
    if not _has("argostranslate"):
        return []
    try:
        from argostranslate import translate
        return sorted({lang.code for lang in translate.get_installed_languages()})
    except Exception:
        return []


def ai_status() -> dict:
    whisper = _has("faster_whisper")
    translator = "argos" if _argos_langs() else "google"     # Google Dịch qua gtx không cần thư viện
    return {"whisper": whisper, "whisper_model": WHISPER_MODEL, "translator": translator,
            "translate_languages": _argos_langs(), "ai_ready": whisper and translator is not None}


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise MediaError("Chưa cài AI phụ đề. Bấm “Tạo phụ đề bằng AI” rồi chọn “Cài ngay”.") from e
        # int8 trên CPU: nhẹ, chạy được trên máy không có card đồ họa rời
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def _translate_argos(texts: list[str], src: str, dst: str) -> list[str] | None:
    if not _has("argostranslate"):
        return None
    try:
        from argostranslate import translate
        installed = {lang.code: lang for lang in translate.get_installed_languages()}
        a, b = installed.get(src), installed.get(dst)
        if not (a and b):
            return None
        tr = a.get_translation(b)
        if tr is None and "en" in installed:   # dịch bắc cầu qua tiếng Anh
            t1, t2 = a.get_translation(installed["en"]), installed["en"].get_translation(b)
            if t1 and t2:
                return [t2.translate(t1.translate(t)) for t in texts]
        return [tr.translate(t) for t in texts] if tr else None
    except Exception:
        return None


def _translate_google(texts: list[str], src: str, dst: str) -> list[str] | None:
    if not _has("deep_translator"):
        return None
    from deep_translator import GoogleTranslator
    source = GOOGLE_CODE.get(src, src) if src and src != "auto" else "auto"
    try:
        tr = GoogleTranslator(source=source, target=dst)
    except Exception:                       # mã ngôn ngữ nguồn lạ -> để Google tự nhận diện
        tr = GoogleTranslator(source="auto", target=dst)
    out = []
    for i in range(0, len(texts), 40):             # gửi từng nhóm cho nhanh và đỡ bị chặn
        out.extend(tr.translate_batch(texts[i:i + 40]))
    return [o if o else t for o, t in zip(out, texts)]


def _gtx_request(text: str, src: str, dst: str) -> str:
    """Gọi Google Dịch qua endpoint công khai translate.googleapis.com (client=gtx)."""
    import json
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    params = urllib.parse.urlencode({"client": "gtx", "sl": src or "auto", "tl": dst, "dt": "t"})
    data = urllib.parse.urlencode({"q": text}).encode()
    for attempt in range(4):
        req = urllib.request.Request(f"https://translate.googleapis.com/translate_a/single?{params}", data=data,
                                     headers={"User-Agent": "Mozilla/5.0",
                                              "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                res = json.loads(r.read().decode("utf-8"))
            return "".join(part[0] for part in (res[0] or []) if part and part[0])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:      # bị giới hạn tạm thời -> chờ rồi thử lại
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Google Dịch không phản hồi")


def _translate_gtx(texts: list[str], src: str, dst: str) -> list[str]:
    """Dịch theo từng nhóm câu (nối bằng xuống dòng) để gửi ít yêu cầu nhất."""
    out: list[str] = []
    batch: list[str] = []

    def flush():
        if not batch:
            return
        joined = "\n".join(t.replace("\n", " ") for t in batch)
        lines = _gtx_request(joined, src, dst).split("\n")
        if len(lines) != len(batch):              # Google gộp/tách dòng -> dịch lẻ từng câu
            lines = [_gtx_request(t, src, dst) for t in batch]
        out.extend(x.strip() for x in lines)
        batch.clear()

    size = 0
    for t in texts:
        if batch and (size + len(t) > 3500 or len(batch) >= 60):
            flush()
            size = 0
        batch.append(t)
        size += len(t) + 1
    flush()
    return [o if o else t for o, t in zip(out, texts)]


def _translate(texts: list[str], src: str, dst: str) -> tuple[list[str], str | None]:
    """Dịch sang MỘT ngôn ngữ (dùng khi thử dịch lúc cài AI)."""
    if not texts or src == dst:
        return texts, None
    out, notes = translate_all(texts, src, [dst])
    if dst in out:
        return out[dst], None
    return texts, notes.get(dst, "Chưa dịch được.") + " Phụ đề đang ở ngôn ngữ gốc."


# Mã ngôn ngữ mà website gửi lên -> mã của Google Dịch / Argos
GOOGLE_CODE = {"zh": "zh-CN", "zh-CN": "zh-CN", "zh-TW": "zh-TW"}
ARGOS_CODE = {"zh-CN": "zh", "zh-TW": "zt"}


def translate_all(texts: list[str], src: str, targets: list[str], report=None) -> tuple[dict, dict]:
    """Dịch một danh sách câu sang nhiều ngôn ngữ. Trả về ({lang: [câu]}, {lang: ghi chú lỗi})."""
    out, notes = {}, {}
    targets = [t for t in dict.fromkeys(targets) if t]          # bỏ trùng, giữ thứ tự
    for i, lang in enumerate(targets):
        if report:
            report(i / max(1, len(targets)), lang)
        if lang.split("-")[0] == (src or "").split("-")[0]:
            out[lang] = list(texts)
            continue
        res = _translate_argos(texts, ARGOS_CODE.get(src, src), ARGOS_CODE.get(lang, lang))
        errors = []
        if not res:
            for engine in (_translate_gtx, _translate_google):   # 2 đường vào Google Dịch
                try:
                    res = engine(texts, GOOGLE_CODE.get(src, src), GOOGLE_CODE.get(lang, lang))
                    if res:
                        break
                except Exception as e:
                    errors.append(f"{engine.__name__.replace('_translate_', '')}: {type(e).__name__}")
            if not res and errors:
                notes[lang] = f"Chưa dịch được sang {lang} ({', '.join(errors)}) – kiểm tra mạng rồi thử lại."
        if res:
            out[lang] = res
        elif lang not in notes:
            notes[lang] = "Chưa có công cụ dịch – hãy cài AI phụ đề."
    return out, notes


def setup(ctx: JobContext) -> dict:
    """Cài AI phụ đề ngay từ website: thư viện -> model Whisper -> thử dịch."""
    req = ROOT / "requirements-ai.txt"
    ctx.report(0.02, "Đang cài thư viện AI (faster-whisper, deep-translator)...")
    proc = subprocess.Popen([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                             "-r", str(req)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert proc.stdout is not None
    log, steps = [], 0
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        log.append(line)
        if line.startswith(("Collecting", "Downloading", "Installing", "Requirement already")):
            steps += 1
            short = line.split(" (")[0][:80]
            ctx.report(min(0.5, 0.03 + steps * 0.02), f"Cài thư viện: {short}")
    (ctx.dir / "pip.log").write_text("\n".join(log), encoding="utf-8")
    if proc.wait() != 0:
        tail = " | ".join(log[-2:]) or "không rõ lỗi"
        raise MediaError(f"Cài thư viện thất bại (máy có mạng không?): {tail[:300]}")
    importlib.invalidate_caches()

    ctx.report(0.55, f"Đang tải model Whisper “{WHISPER_MODEL}” (khoảng 480 MB) – lần đầu mất vài phút...")
    try:
        _get_model()
    except MediaError:
        raise
    except Exception as e:
        raise MediaError(f"Không tải được model Whisper: {e}. Kiểm tra mạng rồi bấm cài lại.") from e

    ctx.report(0.95, "Đang thử dịch một câu")
    sample, note = _translate(["Hello, welcome to Creative Motion Lab!"], "en", "vi")
    return {"file": None, "extra": {**ai_status(), "sample": sample[0], "note": note}}


def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[list]) -> str:
    lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(f"{i}\n{srt_time(float(start))} --> {srt_time(float(end))}\n{str(text).strip()}\n")
    return "\n".join(lines)


def transcribe(ctx: JobContext, src) -> dict:
    """Nghe video -> phụ đề GỐC; nếu có danh sách ngôn ngữ đích thì dịch luôn."""
    targets = ctx.params.get("targets") or ([ctx.params["lang"]] if ctx.params.get("lang") else [])
    ctx.report(0.05, "Đang tách âm thanh")
    audio = decode_mono(src, 16000)
    ctx.report(0.15, f"Đang nhận diện giọng nói (model {WHISPER_MODEL}, lần đầu sẽ tải model)")
    model = _get_model()
    seg_iter, info = model.transcribe(audio, vad_filter=True, beam_size=1)
    total = max(1.0, len(audio) / 16000)
    original = []
    for s_ in seg_iter:
        original.append([round(s_.start, 3), round(s_.end, 3), s_.text.strip()])
        ctx.report(0.15 + 0.65 * min(1.0, s_.end / total), f"Đã nghe được {len(original)} câu")
    if not original:
        raise MediaError("Không nghe thấy lời nói nào trong video.")
    texts = [x[2] for x in original]
    translations, notes = translate_all(
        texts, info.language, targets,
        report=lambda f, lang: ctx.report(0.82 + 0.15 * f, f"Đang dịch sang {lang}"))
    (ctx.dir / f"subtitles.{info.language}.srt").write_text(to_srt(original), encoding="utf-8")
    tracks = {}
    for lang, tr in translations.items():
        tracks[lang] = [[a, b, t] for (a, b, _), t in zip(original, tr)]
        (ctx.dir / f"subtitles.{lang}.srt").write_text(to_srt(tracks[lang]), encoding="utf-8")
    first = next(iter(tracks.values()), original)
    note = "; ".join(notes.values()) or None
    return {"file": None, "extra": {"original": original, "source_lang": info.language,
                                    "tracks": tracks, "notes": notes, "note": note,
                                    "segments": first}}   # "segments": tương thích bản cũ


def translate(ctx: JobContext) -> dict:
    """Dịch phụ đề có sẵn (người dùng đã sửa) sang một hay nhiều ngôn ngữ – không cần video."""
    segs = ctx.params.get("segments") or []
    targets = ctx.params.get("targets") or []
    if not segs:
        raise MediaError("Chưa có phụ đề gốc để dịch.")
    if not targets:
        raise MediaError("Hãy chọn ít nhất một ngôn ngữ cần dịch.")
    texts = [str(x[2]) for x in segs]
    translations, notes = translate_all(
        texts, ctx.params.get("source") or "auto", targets,
        report=lambda f, lang: ctx.report(0.05 + 0.9 * f, f"Đang dịch sang {lang}"))
    if not translations:
        raise MediaError(next(iter(notes.values()), "Không dịch được."))
    tracks = {lang: [[x[0], x[1], t] for x, t in zip(segs, tr)] for lang, tr in translations.items()}
    return {"file": None, "extra": {"tracks": tracks, "notes": notes,
                                    "note": "; ".join(notes.values()) or None}}


def apply(ctx: JobContext, src) -> dict:
    segments = ctx.params.get("segments") or []
    if not segments:
        raise MediaError("Danh sách phụ đề đang trống.")
    if not has_stream(src, "video"):
        raise MediaError("File này không có hình ảnh để gắn phụ đề.")
    (ctx.dir / "subs.srt").write_text(to_srt(segments), encoding="utf-8")
    out = ctx.dir / "subtitled.mp4"
    dur = duration_of(src)
    if ctx.params.get("burn", True):
        # In chữ lên hình: chạy trong thư mục việc và gọi tên file tương đối
        # để tránh lỗi đường dẫn có ổ đĩa "D:\" trên Windows.
        style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=28"
        args = ["-i", str(src), "-vf", f"subtitles=subs.srt:force_style='{style}'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy",
                "-movflags", "+faststart", out.name]
    else:
        args = ["-i", str(src), "-i", "subs.srt", "-map", "0:v", "-map", "0:a?", "-map", "1:0",
                "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={ctx.params.get('lang', 'vi')}", out.name]
    run_ffmpeg(args, duration=dur, cwd=ctx.dir,
               on_progress=lambda f: ctx.report(0.05 + 0.9 * f, "Đang gắn phụ đề vào video"))
    return {"file": out, "extra": {"lines": len(segments), "burned": bool(ctx.params.get("burn", True))}}


def run(ctx: JobContext) -> dict:
    if ctx.params.get("mode") == "translate":
        return translate(ctx)
    src = ctx.input_files("video_")
    if not src:
        raise MediaError("Chưa có video. Hãy bấm “Đổi video” để chọn video.")
    src = src[0]
    if ctx.params.get("mode") == "transcribe":
        if not has_stream(src, "audio"):
            raise MediaError("Video này không có âm thanh nên không tạo được phụ đề.")
        return transcribe(ctx, src)
    return apply(ctx, src)
