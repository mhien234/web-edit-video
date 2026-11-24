"""Worker "Xóa nền": tách người ra khỏi nền video, không cần phông xanh.

Dùng model **Robust Video Matting** (RVM, ResNet-50) chạy bằng onnxruntime trên CPU.

Vì sao chọn RVM thay vì tách nền từng ảnh (như MediaPipe Selfie Segmentation):
RVM giữ một "trí nhớ" (recurrent state r1..r4) truyền từ khung hình này sang khung sau,
nên viền người không rung giật giữa các khung – thứ thấy rõ ngay khi xem video.
Đổi lại, các khung phải chạy LẦN LƯỢT, không nhảy cóc được.

Model là file .onnx tải riêng (khoảng 100 MB), giấy phép GPL-3.0 – xem mục Giấy phép
trong README. Website không kèm sẵn model; hàm `find_model` bên dưới dò vài chỗ quen thuộc.

onnxruntime là TÙY CHỌN, giống faster-whisper ở worker phụ đề: thiếu thì báo lỗi rõ ràng
và website có nút gọi việc `setup_rembg` để máy chủ tự cài (hàm `setup` bên dưới).

Ba kiểu nền đầu ra:
- `color`: nền một màu (mặc định xanh lá chuẩn chroma key), xuất MP4 mở được mọi nơi.
- `blur`: giữ chính khung hình gốc làm nền nhưng làm mờ – kiểu hay dùng khi quay vlog.
- `transparent`: giữ kênh trong suốt, xuất ProRes 4444 trong file .mov.

Vì sao `transparent` KHÔNG dùng WebM: đã thử và đo, FFmpeg **giải mã** được WebM có alpha
nhưng **không ghi** được – cả libvpx (VP8) lẫn libvpx-vp9 đều nhận `-pix_fmt yuva420p` rồi
lặng lẽ xuất ra yuv420p đặc, mất sạch alpha (bộ ghép Matroska của FFmpeg không viết track
alpha phụ). ProRes 4444 giữ đúng alpha và là định dạng Premiere / Resolve / Final Cut đều
đọc được, hợp với đường xuất FCPXML sẵn có của dự án. Đổi lại file nặng hơn nhiều.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import media
from ..jobs import ROOT, JobContext
from ..media import MediaError, duration_of, has_stream, probe, require_ffmpeg

MODEL_NAME = "rvm_resnet50_fp32.onnx"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Nền mặc định: xanh lá chuẩn phông chroma key, để ai muốn tự key lại trong phần mềm khác.
DEFAULT_COLOR = "#00B140"
# Quá rộng thì CPU không kham nổi: thu nhỏ trước khi chạy model, giữ nguyên tỉ lệ khung.
DEFAULT_MAX_WIDTH = 1280
_session = None
_session_path: Path | None = None


# ------------------------------------------------------------------ tìm model
def find_model() -> Path | None:
    """Dò model theo thứ tự: biến môi trường -> thư mục dự án -> gói model của Vidlab."""
    env = os.environ.get("CML_RVM_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    candidates = [
        ROOT / "models" / MODEL_NAME,
        Path(os.environ.get("CML_DATA_DIR") or ROOT / "data") / "models" / MODEL_NAME,
    ]
    # Ai đã cài Vidlab trên cùng máy thì model nằm sẵn ở thư mục dữ liệu của nó.
    openchatcut = os.environ.get("OPENCHATCUT_DATA_DIR")
    if openchatcut:
        candidates.append(Path(openchatcut) / "models" / MODEL_NAME)
    candidates.append(Path.home() / ".openchatcut" / "models" / MODEL_NAME)
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def rembg_status() -> dict:
    """Cho /api/health biết xóa nền đã dùng được chưa (giống ai_status của worker phụ đề)."""
    model = find_model()
    return {
        "rembg_lib": importlib.util.find_spec("onnxruntime") is not None,
        "rembg_model": bool(model),
        "rembg_ready": importlib.util.find_spec("onnxruntime") is not None and bool(model),
    }


def _load_session():
    """Nạp model một lần rồi dùng lại – nạp lại mỗi việc thì mất thêm vài giây."""
    global _session, _session_path
    if importlib.util.find_spec("onnxruntime") is None:
        raise MediaError(
            "Chưa cài thư viện onnxruntime nên chưa xóa nền được. "
            "Bấm nút “Cài ngay” ở mục Xóa nền để máy chủ tự cài (khoảng 60 MB)."
        )
    path = find_model()
    if path is None:
        raise MediaError(
            f"Chưa có model xóa nền ({MODEL_NAME}, khoảng 100 MB). "
            f"Hãy đặt file đó vào thư mục “models” của dự án, "
            f"hoặc trỏ biến môi trường CML_RVM_MODEL tới chỗ bạn đã lưu."
        )
    if _session is not None and _session_path == path:
        return _session
    import onnxruntime as ort  # nhập muộn: chỉ cần khi thật sự xóa nền

    opts = ort.SessionOptions()
    # Máy cá nhân còn phải chạy trình duyệt: để onnxruntime tự chọn số luồng vừa phải.
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    _session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    _session_path = path
    return _session


# ------------------------------------------------------------------ tiện ích
def _video_info(src: Path) -> tuple[int, int, float, int]:
    """Trả về (rộng, cao, fps, tổng số khung ước lượng)."""
    info = probe(src)
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise MediaError("File này không có hình ảnh nên không xóa nền được.")
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if w <= 0 or h <= 0:
        raise MediaError("Không đọc được kích thước khung hình của video.")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    try:
        num, den = (float(x) for x in rate.split("/"))
        fps = num / den if den else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    if not 1.0 <= fps <= 240.0:
        fps = 30.0
    dur = duration_of(src)
    return w, h, fps, max(1, int(round(dur * fps)))


def _fit(w: int, h: int, max_width: int) -> tuple[int, int]:
    """Thu nhỏ về bề ngang tối đa, ép chẵn vì bộ mã hóa yuv420p đòi số chẵn."""
    if w <= max_width:
        return w - (w % 2), h - (h % 2)
    scale = max_width / w
    return max_width - (max_width % 2), max(2, int(round(h * scale)) & ~1)


def _downsample_ratio(w: int, h: int) -> float:
    """Tỉ lệ RVM khuyến nghị: nội bộ model chạy ở độ phân giải thấp hơn cho nhanh."""
    long_side = max(w, h)
    if long_side >= 1920:
        return 0.25
    if long_side >= 1280:
        return 0.4
    if long_side >= 720:
        return 0.6
    return 1.0


def _alpha_encoder() -> tuple[list[str], str, str]:
    """Chọn bộ mã hóa giữ được alpha trong FFmpeg đang có. Trả về (tham số, đuôi file, tên)."""
    require_ffmpeg()
    try:
        listing = subprocess.run([media.FFMPEG, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", creationflags=NO_WINDOW).stdout or ""
    except OSError:
        listing = ""
    if "prores_ks" in listing:
        return (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"],
                ".mov", "ProRes 4444")
    if "qtrle" in listing:
        return ["-c:v", "qtrle", "-pix_fmt", "argb"], ".mov", "QuickTime RLE"
    raise MediaError(
        "FFmpeg trên máy không có bộ mã hóa nào giữ được nền trong suốt "
        "(cần prores_ks hoặc qtrle). Hãy chọn kiểu nền “một màu” rồi tự tách màu sau."
    )


def _parse_color(value: str | None) -> tuple[int, int, int]:
    text = (value or DEFAULT_COLOR).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        text = DEFAULT_COLOR.lstrip("#")
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return 0x00, 0xB1, 0x40


def _box_blur(frame: np.ndarray, radius: int) -> np.ndarray:
    """Làm mờ bằng ảnh tích phân (summed-area table) – không cần OpenCV hay Pillow."""
    if radius < 1:
        return frame
    img = frame.astype(np.float32)
    pad = np.pad(img, ((radius + 1, radius), (radius + 1, radius), (0, 0)), mode="edge")
    integral = pad.cumsum(axis=0).cumsum(axis=1)
    h, w = frame.shape[:2]
    size = 2 * radius + 1
    y0, x0 = np.arange(h), np.arange(w)
    bottom, right = y0 + size, x0 + size
    total = (integral[bottom[:, None], right[None, :]]
             - integral[y0[:, None], right[None, :]]
             - integral[bottom[:, None], x0[None, :]]
             + integral[y0[:, None], x0[None, :]])
    return (total / (size * size)).astype(np.float32)


# ------------------------------------------------------------------ việc chính
def run(ctx: JobContext) -> dict:
    src_files = ctx.input_files("video_")
    if not src_files:
        raise MediaError("Chưa có video. Hãy bấm “Đổi video” để chọn video cần xóa nền.")
    src = src_files[0]
    require_ffmpeg()
    session = _load_session()

    mode = str(ctx.params.get("mode", "color")).lower()
    if mode not in {"transparent", "color", "blur"}:
        mode = "color"
    max_width = int(ctx.params.get("max_width") or DEFAULT_MAX_WIDTH)
    max_width = max(320, min(1920, max_width))

    src_w, src_h, fps, total_frames = _video_info(src)
    w, h = _fit(src_w, src_h, max_width)
    ratio = np.array([_downsample_ratio(w, h)], dtype=np.float32)
    bg_rgb = np.array(_parse_color(ctx.params.get("color")), dtype=np.float32).reshape(3, 1, 1) / 255.0
    blur_radius = max(1, int(ctx.params.get("blur", 12)))
    keep_audio = has_stream(src, "audio")

    if mode == "transparent":
        alpha_args, suffix, codec_name = _alpha_encoder()
    else:
        alpha_args, suffix, codec_name = [], ".mp4", "H.264"
    out = ctx.dir / f"removed-bg{suffix}"
    pix_in = "rgba" if mode == "transparent" else "rgb24"

    decode = [media.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src),
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-vf", f"scale={w}:{h}", "-"]
    encode = [media.FFMPEG, "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
              "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
    if keep_audio:
        encode += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    if mode == "transparent":
        encode += alpha_args
    else:
        encode += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                   "-movflags", "+faststart"]
    encode.append(str(out))

    frame_bytes = w * h * 3
    rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
    names = ["r1i", "r2i", "r3i", "r4i"]
    done = 0
    started = time.monotonic()

    ctx.report(0.02, f"Đang chuẩn bị xóa nền ({w}×{h}, khoảng {total_frames} khung hình)")
    dec = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    enc = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    assert dec.stdout is not None and enc.stdin is not None
    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            src_chw = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            feeds = {"src": src_chw, "downsample_ratio": ratio}
            feeds.update(dict(zip(names, rec)))
            fgr, pha, *rec = session.run(None, feeds)

            alpha = pha[0]                      # (1, h, w) – 0 nền, 1 người
            person = fgr[0]                     # (3, h, w) – màu người đã tách khỏi nền
            if mode == "transparent":
                rgb = np.clip(person, 0.0, 1.0)
                rgba = np.concatenate([rgb, np.clip(alpha, 0.0, 1.0)], axis=0)
                out_frame = (rgba.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
            else:
                if mode == "blur":
                    background = _box_blur(frame.astype(np.float32) / 255.0, blur_radius).transpose(2, 0, 1)
                else:
                    background = bg_rgb
                mixed = person * alpha + background * (1.0 - alpha)
                out_frame = (np.clip(mixed, 0.0, 1.0).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
            enc.stdin.write(out_frame.tobytes())

            done += 1
            if done % 5 == 0 or done == total_frames:
                # Xóa nền chạy CPU nên khá lâu: báo luôn còn bao nhiêu phút để người dùng yên tâm.
                elapsed = time.monotonic() - started
                left = (total_frames - done) * elapsed / done if done else 0.0
                eta = f" – còn khoảng {left / 60:.0f} phút" if left > 90 else ""
                ctx.report(0.02 + 0.95 * min(1.0, done / total_frames),
                           f"Đang xóa nền: khung {done}/{total_frames}{eta}")
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.stdout.close()
        dec_err = dec.stderr.read().decode("utf-8", "replace") if dec.stderr else ""
        enc_err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
        dec.wait()
        enc_code = enc.wait()

    if done == 0:
        raise MediaError("Không đọc được khung hình nào từ video. " + (dec_err.strip()[-200:] or ""))
    if enc_code != 0:
        raise MediaError("Không ghi được video kết quả: " + (enc_err.strip()[-200:] or "FFmpeg báo lỗi"))

    label = {"transparent": "nền trong suốt", "color": "nền một màu", "blur": "nền mờ"}[mode]
    return {"file": out, "extra": {
        "mode": mode, "kieu_nen": label, "frames": done,
        "size": f"{w}x{h}", "fps": round(fps, 2),
        "dinh_dang": codec_name,
        "dung_luong_mb": round(out.stat().st_size / (1024 * 1024), 1),
        "co_tieng": keep_audio,
    }}


# ------------------------------------------------------------------ cài đặt
def setup(ctx: JobContext) -> dict:
    """Cài onnxruntime ngay từ website (nút “Cài ngay” ở mục Xóa nền)."""
    ctx.report(0.05, "Đang cài onnxruntime (khoảng 60 MB)...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "onnxruntime>=1.17"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
    )
    assert proc.stdout is not None
    log, steps = [], 0
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        log.append(line)
        if line.startswith(("Collecting", "Downloading", "Installing", "Requirement already")):
            steps += 1
            ctx.report(min(0.8, 0.08 + steps * 0.08), f"Cài thư viện: {line.split(' (')[0][:80]}")
    (ctx.dir / "pip.log").write_text("\n".join(log), encoding="utf-8")
    if proc.wait() != 0:
        raise MediaError("Cài onnxruntime thất bại (máy có mạng không?): " + (" | ".join(log[-2:])[:300]))
    importlib.invalidate_caches()

    model = find_model()
    ctx.report(1.0, "Đã cài xong onnxruntime")
    return {"file": None, "extra": {
        "onnxruntime": "đã cài",
        "model": str(model) if model else "CHƯA CÓ – hãy đặt rvm_resnet50_fp32.onnx vào thư mục models/",
        "san_sang": bool(model),
    }}
