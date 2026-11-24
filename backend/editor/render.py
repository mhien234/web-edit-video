"""Dựng timeline thành một lệnh FFmpeg, rồi kiểm tra file xuất ra.

Hỗ trợ: track video đầu tiên (cắt, khoảng trống thành khung đen, màu phim cho từng clip),
âm thanh của clip video, mọi track âm thanh (nhạc, giọng, tiếng động) trộn vào, và các
track chữ được in lên hình làm phụ đề. Chưa hỗ trợ chồng hai lớp video (xem PROJECT.md).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from .. import media
from . import srt
from .looks import filter_for
from .models import Media, Project


class RenderError(media.MediaError):
    pass


_SCRIPT_OPT: str | None = None


def filter_script_option(version_line: str) -> str:
    """Tên tùy chọn đọc filter từ file: FFmpeg 7 trở lên dùng "-/filter_complex",
    bản cũ hơn dùng "-filter_complex_script" (bản 8 đã bỏ hẳn tùy chọn này)."""
    import re as _re
    m = _re.search(r"ffmpeg version (?:n)?(\d+)\.", version_line)
    if m and int(m.group(1)) < 7:
        return "-filter_complex_script"
    return "-/filter_complex"     # bản mới, hoặc bản build từ git không ghi số phiên bản


def _script_opt() -> str:
    global _SCRIPT_OPT
    if _SCRIPT_OPT is None:
        media.require_ffmpeg()
        out = subprocess.run([media.FFMPEG, "-hide_banner", "-version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        _SCRIPT_OPT = filter_script_option(out.splitlines()[0] if out else "")
    return _SCRIPT_OPT


def _f(x: float) -> str:
    return f"{x:.3f}"


# Mẫu kiểu chữ phụ đề. "border" theo chuẩn ASS: 1 = chữ có viền, 3 = chữ nằm trên nền đặc.
# Màu chữ (PrimaryColour) lấy từ project.caption_color để người dùng còn đổi được;
# mỗi mẫu quyết định màu viền/nền, độ dày viền, bóng đổ và chữ nghiêng.
CAPTION_PRESETS: dict[str, dict] = {
    "classic":     {"ten": "Cổ điển",       "vien": "#000000", "border": 1, "day": 2, "bong": 0, "nghieng": 0},
    "box":         {"ten": "Nền đen",       "vien": "#000000", "border": 3, "day": 4, "bong": 0, "nghieng": 0},
    "popart":      {"ten": "Pop Art",       "vien": "#22C55E", "border": 1, "day": 3, "bong": 0, "nghieng": 0},
    "bubblegum":   {"ten": "Kẹo ngọt",      "vien": "#FFFFFF", "border": 1, "day": 3, "bong": 0, "nghieng": 1},
    "highlighter": {"ten": "Bút dạ quang",  "vien": "#8B6DFF", "border": 3, "day": 4, "bong": 0, "nghieng": 0},
    "sunset":      {"ten": "Cam tươi",      "vien": "#3A1200", "border": 1, "day": 3, "bong": 1, "nghieng": 0},
}


def _mask_box(project, W: int, H: int) -> tuple[int, int, int, int]:
    """Đổi vùng che từ % sang pixel. yuv420p đòi toạ độ và kích thước chẵn,
    nên làm tròn xuống số chẵn và kẹp lại cho khỏi tràn ra ngoài khung hình."""
    def chan(v: float) -> int:
        return max(2, int(v) // 2 * 2)

    x = chan(max(0.0, min(98.0, project.mask_x)) / 100 * W)
    y = chan(max(0.0, min(98.0, project.mask_y)) / 100 * H)
    w = chan(max(1.0, min(100.0, project.mask_w)) / 100 * W)
    h = chan(max(1.0, min(100.0, project.mask_h)) / 100 * H)
    w = chan(min(w, W - x))     # không cho vùng che chạy quá mép phải
    h = chan(min(h, H - y))     # và quá mép dưới
    return x, y, w, h


def _ass_color(hex_color: str) -> str:
    """Đổi "#RRGGBB" sang màu kiểu ASS/libass: "&H00BBGGRR" (thứ tự BGR, không trong suốt)."""
    h = (hex_color or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def build_args(project: Project, get_media: Callable[[str], Media], out_name: str, work_dir: Path) -> list[str]:
    """Trả về tham số FFmpeg (không gồm tên chương trình). Chạy với cwd=work_dir."""
    W, H, fps = project.width, project.height, project.fps
    total = project.duration
    if total <= 0:
        raise RenderError("Timeline đang trống, chưa có gì để xuất.")

    inputs: list[str] = []
    index: dict[str, int] = {}

    def media_input(mid: str) -> tuple[int, Media]:
        m = get_media(mid)
        if mid not in index:
            index[mid] = len(index)
            inputs.extend(["-i", m.path])
        return index[mid], m

    filters, vlabels, alabels = [], [], []
    norm = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p")
    silence = "anullsrc=channel_layout=stereo:sample_rate=48000"
    main = next((t for t in project.tracks if t.kind == "video"), None)
    segs, cursor = 0, 0.0

    def gap(length: float) -> None:
        nonlocal segs
        filters.append(f"color=c=black:s={W}x{H}:r={fps}:d={_f(length)},format=yuv420p,setsar=1[v{segs}]")
        filters.append(f"{silence},atrim=0:{_f(length)}[a{segs}]")
        vlabels.append(f"[v{segs}]")
        alabels.append(f"[a{segs}]")
        segs += 1

    if main:
        for clip in sorted(main.clips, key=lambda c: c.start):
            if clip.start > cursor + 0.001:
                gap(clip.start - cursor)
            elif clip.start < cursor - 0.001:
                raise RenderError(f"Hai clip chồng lên nhau trên track video chính ở giây {clip.start:.2f}.")
            idx, m = media_input(clip.media_id)
            look = filter_for(clip.look)
            chain = f"[{idx}:v]trim={_f(clip.in_point)}:{_f(clip.out_point)},setpts=PTS-STARTPTS,{norm}"
            filters.append(chain + (f",{look}" if look else "") + f"[v{segs}]")
            if m.has_audio and not main.muted:
                filters.append(f"[{idx}:a]atrim={_f(clip.in_point)}:{_f(clip.out_point)},asetpts=PTS-STARTPTS,"
                               f"aformat=sample_rates=48000:channel_layouts=stereo,volume={clip.volume}[a{segs}]")
            else:
                filters.append(f"{silence},atrim=0:{_f(clip.duration)}[a{segs}]")
            vlabels.append(f"[v{segs}]")
            alabels.append(f"[a{segs}]")
            segs += 1
            cursor = clip.end
    if cursor < total - 0.001:
        gap(total - cursor)
    filters.append("".join(v + a for v, a in zip(vlabels, alabels)) + f"concat=n={segs}:v=1:a=1[vcat][acat]")

    mix = ["[acat]"]
    for t in project.tracks:
        if t.kind != "audio" or t.muted:
            continue
        for clip in t.clips:
            idx, _ = media_input(clip.media_id)
            delay = int(round(clip.start * 1000))
            lab = f"x{len(mix)}"
            filters.append(f"[{idx}:a]atrim={_f(clip.in_point)}:{_f(clip.out_point)},asetpts=PTS-STARTPTS,"
                           f"aformat=sample_rates=48000:channel_layouts=stereo,volume={clip.volume},"
                           f"adelay={delay}|{delay}[{lab}]")
            mix.append(f"[{lab}]")
    if len(mix) > 1:
        filters.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:normalize=0,alimiter=limit=0.95[aout]")
    else:
        filters.append("[acat]anull[aout]")

    # Che phụ đề cháy sẵn trong video gốc. Làm TRƯỚC khi in phụ đề mới, để phụ đề
    # của mình nằm đè lên vùng vừa che chứ không bị che mất.
    vsrc = "[vcat]"
    if project.mask_on:
        mx, my, mw, mh = _mask_box(project, W, H)
        if project.mask_mode == "box":
            filters.append(f"[vcat]drawbox=x={mx}:y={my}:w={mw}:h={mh}:color=black@1:t=fill[vmask]")
        else:
            # Cắt riêng ô cần che -> làm mờ thật mạnh -> dán đè lại đúng chỗ cũ.
            filters.append(f"[vcat]split=2[mbase][mcrop]")
            filters.append(f"[mcrop]crop={mw}:{mh}:{mx}:{my},boxblur=20:2[mblur]")
            filters.append(f"[mbase][mblur]overlay={mx}:{my}[vmask]")
        vsrc = "[vmask]"

    # Track chữ -> file phụ đề nằm ngay trong work_dir, gọi bằng tên tương đối
    # để tránh lỗi đường dẫn "D:\" trong bộ lọc subtitles trên Windows.
    cues = [srt.Cue(c.start, c.end, c.text or "") for t in project.tracks if t.kind == "text" and not t.muted
            for c in t.clips if c.text]
    if cues:
        (work_dir / "captions.srt").write_text(srt.dump(sorted(cues, key=lambda c: c.start)), encoding="utf-8")
        # Cỡ chữ tính theo khung 288 dòng của libass, nên tự co giãn theo độ phân giải.
        size = project.caption_size or (18 if W > H else 14)
        mau = CAPTION_PRESETS.get(project.caption_preset) or CAPTION_PRESETS["classic"]
        # Với BorderStyle=3, tùy bản libass mà nền được tô bằng OutlineColour hoặc BackColour,
        # nên đặt cả hai cùng một màu để mẫu "nền đặc" hiện đúng ở mọi bản FFmpeg.
        vien = _ass_color(mau["vien"])
        style = (f"FontName={project.caption_font or 'Arial'},FontSize={size},"
                 f"PrimaryColour={_ass_color(project.caption_color)},"
                 f"OutlineColour={vien},BackColour={vien},"
                 f"BorderStyle={mau['border']},Outline={mau['day']},Shadow={mau['bong']},"
                 f"Bold={1 if project.caption_bold else 0},Italic={mau['nghieng']},MarginV=40")
        filters.append(f"{vsrc}subtitles=captions.srt:force_style='{style}'[vout]")
    else:
        filters.append(f"{vsrc}null[vout]")

    script = work_dir / "filter.txt"   # lệnh dài trên Windows có giới hạn 32k ký tự
    script.write_text(";\n".join(filters), encoding="utf-8")
    return [*inputs, _script_opt(), script.name, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", _f(total), out_name]


def export(project: Project, get_media: Callable[[str], Media], work_dir: Path,
           report: Callable[[float, str], None] | None = None) -> tuple[Path, dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "video.mp4"
    args = build_args(project, get_media, out.name, work_dir)
    media.run_ffmpeg(args, duration=project.duration, cwd=work_dir,
                     on_progress=(lambda f: report(0.02 + 0.9 * f, "Đang xuất video")) if report else None)
    if report:
        report(0.95, "Đang kiểm tra file xuất ra")
    return out, verify(out, project.duration)


def verify(path: Path, expected: float) -> dict:
    """Kiểm tra file thật sự dùng được, không chỉ tin vào việc FFmpeg 'chạy xong'."""
    info = media.probe(path)
    streams = {s.get("codec_type") for s in info.get("streams", [])}
    duration = float(info.get("format", {}).get("duration") or 0)
    problems = []
    if "video" not in streams:
        problems.append("không có hình")
    if "audio" not in streams:
        problems.append("không có tiếng")
    if abs(duration - expected) > 0.25:
        problems.append(f"dài {duration:.2f}s, mong đợi {expected:.2f}s")
    log = subprocess.run([media.FFMPEG, "-hide_banner", "-nostdin", "-i", str(path),
                          "-vf", "blackdetect=d=1:pix_th=0.05", "-af", "silencedetect=n=-50dB:d=2",
                          "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stderr
    black = sum(float(x) for x in re.findall(r"black_duration:([\d.]+)", log))
    silent = sum(float(x) for x in re.findall(r"silence_duration: ([\d.]+)", log))
    return {"ok": not problems, "problems": problems, "duration": round(duration, 2),
            "black_seconds": round(black, 2), "silent_seconds": round(silent, 2)}
