"""Cổng MCP: cho Claude Code, Codex, Cursor... dựng video trên Creative Motion Lab như một người dùng.

Không cần cài thêm thư viện: nói chuyện MCP qua stdin/stdout (JSON-RPC, mỗi dòng một thông điệp),
còn mọi thao tác thì gọi API của máy chủ đang chạy. Nhờ vậy agent đi chung "một cửa" EditorCore
với người dùng: chung project, chung lịch sử, hoàn tác được cho cả hai.

Chạy (máy chủ start.bat phải đang chạy):   .venv\\Scripts\\python -m backend.editor.mcp_server
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

BASE = os.environ.get("CML_URL", "http://127.0.0.1:8000")
AGENT = os.environ.get("CML_AGENT_NAME", f"agent-{uuid.uuid4().hex[:6]}")
PROTOCOL = "2025-06-18"


def http(method: str, path: str, body: dict | None = None, timeout: float = 120) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-CML": "1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}").get("detail")
        except ValueError:
            detail = None
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Không kết nối được máy chủ {BASE}. Hãy chạy start.bat trước. ({e.reason})") from None


def wait_job(job_id: str, timeout: float = 1800) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = http("GET", f"/api/jobs/{job_id}")
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(1.5)
    return {"id": job_id, "status": "timeout"}


def heartbeat(pid: str) -> None:
    http("POST", f"/api/editor/projects/{pid}/lock", {"owner": AGENT})


# ------------------------------------------------------------------ công cụ
TOOLS: dict[str, tuple[str, dict, Callable[..., Any]]] = {}


def tool(name: str, description: str, props: dict, required: list[str] | None = None):
    def deco(fn):
        TOOLS[name] = (description, {"type": "object", "properties": props, "required": required or []}, fn)
        return fn
    return deco


S, N, I, B = {"type": "string"}, {"type": "number"}, {"type": "integer"}, {"type": "boolean"}
LIST = {"type": "array", "items": {"type": "string"}}


@tool("list_media", "Liệt kê file đã nhập (id, tên, thời lượng, có hình/tiếng).", {})
def list_media():
    return http("GET", "/api/editor/media")


@tool("import_media", "Nhập một file video/nhạc trên máy này bằng đường dẫn đầy đủ.", {"path": S}, ["path"])
def import_media(path):
    return http("POST", "/api/editor/media/import-path", {"path": path})


@tool("list_projects", "Liệt kê các project.", {})
def list_projects():
    return http("GET", "/api/editor/projects")


@tool("create_project", "Tạo project trống. 1080x1920 là dọc (TikTok), 1920x1080 là ngang (YouTube).",
      {"name": S, "width": I, "height": I, "fps": I}, ["name"])
def create_project(name, width=1080, height=1920, fps=30):
    return http("POST", "/api/editor/projects", {"name": name, "width": width, "height": height, "fps": fps})


@tool("open_project", "Bắt đầu sửa một project: giữ khóa sửa (tự hết hạn nếu agent dừng) và trả về timeline.",
      {"project_id": S}, ["project_id"])
def open_project(project_id):
    heartbeat(project_id)
    return http("GET", f"/api/editor/projects/{project_id}")


@tool("get_project", "Đọc timeline hiện tại, gồm cả chỗ người dùng vừa sửa tay. Dùng 'version' làm expected_version.",
      {"project_id": S}, ["project_id"])
def get_project(project_id):
    heartbeat(project_id)
    return http("GET", f"/api/editor/projects/{project_id}")


@tool("edit", "Áp một lệnh sửa timeline. Loại lệnh: add_track{kind:video|audio|text,name}, remove_track{track_id}, update_track{track_id,fields:{name,muted}}, "
      "add_clip{track_id,clip:{media_id,start,in_point,out_point,text,volume,look}}, update_clip{clip_id,fields}, "
      "remove_clip{clip_id}, move_clip{clip_id,track_id,start}, split_clip{clip_id,at}, "
      "set_project{name,width,height,fps}, replace_clips{track_id,clips}, batch{commands:[...]} (một bước hoàn tác). "
      "Luôn gửi expected_version từ lần đọc gần nhất để không ghi đè chỗ người dùng vừa sửa.",
      {"project_id": S, "command": {"type": "object"}, "expected_version": I}, ["project_id", "command"])
def edit(project_id, command, expected_version=None):
    heartbeat(project_id)
    return http("POST", f"/api/editor/projects/{project_id}/commands",
                {"command": command, "actor": AGENT, "expected_version": expected_version})


@tool("undo", "Hoàn tác thay đổi gần nhất (của ai cũng được).", {"project_id": S}, ["project_id"])
def undo(project_id):
    return http("POST", f"/api/editor/projects/{project_id}/undo")


@tool("templates_and_looks", "Các mẫu edit (nhạc, kể chuyện, game) và màu phim dùng được.", {})
def templates_and_looks():
    return {"templates": http("GET", "/api/editor/templates"), "looks": http("GET", "/api/editor/looks")}


@tool("apply_template", "Dựng cả timeline theo một mẫu. Mẫu theo nhạc cần music_id.",
      {"project_id": S, "template": S, "source_ids": LIST, "music_id": S}, ["project_id", "template", "source_ids"])
def apply_template(project_id, template, source_ids, music_id=None):
    heartbeat(project_id)
    return http("POST", f"/api/editor/projects/{project_id}/template",
                {"template": template, "source_ids": source_ids, "music_id": music_id, "actor": AGENT})


@tool("beat_cut", "Cắt các video nguồn theo beat của bài nhạc thành timeline.",
      {"project_id": S, "music_id": S, "source_ids": LIST, "beats_per_cut": I, "look": S, "max_seconds": N},
      ["project_id", "music_id", "source_ids"])
def beat_cut(project_id, music_id, source_ids, beats_per_cut=2, look=None, max_seconds=60):
    heartbeat(project_id)
    return http("POST", f"/api/editor/projects/{project_id}/beat-cut",
                {"music_id": music_id, "source_ids": source_ids, "beats_per_cut": beats_per_cut, "look": look,
                 "max_seconds": max_seconds, "actor": AGENT})


@tool("auto_captions", "Nghe lời nói trong một file rồi đặt phụ đề lên timeline, bắt đầu ở giây 'offset'. "
      "'targets' là các ngôn ngữ cần dịch thêm (vd ['vi','en']); 'lang' chọn bản đặt lên timeline.",
      {"project_id": S, "media_id": S, "offset": N, "targets": LIST, "lang": S}, ["project_id", "media_id"])
def auto_captions(project_id, media_id, offset=0.0, targets=None, lang=None):
    job = http("POST", f"/api/editor/media/{media_id}/tool",
               {"kind": "subtitles", "options": {"mode": "transcribe", "targets": targets or []}})
    done = wait_job(job["job_id"])
    if done["status"] != "done":
        return done
    heartbeat(project_id)
    return http("POST", f"/api/editor/projects/{project_id}/captions-from-job",
                {"job_id": job["job_id"], "lang": lang, "offset": offset, "actor": AGENT})


@tool("clean_voice", "Khử tiếng ồn cho một file (level 0 nhẹ, 1 tiêu chuẩn, 2 mạnh). Trả về media mới.",
      {"media_id": S, "level": I}, ["media_id"])
def clean_voice(media_id, level=1):
    job = http("POST", f"/api/editor/media/{media_id}/tool", {"kind": "denoise", "options": {"level": level}})
    done = wait_job(job["job_id"])
    if done["status"] != "done":
        return done
    return {"job": done.get("extra"), "media": http("POST", f"/api/editor/media/from-job/{job['job_id']}")}


@tool("export_video", "Xuất project ra MP4 và kiểm tra file (độ dài, khung đen, đoạn lặng).",
      {"project_id": S}, ["project_id"])
def export_video(project_id):
    job = http("POST", f"/api/editor/projects/{project_id}/export")
    done = wait_job(job["job_id"])
    if done.get("result_url"):
        done["download"] = BASE + done["result_url"]
    return done


@tool("close_project", "Sửa xong: nhả khóa để agent khác sửa được.", {"project_id": S}, ["project_id"])
def close_project(project_id):
    return http("POST", f"/api/editor/projects/{project_id}/unlock", {"owner": AGENT})


# ------------------------------------------------------------------ giao thức MCP (stdio)
def handle(msg: dict) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:          # thông báo (notification): không trả lời
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "creative-motion-lab", "version": "0.1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (d, s, _) in TOOLS.items()]}
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Không có công cụ {name}"}}
        try:
            out = TOOLS[name][2](**args)
            result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=1)}]}
        except Exception as e:  # noqa: BLE001 - báo lỗi cho agent đọc, không làm sập cổng
            result = {"content": [{"type": "text", "text": f"Lỗi: {e}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Không hỗ trợ {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    # Windows: mặc định không phải UTF-8 và đổi "\n" thành "\r\n"; ép về đúng chuẩn MCP.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    stdin = open(sys.stdin.fileno(), encoding="utf-8", closefd=False)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except Exception as e:  # noqa: BLE001
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
