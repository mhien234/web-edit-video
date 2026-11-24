"""SQLite storage: media library, projects, and a version per change.

Every change is a new version row, so undo/redo is moving a pointer
(`head`) and nothing is ever overwritten in place.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .models import Media, Project

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, head INTEGER NOT NULL, latest INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS versions (
    project_id TEXT NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL,
    actor TEXT, command TEXT, created_at REAL NOT NULL,
    PRIMARY KEY (project_id, version));
CREATE TABLE IF NOT EXISTS locks (
    project_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS media_uid (
    media_id TEXT PRIMARY KEY, uid TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS project_uid (
    project_id TEXT PRIMARY KEY, uid TEXT NOT NULL);
"""


class Store:
    def __init__(self, db_path: Path):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()
        # File media nằm cạnh chính file CSDL này (<data>/editor/media/), nên luôn
        # dựng lại được đường dẫn kể cả khi thư mục dự án đã bị đổi tên hay chuyển chỗ.
        self._media_dir = Path(db_path).resolve().parent / "media"

    # ---- media -------------------------------------------------------
    def add_media(self, media: Media) -> None:
        with self._lock:
            self._db.execute("INSERT INTO media VALUES (?,?,?)", (media.id, json.dumps(asdict(media)), time.time()))
            self._db.commit()

    def _heal_path(self, media: Media) -> Media:
        """Sửa đường dẫn tuyệt đối đã cũ.

        CSDL lưu đường dẫn tuyệt đối lúc nhập file. Đổi tên thư mục dự án (ví dụ bỏ dấu
        tiếng Việt) hay chuyển sang ổ khác là mọi đường dẫn đó chết, kéo theo khử ồn,
        phụ đề, xóa nền và xuất video đều hỏng với lỗi khó hiểu "FileNotFoundError".
        File thật vẫn nằm trong <thư mục của editor.db>/media/ theo đúng mã media,
        nên dò lại ở đó rồi ghi đè đường dẫn mới vào CSDL cho lần sau.
        """
        if Path(media.path).is_file():
            return media
        found = self._media_dir / Path(media.path.replace("\\", "/")).name
        if not found.is_file():
            return media          # thật sự mất file: để lớp trên báo lỗi như cũ
        media.path = str(found)
        try:
            with self._lock:
                self._db.execute("UPDATE media SET data=? WHERE id=?", (json.dumps(asdict(media)), media.id))
                self._db.commit()
        except sqlite3.Error:
            pass                  # không ghi được thì vẫn trả về đường dẫn đúng cho lần chạy này
        return media

    def get_media(self, media_id: str) -> Media:
        row = self._db.execute("SELECT data FROM media WHERE id=?", (media_id,)).fetchone()
        if not row:
            raise KeyError(f"media not found: {media_id}")
        return self._heal_path(Media(**json.loads(row[0])))

    def list_media(self) -> list[Media]:
        rows = self._db.execute("SELECT data FROM media ORDER BY created_at").fetchall()
        return [self._heal_path(Media(**json.loads(r[0]))) for r in rows]

    # ---- projects and versions ---------------------------------------
    def create_project(self, project: Project, actor: str) -> int:
        with self._lock:
            self._db.execute("INSERT INTO projects VALUES (?,?,?,?)", (project.id, project.name, 1, 1))
            self._insert_version(project, 1, actor, {"type": "create_project"})
            self._db.commit()
        return 1

    def _insert_version(self, project: Project, version: int, actor: str, command: dict) -> None:
        self._db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)",
                         (project.id, version, json.dumps(project.to_dict()), actor, json.dumps(command), time.time()))

    def head(self, project_id: str) -> tuple[int, int]:
        row = self._db.execute("SELECT head, latest FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError(f"project not found: {project_id}")
        return row[0], row[1]

    def load(self, project_id: str, version: int | None = None) -> tuple[Project, int]:
        head, _ = self.head(project_id)
        v = version or head
        row = self._db.execute("SELECT data FROM versions WHERE project_id=? AND version=?", (project_id, v)).fetchone()
        return Project.from_dict(json.loads(row[0])), v

    def commit(self, project: Project, actor: str, command: dict) -> int:
        """Save a new version after head; anything that was 'redo-able' is dropped."""
        with self._lock:
            head, _ = self.head(project.id)
            self._db.execute("DELETE FROM versions WHERE project_id=? AND version>?", (project.id, head))
            new_v = head + 1
            self._insert_version(project, new_v, actor, command)
            self._db.execute("UPDATE projects SET head=?, latest=?, name=? WHERE id=?",
                             (new_v, new_v, project.name, project.id))
            self._db.commit()
            return new_v

    def move_head(self, project_id: str, version: int) -> None:
        with self._lock:
            self._db.execute("UPDATE projects SET head=? WHERE id=?", (version, project_id))
            self._db.commit()

    def history(self, project_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT version, actor, command, created_at FROM versions WHERE project_id=? ORDER BY version",
            (project_id,)).fetchall()
        return [{"version": r[0], "actor": r[1], "command": json.loads(r[2]), "at": r[3]} for r in rows]

    def list_projects(self) -> list[dict]:
        """Danh sách dự án kèm thông tin đủ để vẽ thẻ trên trang chủ.

        Lấy luôn bản ghi phiên bản đang ở head để biết lần sửa cuối, thời lượng, số clip
        và clip video đầu tiên (dùng ảnh của nó làm ảnh đại diện). Làm một câu truy vấn
        gộp thay vì hỏi từng dự án: trang chủ chỉ gọi một lần.
        """
        rows = self._db.execute(
            "SELECT p.id, p.name, p.head, v.created_at, v.data "
            "FROM projects p LEFT JOIN versions v ON v.project_id = p.id AND v.version = p.head"
        ).fetchall()
        out: list[dict] = []
        for pid, name, head, updated_at, data in rows:
            info = {"id": pid, "name": name, "version": head, "updated_at": updated_at or 0.0,
                    "duration": 0.0, "clips": 0, "texts": 0, "thumb_media_id": None}
            if data:
                try:
                    project = Project.from_dict(json.loads(data))
                    info["duration"] = round(project.duration, 2)
                    # Tách riêng: một dự án có phụ đề tự động dễ có hàng chục khối chữ,
                    # gộp chung vào "clip" thì con số trên thẻ gây hiểu nhầm.
                    info["clips"] = sum(len(t.clips) for t in project.tracks if t.kind != "text")
                    info["texts"] = sum(len(t.clips) for t in project.tracks if t.kind == "text")
                    for track in project.tracks:
                        if track.kind != "video":
                            continue
                        for clip in track.clips:
                            if clip.media_id:
                                info["thumb_media_id"] = clip.media_id
                                break
                        if info["thumb_media_id"]:
                            break
                except (ValueError, KeyError, TypeError):
                    pass          # phiên bản hỏng thì vẫn hiện tên dự án, không làm sập trang
            out.append(info)
        out.sort(key=lambda d: d["updated_at"], reverse=True)
        return out

    def delete_media(self, media_id: str) -> None:
        """Bỏ một media khỏi thư viện. Xóa file trên đĩa là việc của EditorService."""
        with self._lock:
            self._db.execute("DELETE FROM media WHERE id=?", (media_id,))
            self._db.execute("DELETE FROM media_uid WHERE media_id=?", (media_id,))
            self._db.commit()

    def media_in_use(self, media_id: str) -> list[str]:
        """Tên các dự án đang dùng media này (chỉ xét phiên bản đang mở của mỗi dự án)."""
        rows = self._db.execute(
            "SELECT p.name, v.data FROM projects p "
            "JOIN versions v ON v.project_id = p.id AND v.version = p.head").fetchall()
        dung = []
        for name, data in rows:
            try:
                project = Project.from_dict(json.loads(data))
            except (ValueError, KeyError, TypeError):
                continue
            if any(c.media_id == media_id for t in project.tracks for c in t.clips):
                dung.append(name)
        return dung

    def delete_project(self, project_id: str) -> None:
        """Xóa hẳn một dự án cùng mọi phiên bản của nó. File media vẫn giữ nguyên."""
        with self._lock:
            self._db.execute("DELETE FROM versions WHERE project_id=?", (project_id,))
            self._db.execute("DELETE FROM locks WHERE project_id=?", (project_id,))
            self._db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            self._db.execute("DELETE FROM project_uid WHERE project_id=?", (project_id,))
            self._db.commit()

    # ---- quyền riêng tư: mỗi trình duyệt (cookie) chỉ thấy media/dự án của mình -----
    # Khi deploy công khai (Render...), mỗi khách có một "uid" ngẫu nhiên lưu trong cookie
    # (xem get_uid trong api.py). Media/dự án tạo ra sau khi có tính năng này đều được gắn
    # uid của người tạo. Dòng CHƯA có trong 2 bảng dưới (tạo từ trước, hoặc máy cá nhân
    # chạy một mình ở localhost) coi như "không ai giữ riêng" -> vẫn hiện bình thường,
    # để không phá luồng dùng một mình trên máy của chính chủ.
    def set_media_uid(self, media_id: str, uid: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO media_uid VALUES (?,?)", (media_id, uid))
            self._db.commit()

    def media_uid(self, media_id: str) -> str | None:
        row = self._db.execute("SELECT uid FROM media_uid WHERE media_id=?", (media_id,)).fetchone()
        return row[0] if row else None

    def set_project_uid(self, project_id: str, uid: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO project_uid VALUES (?,?)", (project_id, uid))
            self._db.commit()

    def project_uid(self, project_id: str) -> str | None:
        row = self._db.execute("SELECT uid FROM project_uid WHERE project_id=?", (project_id,)).fetchone()
        return row[0] if row else None

    # ---- edit locks (one agent writes at a time) ----------------------
    def acquire_lock(self, project_id: str, owner: str, ttl: float) -> bool:
        with self._lock:
            now = time.time()
            row = self._db.execute("SELECT owner, expires_at FROM locks WHERE project_id=?", (project_id,)).fetchone()
            if row and row[0] != owner and row[1] > now:
                return False
            self._db.execute("INSERT OR REPLACE INTO locks VALUES (?,?,?)", (project_id, owner, now + ttl))
            self._db.commit()
            return True

    def lock_owner(self, project_id: str) -> str | None:
        row = self._db.execute("SELECT owner, expires_at FROM locks WHERE project_id=?", (project_id,)).fetchone()
        if row and row[1] > time.time():
            return row[0]
        return None

    def release_lock(self, project_id: str, owner: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM locks WHERE project_id=? AND owner=?", (project_id, owner))
            self._db.commit()
