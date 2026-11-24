"""Hàng đợi việc chạy ngầm.

- Trạng thái mỗi việc lưu trong SQLite (data/jobs.db) nên tắt máy chủ đi bật lại vẫn còn.
- Một luồng worker lấy từng việc ra làm, LẦN LƯỢT từng việc một – hợp với máy RAM 8GB.
- Muốn mở rộng sau này: thay lớp JobQueue bằng Celery/RQ + Redis, giữ nguyên các handler.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .media import MediaError

ROOT = Path(__file__).resolve().parent.parent
# Thư mục dữ liệu tách khỏi mã nguồn được: đặt CML_DATA_DIR=D:\cml-data trước khi chạy.
DATA = Path(os.environ.get("CML_DATA_DIR") or ROOT / "data")
JOBS_DIR = DATA / "jobs"
DB_PATH = DATA / "jobs.db"


@dataclass
class JobContext:
    id: str
    dir: Path
    params: dict
    report: Callable[[float, str], None]

    def input_files(self, prefix: str) -> list[Path]:
        return sorted(p for p in self.dir.glob(f"{prefix}*") if p.is_file())


# kind -> hàm xử lý; hàm trả về {"file": Path, "extra": {...}}
Handler = Callable[[JobContext], dict]


class JobQueue:
    def __init__(self, db_path: Path = DB_PATH, jobs_dir: Path = JOBS_DIR, keep_days: int = 3):
        self.db_path = db_path
        self.jobs_dir = jobs_dir
        self.keep_days = keep_days
        self.handlers: dict[str, Handler] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ---------- SQLite ----------
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT, status TEXT, progress REAL, message TEXT,
                params TEXT, result_file TEXT, extra TEXT, error TEXT,
                created REAL, updated REAL)""")

    def _update(self, job_id: str, **fields) -> None:
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    # ---------- API công khai ----------
    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    def new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        d = self.jobs_dir / job_id
        d.mkdir(parents=True)
        return job_id, d

    def submit(self, job_id: str, kind: str, params: dict) -> None:
        if kind not in self.handlers:
            raise ValueError(f"Loại việc không hỗ trợ: {kind}")
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (job_id, kind, "queued", 0.0, "Đang chờ tới lượt", json.dumps(params),
                       None, None, None, now, now))
        self._q.put(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        d["extra"] = json.loads(d["extra"] or "null")
        d["position"] = self.position(job_id) if d["status"] == "queued" else 0
        return d

    def position(self, job_id: str) -> int:
        with self._conn() as c:
            rows = c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created").fetchall()
        ids = [r["id"] for r in rows]
        return ids.index(job_id) + 1 if job_id in ids else 0

    def pending(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    # ---------- Worker ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Việc đang chạy dở khi máy chủ tắt -> báo lỗi; việc đang chờ -> cho vào hàng lại.
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status='failed', error='Máy chủ đã tắt khi việc đang chạy. Hãy gửi lại.' "
                      "WHERE status='running'")
            waiting = [r[0] for r in c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created")]
        for jid in waiting:
            self._q.put(jid)
        self._cleanup_old()
        self._thread = threading.Thread(target=self._loop, name="cml-worker", daemon=True)
        self._thread.start()

    def run_once(self, job_id: str) -> None:
        """Chạy một việc ngay trong luồng hiện tại (dùng cho kiểm thử)."""
        self._process(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._process(job_id)
            finally:
                self._q.task_done()

    def _process(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job or job["status"] != "queued":
            return
        handler = self.handlers[job["kind"]]
        last = {"t": 0.0}

        def report(progress: float, message: str) -> None:
            now = time.time()
            if now - last["t"] < 0.4 and progress < 1:   # đỡ ghi SQLite quá dày
                return
            last["t"] = now
            self._update(job_id, progress=round(max(0.0, min(1.0, progress)), 3), message=message)

        self._update(job_id, status="running", message="Bắt đầu xử lý")
        ctx = JobContext(job_id, self.jobs_dir / job_id, job["params"], report)
        try:
            out = handler(ctx)
            self._update(job_id, status="done", progress=1.0, message="Hoàn tất",
                         result_file=str(out["file"].relative_to(ctx.dir)) if out.get("file") else None,
                         extra=json.dumps(out.get("extra")))
        except MediaError as e:
            self._update(job_id, status="failed", error=str(e), message="Lỗi")
        except Exception as e:  # lỗi lập trình: ghi chi tiết ra file để dễ sửa
            (ctx.dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            self._update(job_id, status="failed", error=f"Lỗi không mong muốn: {e}", message="Lỗi")

    def _cleanup_old(self) -> None:
        cutoff = time.time() - self.keep_days * 86400
        with self._lock, self._conn() as c:
            old = [r[0] for r in c.execute("SELECT id FROM jobs WHERE created < ?", (cutoff,))]
            c.execute("DELETE FROM jobs WHERE created < ?", (cutoff,))
        for jid in old:
            shutil.rmtree(self.jobs_dir / jid, ignore_errors=True)
