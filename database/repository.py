"""All SQL lives here (parameterised queries only) so SQLite can be swapped for PostgreSQL later."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from database.connection import connect, transaction
from database.schema import SCHEMA_SQL, SCHEMA_VERSION

COMPLETE_STATUSES = ("completed", "completed_with_warnings")
REVIEW_DECISIONS = ("reviewed", "confirmed", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Repository:
    def __init__(self, db_path: str | Path):
        self.conn: sqlite3.Connection = connect(db_path)
        self.conn.executescript(SCHEMA_SQL)
        if self.conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 0:
            self.conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Apply additive migrations while preserving existing analysis history."""
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "qwen_explanation" not in columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN qwen_explanation TEXT")
        if "qwen_status" not in columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN qwen_status TEXT")
        self.conn.execute("UPDATE schema_info SET version=?", (SCHEMA_VERSION,))

    def close(self) -> None:
        self.conn.close()

    # ---- videos ----
    def get_video_by_hash(self, sha256: str):
        return self.conn.execute("SELECT * FROM videos WHERE sha256=?", (sha256,)).fetchone()

    def get_video(self, video_id: int):
        return self.conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()

    def add_video(self, sha256: str, original_name: str, stored_path: str, meta: dict) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO videos(sha256, original_name, stored_path, file_size_bytes, duration_s, fps, width,"
                " height, frame_count, container, fourcc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (sha256, original_name, stored_path, meta["file_size_bytes"], meta["duration_s"], meta["fps"],
                 meta["width"], meta["height"], meta["frame_count"], meta.get("container"), meta.get("fourcc")))
            return cur.lastrowid

    def set_video_file(self, video_id: int, stored_path: str, available: bool) -> None:
        with transaction(self.conn):
            self.conn.execute("UPDATE videos SET stored_path=?, file_available=? WHERE id=?",
                              (stored_path, int(available), video_id))

    def refresh_file_availability(self) -> None:
        with transaction(self.conn):
            for r in self.conn.execute("SELECT id, stored_path FROM videos").fetchall():
                self.conn.execute("UPDATE videos SET file_available=? WHERE id=?",
                                  (int(Path(r["stored_path"]).is_file()), r["id"]))

    # ---- runs ----
    def create_run(self, video_id: int, model_key: str, model_info: dict, config_json: str,
                   config_hash: str, app_version: str) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO analysis_runs(video_id, status, model_key, model_info_json, config_json, config_hash,"
                " app_version, started_at) VALUES (?,?,?,?,?,?,?,?)",
                (video_id, "running", model_key, json.dumps(model_info), config_json, config_hash,
                 app_version, _now()))
            return cur.lastrowid

    def finish_run(self, run_id: int, status: str, elapsed_s: float, timing: dict, windows_total: int,
                   windows_done: int, warnings: list[str], error: str | None = None) -> None:
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE analysis_runs SET status=?, finished_at=?, elapsed_s=?, timing_json=?, windows_total=?,"
                " windows_done=?, warnings_json=?, error=? WHERE id=?",
                (status, _now(), elapsed_s, json.dumps(timing), windows_total, windows_done,
                 json.dumps(warnings), error, run_id))

    def update_run_progress(self, run_id: int, windows_total: int, windows_done: int) -> None:
        """Persist window counts before later, optional stages can fail or be interrupted."""
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE analysis_runs SET windows_total=?, windows_done=? WHERE id=? AND status='running'",
                (windows_total, windows_done, run_id))

    def append_run_warning(self, run_id: int, warning: str) -> None:
        """Persist a warning immediately so abrupt process exits do not erase diagnostics."""
        with transaction(self.conn):
            row = self.conn.execute("SELECT warnings_json FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return
            try:
                warnings = json.loads(row["warnings_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                warnings = []
            warnings.append(str(warning))
            self.conn.execute("UPDATE analysis_runs SET warnings_json=? WHERE id=?",
                              (json.dumps(warnings, ensure_ascii=False), run_id))

    def mark_stale_running(self) -> int:
        """Runs left in 'running' (browser refresh / crash) are marked interrupted, never 'completed'."""
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE analysis_runs SET status='interrupted', finished_at=?, "
                "error='Analysis was interrupted before finishing.' WHERE status='running'", (_now(),))
            return cur.rowcount

    def get_run(self, run_id: int):
        return self.conn.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()

    def update_run_model_info(self, run_id: int, model_info: dict) -> None:
        with transaction(self.conn):
            self.conn.execute("UPDATE analysis_runs SET model_info_json=? WHERE id=?",
                              (json.dumps(model_info), run_id))

    def list_runs(self, video_id: int | None = None):
        q = ("SELECT r.*, v.original_name, v.duration_s, v.file_available FROM analysis_runs r "
             "JOIN videos v ON v.id=r.video_id ")
        if video_id is None:
            return self.conn.execute(q + "ORDER BY r.id DESC").fetchall()
        return self.conn.execute(q + "WHERE r.video_id=? ORDER BY r.id DESC", (video_id,)).fetchall()

    def find_complete_run(self, video_id: int, config_hash: str):
        return self.conn.execute(
            "SELECT * FROM analysis_runs WHERE video_id=? AND config_hash=? AND status IN (?,?) "
            "ORDER BY id DESC LIMIT 1", (video_id, config_hash, *COMPLETE_STATUSES)).fetchone()

    def delete_run(self, run_id: int) -> bool:
        """Completely delete an analysis run, its predictions, events, evidence files, and reviews."""
        with transaction(self.conn):
            run = self.get_run(run_id)
            if not run:
                return False

            events = self.conn.execute("SELECT id FROM events WHERE run_id=?", (run_id,)).fetchall()
            for ev in events:
                eid = ev["id"]
                frames = self.conn.execute("SELECT path FROM evidence_frames WHERE event_id=?", (eid,)).fetchall()
                for f in frames:
                    try:
                        p = Path(f["path"])
                        if p.is_file():
                            p.unlink(missing_ok=True)
                    except OSError:
                        pass
                self.conn.execute("DELETE FROM reviews WHERE event_id=?", (eid,))
                self.conn.execute("DELETE FROM evidence_frames WHERE event_id=?", (eid,))

            self.conn.execute("DELETE FROM events WHERE run_id=?", (run_id,))
            self.conn.execute("DELETE FROM window_predictions WHERE run_id=?", (run_id,))
            self.conn.execute("DELETE FROM analysis_runs WHERE id=?", (run_id,))
            return True

    def delete_video(self, video_id: int, delete_file: bool = True) -> bool:
        """Delete a video and all associated runs, evidence, and optionally its source video file."""
        video = self.get_video(video_id)
        if not video:
            return False
        runs = self.conn.execute("SELECT id FROM analysis_runs WHERE video_id=?", (video_id,)).fetchall()
        for r in runs:
            self.delete_run(r["id"])
        if delete_file and video["stored_path"]:
            try:
                p = Path(video["stored_path"])
                if p.is_file():
                    p.unlink(missing_ok=True)
            except OSError:
                pass
        with transaction(self.conn):
            self.conn.execute("DELETE FROM videos WHERE id=?", (video_id,))
        return True

    def clear_all_history(self) -> int:
        """Clear all analysis runs and associated evidence files. Returns count of deleted runs."""
        runs = self.conn.execute("SELECT id FROM analysis_runs").fetchall()
        count = len(runs)
        for r in runs:
            self.delete_run(r["id"])
        return count

    # ---- predictions, events, evidence ----
    def add_windows(self, run_id: int, windows: list) -> None:
        with transaction(self.conn):
            self.conn.executemany(
                "INSERT INTO window_predictions(run_id, window_index, start_s, end_s, p_violent, status,"
                " frame_indices_json, frame_times_json, inference_ms, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(run_id, w.index, w.start_s, w.end_s, w.p_violent, w.status, json.dumps(w.frame_indices),
                  json.dumps(w.frame_times), w.inference_ms, w.note) for w in windows])

    def add_model_window_scores(self, run_id: int, model_key: str, rows: list[dict]) -> None:
        """Persist optional comparison outputs without changing X3D window predictions."""
        if not rows:
            return
        with transaction(self.conn):
            self.conn.executemany(
                "INSERT INTO model_window_predictions(run_id, model_key, window_index, start_s, end_s, "
                "p_violent, status, inference_ms, note) VALUES (?,?,?,?,?,?,?,?,?)",
                [(run_id, model_key, int(r["window_index"]), float(r["start_s"]), float(r["end_s"]),
                  r.get("p_violent"), r.get("status", "ok"), r.get("inference_ms"), r.get("note")) for r in rows])

    def list_model_window_scores(self, run_id: int, model_key: str | None = None):
        if model_key is None:
            return self.conn.execute(
                "SELECT * FROM model_window_predictions WHERE run_id=? ORDER BY model_key, window_index", (run_id,)
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM model_window_predictions WHERE run_id=? AND model_key=? ORDER BY window_index",
            (run_id, model_key),
        ).fetchall()

    def list_windows(self, run_id: int):
        return self.conn.execute("SELECT * FROM window_predictions WHERE run_id=? ORDER BY window_index",
                                 (run_id,)).fetchall()

    def add_event(self, run_id: int, e: dict, evidence: list[dict]) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO events(run_id, event_no, start_s, end_s, duration_s, label, peak_prob, mean_prob,"
                " n_windows, n_positive_runs, peak_time_s, window_indices_json, severity_score, severity_label,"
                " severity_method, severity_json, motion_json, explanation, description_method, qwen_explanation, qwen_status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, e["event_no"], e["start_s"], e["end_s"], e["duration_s"], e["label"], e["peak_prob"],
                 e["mean_prob"], e["n_windows"], e["n_positive_runs"], e["peak_time_s"],
                 json.dumps(e["window_indices"]), e["severity_score"], e["severity_label"], e["severity_method"],
                 json.dumps(e["severity_json"]), json.dumps(e["motion_json"]), e["explanation"],
                 e["description_method"], e.get("qwen_explanation"), e.get("qwen_status")))
            eid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO evidence_frames(event_id, path, timestamp_s, kind) VALUES (?,?,?,?)",
                [(eid, f["path"], f["timestamp_s"], f["kind"]) for f in evidence])
            return eid

    def list_events(self, run_id: int):
        return self.conn.execute("SELECT * FROM events WHERE run_id=? ORDER BY start_s, event_no",
                                 (run_id,)).fetchall()

    def update_event_qwen(self, event_id: int, explanation: str | None, status: str) -> None:
        with transaction(self.conn):
            self.conn.execute("UPDATE events SET qwen_explanation=?, qwen_status=? WHERE id=?",
                              (explanation, status, event_id))

    def list_evidence(self, event_id: int):
        return self.conn.execute("SELECT * FROM evidence_frames WHERE event_id=? ORDER BY timestamp_s",
                                 (event_id,)).fetchall()

    def add_audio_segments(self, run_id: int, segments: list[dict]) -> None:
        if not segments:
            return
        with transaction(self.conn):
            self.conn.executemany(
                "INSERT INTO audio_segments(run_id, kind, start_s, end_s, language, text, category, confidence, source, details_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(run_id, s.get("kind", "transcript"), float(s.get("start_s", 0)), float(s.get("end_s", 0)),
                  s.get("language"), s.get("text"), s.get("category"), s.get("confidence"), s.get("source"),
                  json.dumps(s.get("details", {}), ensure_ascii=False)) for s in segments])

    def list_audio_segments(self, run_id: int):
        return self.conn.execute("SELECT * FROM audio_segments WHERE run_id=? ORDER BY start_s, id",
                                 (run_id,)).fetchall()

    # ---- reviews (append-only, separate from model output) ----
    def add_review(self, event_id: int, decision: str, notes: str = "", human_severity: int | None = None) -> int:
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
        if human_severity is not None and not 0 <= human_severity <= 3:
            raise ValueError("human_severity must be 0-3")
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO reviews(event_id, decision, human_severity, notes, created_at) VALUES (?,?,?,?,?)",
                (event_id, decision, human_severity, notes.strip() or None, _now()))
            return cur.lastrowid

    def list_reviews(self, event_id: int):
        return self.conn.execute("SELECT * FROM reviews WHERE event_id=? ORDER BY id DESC", (event_id,)).fetchall()

    def latest_review(self, event_id: int):
        return self.conn.execute("SELECT * FROM reviews WHERE event_id=? ORDER BY id DESC LIMIT 1",
                                 (event_id,)).fetchone()

    def recent_events(self, limit: int = 30):
        ph = ",".join("?" * len(COMPLETE_STATUSES))
        return self.conn.execute(
            f"SELECT e.*, r.id AS run_no, v.original_name FROM events e JOIN analysis_runs r ON r.id=e.run_id "
            f"JOIN videos v ON v.id=r.video_id WHERE r.status IN ({ph}) ORDER BY r.id DESC, e.start_s LIMIT ?",
            (*COMPLETE_STATUSES, limit)).fetchall()

    # ---- dashboard ----
    def dashboard_stats(self) -> dict:
        c = self.conn
        ph = ",".join("?" * len(COMPLETE_STATUSES))
        total_videos = c.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        total_runs = c.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
        completed = c.execute(f"SELECT COUNT(*) FROM analysis_runs WHERE status IN ({ph})",
                              COMPLETE_STATUSES).fetchone()[0]
        total_events = c.execute(
            f"SELECT COUNT(*) FROM events e JOIN analysis_runs r ON r.id=e.run_id WHERE r.status IN ({ph})",
            COMPLETE_STATUSES).fetchone()[0]
        sev = c.execute(
            f"SELECT e.severity_label AS label, COUNT(*) AS n FROM events e JOIN analysis_runs r ON r.id=e.run_id "
            f"WHERE r.status IN ({ph}) GROUP BY e.severity_label", COMPLETE_STATUSES).fetchall()
        cats = c.execute(
            f"SELECT e.label AS label, COUNT(*) AS n FROM events e JOIN analysis_runs r ON r.id=e.run_id "
            f"WHERE r.status IN ({ph}) GROUP BY e.label", COMPLETE_STATUSES).fetchall()
        reviewed = c.execute(
            "SELECT COUNT(DISTINCT event_id) FROM reviews").fetchone()[0]
        return {"total_videos": total_videos, "total_runs": total_runs, "completed_runs": completed,
                "total_events": total_events, "reviewed_events": reviewed,
                "severity": {r["label"]: r["n"] for r in sev}, "categories": {r["label"]: r["n"] for r in cats}}
