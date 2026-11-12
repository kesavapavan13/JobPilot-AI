"""MySQL tracker for JobPilot AI.

The schema is created (and migrated from older versions) once per process.
"""


import hashlib
import json
import os
import threading

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("MYSQL_DATABASE", "jobpilot")
STATUSES = ["Draft", "Applied", "Interview", "Offer", "Rejected"]

# Columns added after the first version of the table; added automatically if missing.
_MIGRATION_COLUMNS = {
    "bullets": "TEXT",
    "cover_letter": "TEXT",
    "missing_skills": "TEXT",
    "status": "VARCHAR(32) NOT NULL DEFAULT 'Draft'",
    "content_hash": "CHAR(64)",
}

_init_lock = threading.Lock()
_initialized = False


def _config(with_db=True):
    cfg = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
    }
    if with_db:
        cfg["database"] = DB_NAME
    return cfg


def content_hash(jd: str, resume: str) -> str:
    """Stable fingerprint of a (job description, resume) pair, used to flag duplicates."""
    raw = " ".join(jd.split()) + "||" + " ".join(resume.split())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def init_db(force=False):
    """Create the database/table if needed and add any missing columns. Runs once per process."""
    global _initialized
    with _init_lock:
        if _initialized and not force:
            return
        conn = mysql.connector.connect(**_config(with_db=False))
        try:
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
            cur.execute(f"USE `{DB_NAME}`")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS applications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company VARCHAR(255),
                    role VARCHAR(255),
                    match_score INT,
                    attempts INT,
                    approved BOOLEAN,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            cur.execute("SHOW COLUMNS FROM applications")
            existing = {row[0] for row in cur.fetchall()}
            for col, ddl in _MIGRATION_COLUMNS.items():
                if col not in existing:
                    cur.execute(f"ALTER TABLE applications ADD COLUMN `{col}` {ddl}")
            # Index for duplicate lookups; ignore if it already exists.
            try:
                cur.execute("CREATE INDEX idx_applications_hash ON applications (content_hash)")
            except mysql.connector.Error:
                pass
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def _conn():
    init_db()
    return mysql.connector.connect(**_config())


def save_application(company, role, score, attempts, approved,
                     bullets=None, cover_letter="", missing_skills=None,
                     content_hash_value=None, status="Draft") -> int:
    """Tracker tool: store one application row and return its id."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO applications (company, role, match_score, attempts, approved, "
            "bullets, cover_letter, missing_skills, status, content_hash) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                company, role, score, attempts, bool(approved),
                json.dumps(bullets or []), cover_letter,
                json.dumps(missing_skills or []), status, content_hash_value,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _row_to_dict(row):
    for key in ("bullets", "missing_skills"):
        try:
            row[key] = json.loads(row.get(key) or "[]")
        except (TypeError, ValueError):
            row[key] = []
    row["approved"] = bool(row.get("approved"))
    return row


def list_applications():
    """All applications, newest first, as dicts (bullets/missing_skills decoded to lists)."""
    conn = _conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, company, role, match_score, attempts, approved, status, "
            "bullets, cover_letter, missing_skills, created_at "
            "FROM applications ORDER BY id DESC"
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def find_duplicate(content_hash_value):
    """Return the earlier application with the same JD + resume, or None."""
    conn = _conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, company, role, created_at FROM applications "
            "WHERE content_hash = %s ORDER BY id DESC LIMIT 1",
            (content_hash_value,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def update_status(app_id, status):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE applications SET status = %s WHERE id = %s", (status, app_id))
        conn.commit()
    finally:
        conn.close()


def delete_application(app_id):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM applications WHERE id = %s", (app_id,))
        conn.commit()
    finally:
        conn.close()
