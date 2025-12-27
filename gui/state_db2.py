import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

KYIV_TZ = ZoneInfo("Europe/Kyiv")

def init_state_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        box_blob TEXT PRIMARY KEY,
        cam TEXT,
        label TEXT,
        labeled_at TEXT,
        created_ts REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exported_frames (
        frame_blob TEXT PRIMARY KEY,
        cam TEXT,
        split TEXT,
        exported_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exported_boxes (
        box_blob TEXT PRIMARY KEY,
        cam TEXT,
        split TEXT,
        cls TEXT,
        exported_at TEXT
    )""")
    con.commit()
    return con

def get_labels_for_cam(con: sqlite3.Connection, cam: str) -> dict[str, str]:
    """
    Возвращает словарь box_blob -> label для данной камеры.
    Это быстрее, чем искать по одному боксу при листании.
    """
    cur = con.cursor()
    cur.execute("SELECT box_blob, label FROM labels WHERE cam=?", (cam,))
    return {r[0]: r[1] for r in cur.fetchall()}

def upsert_label(con: sqlite3.Connection, cam: str, box_blob: str, label: str, created_ts: float):
    cur = con.cursor()
    cur.execute("""
    INSERT INTO labels(box_blob, cam, label, labeled_at, created_ts)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(box_blob) DO UPDATE SET
      cam=excluded.cam,
      label=excluded.label,
      labeled_at=excluded.labeled_at,
      created_ts=excluded.created_ts
    """, (box_blob, cam, label, datetime.now(tz=KYIV_TZ).isoformat(), float(created_ts)))
    con.commit()

def is_frame_exported(con: sqlite3.Connection, frame_blob: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM exported_frames WHERE frame_blob=? LIMIT 1", (frame_blob,))
    return cur.fetchone() is not None

def mark_frame_exported(con: sqlite3.Connection, cam: str, frame_blob: str, split: str):
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO exported_frames(frame_blob, cam, split, exported_at)
    VALUES (?, ?, ?, ?)
    """, (frame_blob, cam, split, datetime.now(tz=KYIV_TZ).isoformat()))
    con.commit()

def is_box_exported(con: sqlite3.Connection, box_blob: str, cls: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM exported_boxes WHERE box_blob=? AND cls=? LIMIT 1", (box_blob, cls))
    return cur.fetchone() is not None

def mark_box_exported(con: sqlite3.Connection, cam: str, box_blob: str, split: str, cls: str):
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO exported_boxes(box_blob, cam, split, cls, exported_at)
    VALUES (?, ?, ?, ?, ?)
    """, (box_blob, cam, split, cls, datetime.now(tz=KYIV_TZ).isoformat()))
    con.commit()