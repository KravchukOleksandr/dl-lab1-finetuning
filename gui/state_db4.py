import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

KYIV_TZ = ZoneInfo("Europe/Kyiv")


def init_state_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite DB for labeling progress and exported artifacts.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Per-box labeling state
    cur.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        box_blob TEXT PRIMARY KEY,
        cam TEXT,
        label TEXT,          -- TP / FP / SKIP
        labeled_at TEXT,
        created_ts REAL
    )""")

    # Exported frames for YOLO (1 row per frame saved to disk)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exported_frames (
        frame_blob TEXT PRIMARY KEY,
        cam TEXT,
        split TEXT,          -- train / val
        exported_at TEXT
    )""")

    # Exported crops for ConvNeXt (1 row per box per class)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exported_boxes (
        box_blob TEXT,
        cam TEXT,
        split TEXT,          -- train / val
        cls TEXT,            -- TP / FP
        exported_at TEXT,
        PRIMARY KEY (box_blob, cls)
    )""")

    # Reference tracking: which boxes caused a frame to be exported.
    # This allows safe deletion on relabel: delete frame only if no refs remain.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS frame_refs (
        frame_blob TEXT,
        box_blob TEXT,
        PRIMARY KEY (frame_blob, box_blob)
    )""")

    con.commit()
    return con


def get_labels_for_cam(con: sqlite3.Connection, cam: str) -> dict[str, str]:
    """
    Return mapping box_blob -> label for a camera.
    """
    cur = con.cursor()
    cur.execute("SELECT box_blob, label FROM labels WHERE cam=?", (cam,))
    return {r[0]: r[1] for r in cur.fetchall()}


def upsert_label(con: sqlite3.Connection, cam: str, box_blob: str, label: str, created_ts: float) -> None:
    """
    Insert or update label for a given box.
    """
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


# --------------------------
# YOLO frame export helpers
# --------------------------

def is_frame_exported(con: sqlite3.Connection, frame_blob: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM exported_frames WHERE frame_blob=? LIMIT 1", (frame_blob,))
    return cur.fetchone() is not None


def get_exported_frame_info(con: sqlite3.Connection, frame_blob: str) -> tuple[str, str] | None:
    """
    Return (cam, split) if frame is exported, otherwise None.
    """
    cur = con.cursor()
    cur.execute("SELECT cam, split FROM exported_frames WHERE frame_blob=? LIMIT 1", (frame_blob,))
    row = cur.fetchone()
    if not row:
        return None
    return row[0], row[1]


def mark_frame_exported(con: sqlite3.Connection, cam: str, frame_blob: str, split: str) -> None:
    """
    Mark that a frame file exists on disk (exported).
    """
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO exported_frames(frame_blob, cam, split, exported_at)
    VALUES (?, ?, ?, ?)
    """, (frame_blob, cam, split, datetime.now(tz=KYIV_TZ).isoformat()))
    con.commit()


def add_frame_ref(con: sqlite3.Connection, frame_blob: str, box_blob: str) -> None:
    """
    Add a reference: this box caused this frame to be exported.
    """
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO frame_refs(frame_blob, box_blob)
    VALUES (?, ?)
    """, (frame_blob, box_blob))
    con.commit()


def remove_frame_ref(con: sqlite3.Connection, frame_blob: str, box_blob: str) -> None:
    """
    Remove a reference between frame and box (used during relabel).
    """
    cur = con.cursor()
    cur.execute("DELETE FROM frame_refs WHERE frame_blob=? AND box_blob=?", (frame_blob, box_blob))
    con.commit()


def get_frame_refcount(con: sqlite3.Connection, frame_blob: str) -> int:
    """
    Return how many boxes currently reference this exported frame.
    """
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM frame_refs WHERE frame_blob=?", (frame_blob,))
    return int(cur.fetchone()[0])


def delete_exported_frame_row(con: sqlite3.Connection, frame_blob: str) -> None:
    """
    Remove exported_frames row (file should be removed by caller).
    """
    cur = con.cursor()
    cur.execute("DELETE FROM exported_frames WHERE frame_blob=?", (frame_blob,))
    con.commit()


# --------------------------
# ConvNeXt crop export helpers
# --------------------------

def is_box_exported(con: sqlite3.Connection, box_blob: str, cls: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM exported_boxes WHERE box_blob=? AND cls=? LIMIT 1", (box_blob, cls))
    return cur.fetchone() is not None


def get_exported_box_info(con: sqlite3.Connection, box_blob: str, cls: str) -> tuple[str, str, str] | None:
    """
    Return (cam, split, cls) if crop was exported, else None.
    """
    cur = con.cursor()
    cur.execute("SELECT cam, split, cls FROM exported_boxes WHERE box_blob=? AND cls=? LIMIT 1", (box_blob, cls))
    row = cur.fetchone()
    if not row:
        return None
    return row[0], row[1], row[2]


def mark_box_exported(con: sqlite3.Connection, cam: str, box_blob: str, split: str, cls: str) -> None:
    """
    Mark that a crop file exists on disk (exported).
    """
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO exported_boxes(box_blob, cam, split, cls, exported_at)
    VALUES (?, ?, ?, ?, ?)
    """, (box_blob, cam, split, cls, datetime.now(tz=KYIV_TZ).isoformat()))
    con.commit()


def delete_exported_box_row(con: sqlite3.Connection, box_blob: str, cls: str) -> None:
    """
    Remove exported_boxes row (file should be removed by caller).
    """
    cur = con.cursor()
    cur.execute("DELETE FROM exported_boxes WHERE box_blob=? AND cls=?", (box_blob, cls))
    con.commit()
