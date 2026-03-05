"""
session_shell.py — PBM Step 2, Layer A: Session Lifecycle Manager
==================================================================
Orchestrates a single measurement session from start to finish.

Responsibilities:
  1. Initialise local SQLite buffer (schema per local_buffer_sync spec)
  2. Run pixel-sync flash sequence and derive t_zero
  3. Launch video playback and camera capture in parallel
  4. Write every iris frame to local SQLite as it arrives
  5. On session end: validate minimum frame count, write session record,
     hand off session_id to the biomarker layer (Layer B)
  6. Trigger background Supabase sync

This file does NOT calculate any biomarkers.
It guarantees that a valid, time-anchored iris stream exists when it exits.

Integration points:
  - Pixel-sync flash spec:    pixel_sync_flash.docx  (v1.1)
  - Local buffer sync spec:   local_buffer_sync.docx (v1.0)
  - Camera timestamp spec:    camera_timestamps.docx
  - Biomarker layer (Layer B): biomarker_calculator.py  [not yet written]

Usage (server-side Python, called after React Native hands off session data):
  from session_shell import SessionShell
  shell = SessionShell(video_id="NinjaGo1", child_id="child_uuid")
  session_id = shell.run(iris_stream, sync_metadata)
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pbm.session_shell")

# ---------------------------------------------------------------------------
# Environment — Supabase credentials read from env, never hardcoded
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ---------------------------------------------------------------------------
# Constants — all timing values from pixel_sync_flash spec v1.1
# ---------------------------------------------------------------------------

# Flash validation
FLASH_COUNT_REQUIRED       = 3          # Exactly 3 flashes required
INTER_FLASH_MIN_MS         = 400.0      # Minimum gap between flashes (ms)
INTER_FLASH_MAX_MS         = 600.0      # Maximum gap between flashes (ms)
INTER_FLASH_NOMINAL_MS     = 500.0      # Expected inter-flash interval (ms)

# Session integrity
MIN_FRAMES_FOR_VALID_SESSION = 100      # Below this: session is unusable
SESSION_DURATION_MS          = 60_000  # Target session length (ms)
CAMERA_FPS                   = 60      # Expected camera capture rate

# Local SQLite path — sits alongside session_shell.py in the backend folder
LOCAL_DB_PATH = os.path.join(os.path.dirname(__file__), "local_buffer.db")


# ---------------------------------------------------------------------------
# Data classes — typed containers for structured data passing
# ---------------------------------------------------------------------------

@dataclass
class FlashDetection:
    """
    The three timestamps returned by the React Native pixel-sync module.
    All values are wall-clock milliseconds derived from CMSampleBuffer
    hardware timestamps (per camera_timestamps spec).
    """
    t_detect: List[float]           # [t0_ms, t1_ms, t2_ms] — camera-side detection times
    inter_flash_gaps_ms: List[float] # [gap_0_to_1, gap_1_to_2]
    sync_valid: bool                 # True if pattern + interval checks passed


@dataclass
class GazeFrame:
    """
    One camera frame of iris tracking data.
    All coordinates are normalised [0.0, 1.0] viewport fractions.
    iris_offset_x/y are the iris-in-socket offset (for SGS density map).
    timestamp_ms is elapsed milliseconds since t_zero.
    """
    session_id:      str
    frame_index:     int
    timestamp_ms:    int            # Hardware timestamp, ms from t_zero
    iris_x:          float          # Normalised [0,1]
    iris_y:          float          # Normalised [0,1]
    iris_offset_x:   float          # Iris-in-socket offset x (for SGS)
    iris_offset_y:   float          # Iris-in-socket offset y (for SGS)
    pupil_diameter:  float          # Raw MediaPipe estimate
    blink:           bool           # True if blink detected this frame
    landmark_valid:  bool           # False if MediaPipe lost tracking


@dataclass
class SessionMetadata:
    """
    Written to the sessions table before biomarker calculation begins.
    Schema is identical in local SQLite and Supabase (per local_buffer_sync spec).
    """
    session_id:       str
    child_id:         str
    video_id:         str
    t_zero_ms:        int           # Wall-clock ms of first flash detection
    sync_valid:       bool          # True if flash pattern + intervals passed
    flash_gaps_ms:    List[float]   # [gap_0_to_1, gap_1_to_2]
    video_fps:        float         # Native fps of the content video — NOT hardcoded
    created_at:       int           # Unix ms
    sync_status:      str = "local_only"


# ---------------------------------------------------------------------------
# Local SQLite — initialise tables
# ---------------------------------------------------------------------------

def init_local_db() -> None:
    """
    Creates the local SQLite tables if they do not already exist.
    Schema is kept identical to Supabase so bulk upload is a direct INSERT.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()

    # Sessions table — one row per session
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            child_id     TEXT,
            video_id     TEXT,
            t_zero_ms    INTEGER,
            sync_valid   INTEGER,   -- boolean stored as 0/1
            flash_gaps_ms TEXT,     -- JSON array: [gap_01, gap_12]
            video_fps    REAL,
            created_at   INTEGER,   -- unix ms
            sync_status  TEXT       -- local_only | sync_pending | synced
        )
    """)

    # Gaze frames table — 3600 rows per 60-second session at 60fps
    c.execute("""
        CREATE TABLE IF NOT EXISTS gaze_frames (
            session_id      TEXT,
            frame_index     INTEGER,
            timestamp_ms    INTEGER,   -- hardware timestamp, ms from t_zero
            iris_x          REAL,
            iris_y          REAL,
            iris_offset_x   REAL,
            iris_offset_y   REAL,
            pupil_diameter  REAL,
            blink           INTEGER,   -- boolean stored as 0/1
            landmark_valid  INTEGER,   -- boolean stored as 0/1
            PRIMARY KEY (session_id, frame_index)
        )
    """)

    conn.commit()
    conn.close()
    log.info("local_db: tables initialised.")


# ---------------------------------------------------------------------------
# Flash validation
# ---------------------------------------------------------------------------

def validate_flash_sync(flash: FlashDetection) -> Tuple[bool, str]:
    """
    Validates the pixel-sync flash result against the rules in
    pixel_sync_flash.docx Section 6.

    Returns (is_valid, reason_if_invalid).
    """
    # Rule 1: must have detected exactly 3 flashes
    if len(flash.t_detect) != FLASH_COUNT_REQUIRED:
        return False, (
            f"Expected {FLASH_COUNT_REQUIRED} flash detections, "
            f"got {len(flash.t_detect)}. Discard session."
        )

    # Rule 2: inter-flash intervals must be within tolerance
    for i, gap in enumerate(flash.inter_flash_gaps_ms):
        if not (INTER_FLASH_MIN_MS <= gap <= INTER_FLASH_MAX_MS):
            return False, (
                f"Inter-flash gap {i} = {gap:.1f}ms is outside "
                f"[{INTER_FLASH_MIN_MS}, {INTER_FLASH_MAX_MS}]ms tolerance. "
                f"Discard session."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Session record write
# ---------------------------------------------------------------------------

def write_session_record(meta: SessionMetadata) -> None:
    """
    Writes the session metadata row to local SQLite.
    Called BEFORE gaze_frames rows are written, so the foreign key
    relationship is intact from the first frame.
    """
    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions
            (session_id, child_id, video_id, t_zero_ms, sync_valid,
             flash_gaps_ms, video_fps, created_at, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta.session_id,
        meta.child_id,
        meta.video_id,
        meta.t_zero_ms,
        int(meta.sync_valid),
        json.dumps(meta.flash_gaps_ms),
        meta.video_fps,
        meta.created_at,
        meta.sync_status,
    ))
    conn.commit()
    conn.close()
    log.info(f"session_record: written → session_id={meta.session_id}")


# ---------------------------------------------------------------------------
# Gaze frame write — called per frame during capture
# ---------------------------------------------------------------------------

def write_gaze_frame(conn: sqlite3.Connection, frame: GazeFrame) -> None:
    """
    Writes a single iris frame to local SQLite.
    Accepts an open connection for performance — caller manages open/close
    around the capture loop (one connection for the whole session).

    Landmark coordinates are kept as full float throughout.
    No rounding or quantization at this layer (SPV sub-pixel requirement).
    """
    conn.execute("""
        INSERT INTO gaze_frames
            (session_id, frame_index, timestamp_ms,
             iris_x, iris_y, iris_offset_x, iris_offset_y,
             pupil_diameter, blink, landmark_valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        frame.session_id,
        frame.frame_index,
        frame.timestamp_ms,
        frame.iris_x,
        frame.iris_y,
        frame.iris_offset_x,
        frame.iris_offset_y,
        frame.pupil_diameter,
        int(frame.blink),
        int(frame.landmark_valid),
    ))
    # Do NOT call conn.commit() per frame — batch commit at session end
    # for performance. See flush_gaze_frames() below.


def flush_gaze_frames(conn: sqlite3.Connection) -> None:
    """
    Commits all buffered gaze_frames writes in a single transaction.
    Call once at the end of the capture loop.
    """
    conn.commit()
    log.info("gaze_frames: all frames committed to local SQLite.")


# ---------------------------------------------------------------------------
# Session integrity check
# ---------------------------------------------------------------------------

def check_session_integrity(session_id: str) -> Tuple[bool, int]:
    """
    After the capture loop ends, verifies that enough frames were written
    to support biomarker calculation.

    Returns (is_usable, actual_frame_count).
    """
    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM gaze_frames WHERE session_id = ?",
        (session_id,)
    )
    count = c.fetchone()[0]
    conn.close()

    is_usable = count >= MIN_FRAMES_FOR_VALID_SESSION
    if not is_usable:
        log.warning(
            f"integrity_check: only {count} frames for session {session_id}. "
            f"Minimum is {MIN_FRAMES_FOR_VALID_SESSION}. Session unusable."
        )
    else:
        log.info(f"integrity_check: {count} frames — session usable.")
    return is_usable, count


# ---------------------------------------------------------------------------
# Supabase sync — post-session bulk upload
# ---------------------------------------------------------------------------

def sync_session(session_id: str) -> bool:
    """
    Uploads one completed session from local SQLite to Supabase.
    Runs in a background thread so it does not block the calling code.

    Protocol (per local_buffer_sync spec v1.0):
      1. Mark session as sync_pending
      2. Serialise all gaze_frames rows to JSON
      3. POST to Supabase in a single request
      4. On 200: mark synced, delete local gaze_frames rows
      5. On failure: revert to local_only for background retry

    Returns True if sync succeeded, False otherwise.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("sync: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Skipping sync.")
        return False

    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Step 1 — mark as pending so duplicate upload is prevented on crash/retry
    c.execute(
        "UPDATE sessions SET sync_status = 'sync_pending' WHERE session_id = ?",
        (session_id,)
    )
    conn.commit()
    log.info(f"sync: {session_id} → sync_pending")

    # Step 2 — load rows
    c.execute("SELECT * FROM gaze_frames WHERE session_id = ?", (session_id,))
    frames = [dict(row) for row in c.fetchall()]

    c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session_row = dict(c.fetchone())

    if not frames:
        log.warning(f"sync: no gaze_frames for {session_id}. Aborting sync.")
        conn.close()
        return False

    log.info(f"sync: {len(frames)} gaze_frames rows ready for upload.")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    try:
        # Step 3a — upsert session row
        r_session = requests.post(
            f"{SUPABASE_URL}/rest/v1/sessions",
            headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=session_row,
            timeout=30,
        )
        r_session.raise_for_status()

        # Step 3b — bulk insert gaze_frames (single POST, ~500KB–1MB for 60s)
        r_frames = requests.post(
            f"{SUPABASE_URL}/rest/v1/gaze_frames",
            headers=headers,
            json=frames,
            timeout=60,
        )
        r_frames.raise_for_status()

        # Step 4 — mark synced and clean up local rows
        c.execute(
            "UPDATE sessions SET sync_status = 'synced' WHERE session_id = ?",
            (session_id,)
        )
        c.execute(
            "DELETE FROM gaze_frames WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        log.info(f"sync: {session_id} → synced. Local gaze_frames deleted.")
        conn.close()
        return True

    except requests.RequestException as e:
        # Step 5 — revert to local_only for background retry
        c.execute(
            "UPDATE sessions SET sync_status = 'local_only' WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        conn.close()
        log.error(f"sync: upload failed for {session_id}: {e}. Reverted to local_only.")
        return False


def sync_session_background(session_id: str) -> None:
    """
    Fires sync_session() in a daemon thread.
    The session shell does not wait for it — biomarker calculation
    runs from local SQLite and is unaffected by sync status.
    """
    t = threading.Thread(target=sync_session, args=(session_id,), daemon=True)
    t.start()
    log.info(f"sync: background thread started for {session_id}.")


def retry_pending_syncs() -> None:
    """
    Called at app start to pick up any sessions that failed to sync
    in a previous run (sync_status = 'local_only').
    Runs each in a background thread.
    """
    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE sync_status = 'local_only'")
    rows = c.fetchall()
    conn.close()

    if not rows:
        log.info("retry_pending_syncs: nothing to retry.")
        return

    log.info(f"retry_pending_syncs: {len(rows)} session(s) pending.")
    for (sid,) in rows:
        sync_session_background(sid)


# ---------------------------------------------------------------------------
# SessionShell — main orchestrator
# ---------------------------------------------------------------------------

class SessionShell:
    """
    Manages the full lifecycle of one measurement session.

    Typical call sequence (server-side, receiving data from React Native):

        shell = SessionShell(video_id="NinjaGo1", child_id="child_uuid_here")
        session_id = shell.open(flash_detection, video_fps)
        # ... React Native streams iris frames as they arrive ...
        shell.record_frame(gaze_frame)          # called per camera frame
        # ... session ends (60s elapsed or user exits) ...
        result = shell.close()
        # result.session_id → pass to biomarker_calculator.py (Layer B)
    """

    def __init__(self, video_id: str, child_id: str):
        self.video_id  = video_id
        self.child_id  = child_id
        self.session_id: Optional[str] = None
        self._db_conn:   Optional[sqlite3.Connection] = None
        self._frame_count = 0
        self._t_zero_ms: Optional[int] = None
        self._video_fps: Optional[float] = None

        # Ensure local DB tables exist
        init_local_db()

    def open(self, flash: FlashDetection, video_fps: float) -> str:
        """
        Validates the flash sync, writes the session record, opens the
        persistent SQLite connection for the capture loop.

        Raises ValueError if flash sync is invalid — do not proceed with
        a session that has no reliable t_zero.  Per the pixel_sync_flash
        spec: 'Do not attempt to calculate biomarkers from an unanchored
        iris stream.'

        Returns session_id.
        """
        # --- Flash validation ---
        is_valid, reason = validate_flash_sync(flash)
        if not is_valid:
            raise ValueError(f"SessionShell.open: flash sync invalid — {reason}")

        # --- Assign session identity ---
        self.session_id = str(uuid.uuid4())
        self._t_zero_ms = int(flash.t_detect[0])   # First flash detection = t_zero
        self._video_fps = video_fps

        log.info(
            f"session: opened  id={self.session_id}  "
            f"t_zero={self._t_zero_ms}ms  video_fps={video_fps}"
        )

        # --- Write session record to local SQLite ---
        meta = SessionMetadata(
            session_id    = self.session_id,
            child_id      = self.child_id,
            video_id      = self.video_id,
            t_zero_ms     = self._t_zero_ms,
            sync_valid    = True,
            flash_gaps_ms = flash.inter_flash_gaps_ms,
            video_fps     = video_fps,
            created_at    = int(time.time() * 1000),
        )
        write_session_record(meta)

        # --- Open persistent connection for capture loop ---
        # One connection for all frame writes avoids repeated open/close overhead
        # at 60fps. Commit is deferred until session.close().
        self._db_conn = sqlite3.connect(LOCAL_DB_PATH)
        self._frame_count = 0

        return self.session_id

    def record_frame(self, frame: GazeFrame) -> None:
        """
        Writes one iris frame to local SQLite.
        Called by the React Native bridge for every camera frame during capture.

        Landmark coordinates are stored as full float — no rounding.
        This preserves sub-pixel precision required for SPV reconstruction.
        """
        if self._db_conn is None:
            raise RuntimeError("record_frame called before open(). Call open() first.")

        # Stamp with the correct session_id and auto-incrementing frame_index
        frame.session_id  = self.session_id
        frame.frame_index = self._frame_count

        write_gaze_frame(self._db_conn, frame)
        self._frame_count += 1

    def close(self) -> dict:
        """
        Ends the capture loop:
          1. Commits all buffered gaze_frames writes
          2. Checks session integrity (minimum frame count)
          3. Triggers background Supabase sync
          4. Returns a result dict for handoff to Layer B

        Returns:
          {
            'session_id':   str,
            'usable':       bool,    # False → do not run biomarker calculation
            'frame_count':  int,
            't_zero_ms':    int,
            'video_fps':    float,
            'video_id':     str,
          }
        """
        if self._db_conn is None:
            raise RuntimeError("close() called without a matching open().")

        # Commit all frames in one transaction
        flush_gaze_frames(self._db_conn)
        self._db_conn.close()
        self._db_conn = None

        log.info(f"session: capture loop ended. {self._frame_count} frames written.")

        # Integrity check
        usable, actual_count = check_session_integrity(self.session_id)

        # Trigger background sync regardless of usability
        # (even partial sessions may have research value)
        sync_session_background(self.session_id)

        result = {
            "session_id":  self.session_id,
            "usable":      usable,
            "frame_count": actual_count,
            "t_zero_ms":   self._t_zero_ms,
            "video_fps":   self._video_fps,
            "video_id":    self.video_id,
        }

        if usable:
            log.info(
                f"session: COMPLETE  id={self.session_id}  "
                f"frames={actual_count}  → ready for biomarker calculation."
            )
        else:
            log.warning(
                f"session: UNUSABLE  id={self.session_id}  "
                f"frames={actual_count}  → skip biomarker calculation."
            )

        return result


# ---------------------------------------------------------------------------
# Background sync retry — call once at app start
# ---------------------------------------------------------------------------

# Uncomment in the main entry point of the server:
# retry_pending_syncs()