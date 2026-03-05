"""
test_session_shell.py — Standalone test for session_shell.py (Step 2, Layer A)

Simulates a complete session without React Native or a real camera.
Fabricates a valid flash detection and a synthetic iris stream, then
runs the full SessionShell call sequence and inspects the result.

Run from /workspaces/pbm/:
    python test_session_shell.py

Expected outcome:
    - No exceptions
    - local_buffer.db created in backend/
    - Sessions table: 1 row, sync_status = local_only
    - Gaze frames table: 3600 rows
    - Result dict: usable = True
    - Failure test: ValueError raised on bad flash
"""

import sqlite3
import os
import sys
import math

# Make sure the backend package is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.session_shell import (
    SessionShell, FlashDetection, GazeFrame,
    LOCAL_DB_PATH, validate_flash_sync
)

# ---------------------------------------------------------------------------
# Helper: build a synthetic iris stream (3600 frames = 60s at 60fps)
# Iris traces a slow circle — gives non-trivial x/y values without needing
# a real camera
# ---------------------------------------------------------------------------
def make_iris_stream(session_id: str, n_frames: int = 3600):
    frames = []
    for i in range(n_frames):
        t_ms = int((i / 60.0) * 1000)          # 60fps → ms elapsed from t_zero
        angle = (i / n_frames) * 2 * math.pi   # one full circle across the session
        frames.append(GazeFrame(
            session_id     = session_id,        # will be overwritten by record_frame()
            frame_index    = i,                 # will be overwritten by record_frame()
            timestamp_ms   = t_ms,
            iris_x         = 0.5 + 0.15 * math.cos(angle),
            iris_y         = 0.5 + 0.10 * math.sin(angle),
            iris_offset_x  = 0.02 * math.cos(angle),
            iris_offset_y  = 0.01 * math.sin(angle),
            pupil_diameter = 0.04 + 0.005 * math.sin(angle * 3),
            blink          = (i % 300 == 0),    # blink every 5 seconds
            landmark_valid = True,
        ))
    return frames

# ---------------------------------------------------------------------------
# Test 1 — VALID SESSION: full call sequence should complete without error
# ---------------------------------------------------------------------------
def test_valid_session():
    print("\n--- Test 1: Valid session ---")

    flash = FlashDetection(
        t_detect           = [1000.0, 1500.0, 2000.0],   # 3 flashes, 500ms apart
        inter_flash_gaps_ms = [500.0, 500.0],
        sync_valid         = True,
    )

    shell = SessionShell(video_id="NinjaGo1", child_id="test_child_001")
    session_id = shell.open(flash, video_fps=24.0)
    print(f"  session_id: {session_id}")

    frames = make_iris_stream(session_id)
    for frame in frames:
        shell.record_frame(frame)

    result = shell.close()

    print(f"  usable:      {result['usable']}")
    print(f"  frame_count: {result['frame_count']}")
    print(f"  t_zero_ms:   {result['t_zero_ms']}")
    print(f"  video_fps:   {result['video_fps']}")

    assert result["usable"] is True,        "FAIL: session should be usable"
    assert result["frame_count"] == 3600,   "FAIL: expected 3600 frames"
    assert result["t_zero_ms"] == 1000,     "FAIL: t_zero should be first flash t_detect"
    assert result["video_fps"] == 24.0,     "FAIL: video_fps mismatch"

    print("  PASSED")
    return session_id

# ---------------------------------------------------------------------------
# Test 2 — SQLITE INTEGRITY: check the DB directly after Test 1
# ---------------------------------------------------------------------------
def test_sqlite_contents(session_id: str):
    print("\n--- Test 2: SQLite contents ---")

    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()

    # Sessions table
    c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    assert row is not None,          "FAIL: session row missing from SQLite"
    print(f"  sessions row found: sync_status = {row[8]}")
    # Note: sync_status may be 'synced' or 'local_only' depending on network.
    # Either is valid — we just confirm the row exists.

    # Gaze frames table
    c.execute("SELECT COUNT(*) FROM gaze_frames WHERE session_id = ?", (session_id,))
    count = c.fetchone()[0]
    # Frames are deleted post-sync if Supabase is reachable.
    # If no Supabase env vars are set (expected in test), they remain.
    print(f"  gaze_frames row count: {count}")

    # Spot-check: first and last frame timestamps
    c.execute(
        "SELECT timestamp_ms FROM gaze_frames WHERE session_id = ? ORDER BY frame_index ASC LIMIT 1",
        (session_id,)
    )
    first_ts = c.fetchone()[0]
    c.execute(
        "SELECT timestamp_ms FROM gaze_frames WHERE session_id = ? ORDER BY frame_index DESC LIMIT 1",
        (session_id,)
    )
    last_ts = c.fetchone()[0]
    print(f"  first frame timestamp_ms: {first_ts}")
    print(f"  last  frame timestamp_ms: {last_ts}")
    assert first_ts == 0,       "FAIL: first frame should be at t=0ms"
    assert last_ts == 59983,    "FAIL: last frame should be at ~59983ms"

    conn.close()
    print("  PASSED")

# ---------------------------------------------------------------------------
# Test 3 — INVALID FLASH: bad inter-flash interval should raise ValueError
# ---------------------------------------------------------------------------
def test_invalid_flash():
    print("\n--- Test 3: Invalid flash (bad interval) ---")

    bad_flash = FlashDetection(
        t_detect            = [1000.0, 1200.0, 1400.0],  # 200ms gaps — too fast
        inter_flash_gaps_ms = [200.0, 200.0],
        sync_valid          = False,
    )

    shell = SessionShell(video_id="NinjaGo1", child_id="test_child_002")
    try:
        shell.open(bad_flash, video_fps=24.0)
        print("  FAIL: should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")
        print("  PASSED")

# ---------------------------------------------------------------------------
# Test 4 — WRONG FLASH COUNT: only 1 flash detected
# ---------------------------------------------------------------------------
def test_wrong_flash_count():
    print("\n--- Test 4: Invalid flash (only 1 detected) ---")

    bad_flash = FlashDetection(
        t_detect            = [1000.0],   # only 1 of 3 required
        inter_flash_gaps_ms = [],
        sync_valid          = False,
    )

    shell = SessionShell(video_id="NinjaGo1", child_id="test_child_003")
    try:
        shell.open(bad_flash, video_fps=24.0)
        print("  FAIL: should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")
        print("  PASSED")

# ---------------------------------------------------------------------------
# Test 5 — SHORT SESSION: fewer than 100 frames → usable = False
# ---------------------------------------------------------------------------
def test_short_session():
    print("\n--- Test 5: Short session (50 frames) ---")

    flash = FlashDetection(
        t_detect            = [5000.0, 5500.0, 6000.0],
        inter_flash_gaps_ms = [500.0, 500.0],
        sync_valid          = True,
    )

    shell = SessionShell(video_id="NinjaGo1", child_id="test_child_004")
    shell.open(flash, video_fps=24.0)

    short_frames = make_iris_stream("placeholder", n_frames=50)
    for frame in short_frames:
        shell.record_frame(frame)

    result = shell.close()
    print(f"  usable: {result['usable']}  frame_count: {result['frame_count']}")
    assert result["usable"] is False,     "FAIL: session with 50 frames should not be usable"
    assert result["frame_count"] == 50,   "FAIL: expected 50 frames"
    print("  PASSED")

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("session_shell.py — Test Suite")
    print("=" * 50)

    session_id = test_valid_session()
    test_sqlite_contents(session_id)
    test_invalid_flash()
    test_wrong_flash_count()
    test_short_session()

    print("\n" + "=" * 50)
    print("All tests passed.")
    print("=" * 50)
    print(f"\nlocal_buffer.db written to: {LOCAL_DB_PATH}")
    print("Inspect it with: sqlite3 backend/local_buffer.db")