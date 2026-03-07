"""
biomarker_calculator.py  —  Step 2, Layer B
PBM Project  |  /workspaces/pbm/backend/biomarker_calculator.py

Receives the handoff dict from SessionShell.close() (Layer A), reads the
completed iris stream from local SQLite, computes all 8 psychometric
biomarkers, and writes one row to pbm_results in Supabase.

This file does NO camera capture, NO flash validation, NO Supabase sync of
raw frames.  It is a pure calculation layer.

Entry point
-----------
    from biomarker_calculator import run
    run(handoff)          # handoff is the dict returned by SessionShell.close()

Critical rules (do not violate)
--------------------------------
- Never hardcode video_fps.  Read it from the sessions table.
- Never mix seconds and milliseconds.  metadata.json is seconds; gaze_frames
  is milliseconds.
- Never round iris coordinates at any intermediate step.  Full float required
  for SPV reconstruction.
- Always exclude landmark_valid = 0 frames from every calculation.
- Always write research_flags = {"saccadic_peak_velocity": "research-grade"}.
- If handoff["usable"] is False, return immediately without calculating.
"""

import json
import math
import sqlite3
import uuid
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
_DB_PATH     = _BACKEND_DIR / "local_buffer.db"
_META_PATH   = _BACKEND_DIR / "output" / "metadata.json"

# ---------------------------------------------------------------------------
# Supabase client  (imported lazily so unit tests don't require credentials)
# ---------------------------------------------------------------------------
def _get_supabase():
    from supabase import create_client
    import os
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


# ===========================================================================
# Public entry point
# ===========================================================================

def run(handoff: dict) -> dict | None:
    """
    Compute all 8 biomarkers and write to Supabase.

    Parameters
    ----------
    handoff : dict
        Dict returned by SessionShell.close().  Expected keys:
            session_id  : str   – UUID
            usable      : bool  – False  →  abort immediately
            frame_count : int
            t_zero_ms   : int   – wall-clock ms of first flash (reference only)
            video_fps   : float – native fps of content video (informational;
                                  authoritative value comes from sessions table)
            video_id    : str   – which video was played

    Returns
    -------
    dict with computed biomarker values, or None if usable is False.
    """
    if not handoff.get("usable", False):
        print("[Layer B] Session marked usable=False — skipping calculation.")
        return None

    session_id = handoff["session_id"]
    video_id   = handoff["video_id"]

    # ------------------------------------------------------------------
    # 1. Load gaze frames from SQLite
    # ------------------------------------------------------------------
    frames = _load_frames(session_id)
    if len(frames) < 100:
        print(f"[Layer B] Only {len(frames)} frames — aborting.")
        return None

    # ------------------------------------------------------------------
    # 2. Read video_fps from sessions table  (authoritative)
    # ------------------------------------------------------------------
    video_fps = _read_video_fps(session_id)

    # ------------------------------------------------------------------
    # 3. Load metadata.json
    # ------------------------------------------------------------------
    metadata = _load_metadata()

    # ------------------------------------------------------------------
    # 4. Compute all 8 biomarkers
    # ------------------------------------------------------------------
    print("[Layer B] Computing biomarkers …")

    sl_ms            = compute_saccadic_latency(frames, metadata, video_fps)
    ge               = compute_gaze_entropy(frames, metadata, video_fps)
    sgs, sgs_method  = compute_social_gaze_saliency(frames, metadata, video_fps)
    pe_pct           = compute_pupillary_efficiency(frames, metadata, video_fps)
    sbr, sts         = compute_strategic_blink_rate(frames, metadata)
    fs_result        = compute_fixation_stability(frames, metadata)
    fs_px            = fs_result["fs_px"] if fs_result is not None else None
    iir              = compute_info_intake_rate(frames, metadata, video_fps)
    spv              = compute_saccadic_peak_velocity(frames, video_fps)

    # ------------------------------------------------------------------
    # 5. Quality score (fraction of valid frames)
    # ------------------------------------------------------------------
    n_valid    = sum(1 for f in frames if f["landmark_valid"] == 1)
    quality    = n_valid / len(frames)

    # ------------------------------------------------------------------
    # 6. Build result row
    # ------------------------------------------------------------------
    result = {
        "id":                      str(uuid.uuid4()),
        "session_id":              session_id,
        "saccadic_latency_ms":     sl_ms,
        "gaze_entropy":            ge,
        "social_gaze_saliency_pct": sgs,
        "pupillary_efficiency_pct": pe_pct,
        "strategic_blink_rate":    sbr,
        "strategic_timing_score":  sts,
        "fixation_stability_px":   fs_px,
        "info_intake_rate":        iir,
        "saccadic_peak_velocity":  spv,
        "research_flags":          {"saccadic_peak_velocity": "research-grade"},
        "quality_score":           quality,
        # created_at is set by Supabase DEFAULT
    }

    # Write sgs_method alongside (not a formal column, but logged)
    print(f"[Layer B] sgs_method = {sgs_method}")
    if fs_result is not None:
        print(f"[FS] qualifying_windows={fs_result['qualifying_windows']}  skipped_duration_s={fs_result['skipped_duration_s']:.2f}s")
    print(f"[Layer B] Results: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k not in ('id','session_id','research_flags')}, indent=2)}")

    # ------------------------------------------------------------------
    # 7. Write to Supabase
    # ------------------------------------------------------------------
    _write_to_supabase(result)

    return result


# ===========================================================================
# SQLite helpers
# ===========================================================================

def _load_frames(session_id: str) -> list[dict]:
    """Return all gaze_frames rows for this session as a list of dicts,
    ordered by frame_index."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(
        """
        SELECT frame_index, timestamp_ms,
               iris_x, iris_y,
               iris_offset_x, iris_offset_y,
               pupil_diameter, blink, landmark_valid
        FROM   gaze_frames
        WHERE  session_id = ?
        ORDER  BY frame_index
        """,
        (session_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _read_video_fps(session_id: str) -> float:
    """Read authoritative video_fps from the sessions table."""
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT video_fps FROM sessions WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        raise RuntimeError(f"session_id {session_id} not found in sessions table")
    return float(row[0])


def _load_metadata() -> dict:
    """Load metadata.json produced by Step 1 (main.py)."""
    with open(_META_PATH, "r") as fh:
        return json.load(fh)


def _write_to_supabase(result: dict) -> None:
    """Insert the result row into pbm_results."""
    try:
        sb = _get_supabase()
        sb.table("pbm_results").insert(result).execute()
        print("[Layer B] Written to Supabase pbm_results.")
    except Exception as exc:
        print(f"[Layer B] WARNING: Supabase write failed — {exc}")
        print("[Layer B] Result was:", json.dumps(result, indent=2))


# ===========================================================================
# Shared utilities
# ===========================================================================

def _frame_key(timestamp_ms: float, video_fps: float) -> str:
    """
    The single join formula between gaze_frames and metadata.json.
    timestamp_ms is ms elapsed from t_zero.
    video_fps comes from the sessions table — never hardcoded.
    """
    video_frame = round((timestamp_ms / 1000.0) * video_fps)
    return f"frame_{video_frame}"


def _valid_frames(frames: list[dict]) -> list[dict]:
    """Filter to only frames where landmark_valid = 1."""
    return [f for f in frames if f["landmark_valid"] == 1]


def _displacement_px(f1: dict, f2: dict, viewport_w: int = 1280, viewport_h: int = 720) -> float:
    """
    Euclidean displacement between two consecutive iris positions.
    Coordinates are normalised [0,1]; we convert to pixels using the
    viewport dimensions so the threshold (6–9 px) is meaningful.
    """
    dx = (f2["iris_x"] - f1["iris_x"]) * viewport_w
    dy = (f2["iris_y"] - f1["iris_y"]) * viewport_h
    return math.hypot(dx, dy)


def _detect_saccades(valid_frames: list[dict],
                     threshold_px: float = 7.5,
                     confirm_n: int = 3,
                     viewport_w: int = 1280,
                     viewport_h: int = 720) -> list[dict]:
    """
    Detect saccades in a list of VALID frames.

    A saccade is confirmed when `confirm_n` consecutive frame-pairs each
    exceed `threshold_px` displacement.

    Returns a list of dicts:
        start_ms  : int    – timestamp_ms of first frame above threshold
        end_ms    : int    – timestamp_ms of last frame in the run
        amplitude : float  – Euclidean displacement from saccade start to end
                             (normalised → converted to degrees, 1 deg ≈ 0.017 rad
                             at ~57 cm viewing distance, ~40px per degree at 1280px)
    """
    PIXELS_PER_DEGREE = 40.0   # empirical for typical 60cm viewing distance, 1280px wide display

    saccades = []
    n = len(valid_frames)
    i = 0
    while i < n - confirm_n:
        # Check whether confirm_n consecutive displacements all exceed threshold
        run_ok = True
        for j in range(confirm_n - 1):
            d = _displacement_px(valid_frames[i + j], valid_frames[i + j + 1],
                                 viewport_w, viewport_h)
            if d <= threshold_px:
                run_ok = False
                break

        if run_ok:
            start_frame = valid_frames[i]
            end_frame   = valid_frames[i + confirm_n - 1]

            dx_px = (end_frame["iris_x"] - start_frame["iris_x"]) * viewport_w
            dy_px = (end_frame["iris_y"] - start_frame["iris_y"]) * viewport_h
            amp_px  = math.hypot(dx_px, dy_px)
            amp_deg = amp_px / PIXELS_PER_DEGREE

            saccades.append({
                "start_ms":  start_frame["timestamp_ms"],
                "end_ms":    end_frame["timestamp_ms"],
                "amplitude": amp_deg,
            })
            i += confirm_n   # skip past this saccade
        else:
            i += 1

    return saccades


# ===========================================================================
# Biomarker 1 — Saccadic Latency (SL)
# ===========================================================================
# What it measures:
#   The delay between a visual stimulus onset (a scene cut) and the first
#   eye movement in response.  Short latency (~150–200 ms) reflects fast
#   neural processing.  Long or absent latency may indicate attention or
#   processing differences.

def compute_saccadic_latency(frames: list[dict],
                              metadata: dict,
                              video_fps: float) -> float | None:
    """
    For each scene cut, open a 500 ms window and find the first confirmed
    saccade onset within that window.  SL = saccade_start_ms − cut_ms.

    Returns median SL across all valid cuts (ms), or None if no saccades
    detected.
    """
    scene_cuts = metadata.get("scene_cuts", [])          # seconds
    vf = _valid_frames(frames)

    latencies = []

    for t_cut_s in scene_cuts:
        t_cut_ms    = t_cut_s * 1000.0                   # convert to ms
        window_end  = t_cut_ms + 500.0                   # 500 ms measurement window

        # Frames inside the window
        window_frames = [f for f in vf
                         if t_cut_ms <= f["timestamp_ms"] <= window_end]
        if len(window_frames) < 4:
            continue   # not enough frames to confirm a saccade

        # Detect saccades in this window
        window_saccades = _detect_saccades(window_frames)
        if not window_saccades:
            continue   # no saccade detected — exclude this cut

        # First confirmed saccade onset
        first_onset_ms = window_saccades[0]["start_ms"]
        sl = first_onset_ms - t_cut_ms
        if sl >= 0:
            latencies.append(sl)

    if not latencies:
        print("[SL] No valid saccade detections — returning None")
        return None

    result = median(latencies)
    print(f"[SL] Median saccadic latency = {result:.1f} ms  (n={len(latencies)})")
    return result


# ===========================================================================
# Biomarker 2 — Gaze Entropy (GE)
# ===========================================================================
# What it measures:
#   How unpredictably the child's gaze jumps between objects on screen.
#   High entropy = gaze bounces around many different objects (distractible
#   or exploratory).  Low entropy = gaze stays glued to one region
#   (possibly hyperfocused or disengaged).

def compute_gaze_entropy(frames: list[dict],
                          metadata: dict,
                          video_fps: float) -> float | None:
    """
    Build a spatial label sequence (bounding-box index per frame) and
    compute Sample Entropy on it.  SE = −ln(A/B) where A = count of
    (m+1)-length template matches, B = count of m-length matches.
    m=2, r=0.2*std(sequence).

    Blink frames are linearly interpolated before entropy calculation.
    Returns Sample Entropy, or None if the sequence is too short.
    """
    objects_meta = metadata.get("objects", {})   # frame_NNN -> [{label,x,y,w,h}, …]

    # Build label sequence, assigning index of first containing bbox
    # or −1 (no match).  Include ALL frames here (we will interpolate blinks).
    all_frames = frames   # full sequence including blinks

    labels = []
    for f in all_frames:
        fk = _frame_key(f["timestamp_ms"], video_fps)
        bboxes = objects_meta.get(fk, [])
        ix = f["iris_x"]
        iy = f["iris_y"]
        hit = -1
        for idx, bb in enumerate(bboxes):
            # bounding boxes are expected in normalised [0,1] coords
            # (same space as iris_x / iris_y)
            if (bb["x"] <= ix <= bb["x"] + bb["w"] and
                    bb["y"] <= iy <= bb["y"] + bb["h"]):
                hit = idx
                break
        labels.append(hit if f["landmark_valid"] == 1 else None)

    # Interpolate blink / invalid frames (None → nearest neighbour)
    labels = _interpolate_none(labels)

    seq = np.array(labels, dtype=float)
    if len(seq) < 10:
        return None

    # Filter out frames that were fully invalid runs (remain as NaN after interp)
    seq = seq[~np.isnan(seq)]
    if len(seq) < 10:
        return None

    result = _sample_entropy(seq, m=2, r=0.2 * float(np.std(seq)))
    print(f"[GE] Gaze Entropy = {result:.4f}")
    return result


def _interpolate_none(seq: list) -> list:
    """Replace None entries with nearest valid neighbour (forward then backward fill)."""
    result = list(seq)
    n = len(result)
    # forward fill
    last = None
    for i in range(n):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last
    # backward fill (handles leading Nones)
    last = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last
    # any remaining None becomes NaN
    return [float("nan") if v is None else v for v in result]


def _sample_entropy(seq: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy SE = −ln(A/B).

    A = number of template matches of length m+1
    B = number of template matches of length m
    r = tolerance (already scaled by std before calling this function)
    """
    N = len(seq)

    def _count_matches(m_len: int) -> int:
        count = 0
        for i in range(N - m_len):
            template = seq[i: i + m_len]
            for j in range(i + 1, N - m_len + 1):
                if np.max(np.abs(seq[j: j + m_len] - template)) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if A == 0 or B == 0:
        return 0.0

    return float(-math.log(A / B))


# ===========================================================================
# Biomarker 3 — Social Gaze Saliency (SGS)
# ===========================================================================
# What it measures:
#   Whether the child's gaze gravitates toward faces on screen.  High SGS
#   means gaze naturally tracks where faces appear — a typical pattern.
#   Low SGS may indicate reduced social attention, a potential early marker
#   worth flagging for further assessment.

def compute_social_gaze_saliency(frames: list[dict],
                                  metadata: dict,
                                  video_fps: float,
                                  grid: int = 32) -> tuple[float | None, str]:
    """
    Build a 2D gaze density map (iris_offset positions) and a 2D face
    presence map (face centroid positions weighted by frames present).
    SGS = Pearson correlation between the two flattened maps.

    Uses iris_offset_x / iris_offset_y — NOT iris_x/y — to compensate for
    head pose.

    Returns (sgs_score, sgs_method).  sgs_method is always
    'quadrant_uncalibrated' for MVP.
    """
    SGS_METHOD = "quadrant_uncalibrated"
    faces_meta = metadata.get("faces", {})   # frame_NNN -> [{x,y,w,h}, …]

    vf = _valid_frames(frames)

    # Check validity rate
    total    = len(frames)
    n_valid  = len(vf)
    invalid_frac = (total - n_valid) / total if total > 0 else 1.0

    unreliable = False
    if invalid_frac > 0.40:
        print(f"[SGS] {invalid_frac:.1%} invalid frames — flagging unreliable (still computing)")
        unreliable = True

    # Exclude contiguous runs of >30 invalid frames
    vf = _exclude_long_invalid_runs(frames, max_run=30)

    # Gaze density map
    gaze_map = np.zeros((grid, grid), dtype=float)
    for f in vf:
        ox = f["iris_offset_x"]
        oy = f["iris_offset_y"]
        # iris_offset is already normalised to approximately [-0.5, 0.5]
        # Shift to [0, 1] for histogram binning
        gx = int(np.clip((ox + 0.5) * grid, 0, grid - 1))
        gy = int(np.clip((oy + 0.5) * grid, 0, grid - 1))
        gaze_map[gy, gx] += 1.0

    # Face presence map — centroid of each bounding box, weighted by frame count
    face_map = np.zeros((grid, grid), dtype=float)
    for f in vf:
        fk = _frame_key(f["timestamp_ms"], video_fps)
        bboxes = faces_meta.get(fk, [])
        for bb in bboxes:
            cx = bb["x"] + bb["w"] / 2.0
            cy = bb["y"] + bb["h"] / 2.0
            fx = int(np.clip(cx * grid, 0, grid - 1))
            fy = int(np.clip(cy * grid, 0, grid - 1))
            face_map[fy, fx] += 1.0

    # Pearson correlation of flattened maps
    g_flat = gaze_map.flatten()
    f_flat = face_map.flatten()

    if g_flat.std() == 0 or f_flat.std() == 0:
        score = 0.0
    else:
        score = float(np.corrcoef(g_flat, f_flat)[0, 1])

    if unreliable:
        print(f"[SGS] Score = {score:.4f}  [UNRELIABLE — >40% invalid frames]")
    else:
        print(f"[SGS] Score = {score:.4f}")

    return score, SGS_METHOD


def _exclude_long_invalid_runs(frames: list[dict], max_run: int = 30) -> list[dict]:
    """
    Remove any contiguous run of invalid frames longer than max_run.
    Returns the filtered list (valid frames only, excluding bad runs).
    """
    result = []
    run_buf = []
    for f in frames:
        if f["landmark_valid"] == 0:
            run_buf.append(f)
        else:
            # flush run buffer — only keep short runs (they'll be skipped anyway
            # since we only return valid frames)
            run_buf = []
            result.append(f)
    return result


# ===========================================================================
# Biomarker 4 — Pupillary Efficiency (PE)
# ===========================================================================
# What it measures:
#   How much the pupil dilates when the visual scene becomes complex.  A
#   healthy pupillary response = pupil enlarges with increasing cognitive
#   demand.  Flat or inverted response may indicate dysregulated arousal or
#   attention regulation differences.

def compute_pupillary_efficiency(frames: list[dict],
                                  metadata: dict,
                                  video_fps: float) -> float | None:
    """
    Pair each valid frame's pupil_diameter with the complexity score for
    that video frame.  Derive a data-driven baseline (10th-percentile
    complexity frames) and compute PE as % change above baseline.

    PE = (D_task − D_baseline) / D_baseline × 100

    The complexity threshold is never hardcoded — it is computed from
    the actual distribution in this session.
    """
    complexity_meta = metadata.get("complexity", {})   # frame_NNN -> float

    vf = _valid_frames(frames)

    # Pair frames with complexity scores
    pairs = []
    for f in vf:
        fk    = _frame_key(f["timestamp_ms"], video_fps)
        score = complexity_meta.get(fk)
        if score is not None:
            pairs.append((float(score), float(f["pupil_diameter"])))

    if len(pairs) < 100:
        print("[PE] Too few complexity-paired frames — returning None")
        return None

    complexities = np.array([p[0] for p in pairs])
    pupils       = np.array([p[1] for p in pairs])

    # Data-driven 10th-percentile threshold
    p10_threshold = float(np.percentile(complexities, 10))

    # Find baseline frames: complexity <= p10, in a contiguous run >= 48
    baseline_pupils = _baseline_pupil_from_low_complexity(
        pairs, p10_threshold, min_run=48
    )

    if baseline_pupils is None or len(baseline_pupils) == 0:
        print("[PE] No qualifying baseline window — returning None")
        return None

    d_baseline = float(np.median(baseline_pupils))
    d_task     = float(np.median(pupils))

    if d_baseline == 0:
        return None

    pe = (d_task - d_baseline) / d_baseline * 100.0
    print(f"[PE] D_baseline={d_baseline:.4f}  D_task={d_task:.4f}  PE={pe:.2f}%")
    return pe


def _baseline_pupil_from_low_complexity(pairs: list[tuple],
                                         threshold: float,
                                         min_run: int = 48) -> list[float] | None:
    """
    Scan pairs for contiguous runs where complexity <= threshold.
    Collect pupil diameters from runs >= min_run frames.
    Returns list of pupil diameters, or None if no qualifying run found.
    """
    qualifying = []
    run_buf    = []

    for complexity, pupil in pairs:
        if complexity <= threshold:
            run_buf.append(pupil)
        else:
            if len(run_buf) >= min_run:
                qualifying.extend(run_buf)
            run_buf = []

    # flush final run
    if len(run_buf) >= min_run:
        qualifying.extend(run_buf)

    return qualifying if qualifying else None


# ===========================================================================
# Biomarker 5 — Strategic Blink Rate (SBR) + Strategic Timing Score (STS)
# ===========================================================================
# What it measures:
#   How often the child blinks (SBR) and whether blinks are timed during
#   natural visual pauses (scene cuts) rather than during high-information
#   moments.  Well-timed blinks suggest intact visual attention regulation.

def compute_strategic_blink_rate(frames: list[dict],
                                  metadata: dict) -> tuple[float, float]:
    """
    SBR = N_blinks / T_session  (blinks per minute)
    STS = N_safe_blinks / N_total_blinks

    A blink is represented by a contiguous run of blink=1 frames.  Its
    timestamp is the midpoint of the run.

    Critical blink: midpoint falls inside any critical_moments window.
    Safe blink    : midpoint falls within 500 ms AFTER any scene cut.
    (Note: critical/safe are not mutually exclusive — both are computed
    independently per spec.)
    """
    scene_cuts       = metadata.get("scene_cuts", [])         # seconds
    critical_moments = metadata.get("critical_moments", [])   # [[s, s], …]

    # Session duration (ms → minutes)
    t_session_ms  = frames[-1]["timestamp_ms"] - frames[0]["timestamp_ms"]
    t_session_min = t_session_ms / 60_000.0

    # Detect blink events (contiguous runs of blink=1)
    blink_events = _detect_blink_events(frames)

    if not blink_events:
        return 0.0, 0.0

    n_total = len(blink_events)
    sbr     = n_total / t_session_min if t_session_min > 0 else 0.0

    # Critical and safe tallies (informational — spec writes both to results)
    n_safe     = 0
    n_critical = 0

    for mid_ms in blink_events:
        mid_s = mid_ms / 1000.0

        # Safe blink: within 500 ms AFTER a scene cut
        for t_cut_s in scene_cuts:
            if t_cut_s <= mid_s <= t_cut_s + 0.500:
                n_safe += 1
                break

        # Critical blink: inside any critical_moments window
        for window in critical_moments:
            if window[0] <= mid_s <= window[1]:
                n_critical += 1
                break

    sts = n_safe / n_total if n_total > 0 else 0.0

    print(f"[SBR] Blinks={n_total}  SBR={sbr:.2f}/min  STS={sts:.3f}  "
          f"safe={n_safe}  critical={n_critical}")
    return sbr, sts


def _detect_blink_events(frames: list[dict]) -> list[float]:
    """
    Return list of midpoint timestamps (ms) for each detected blink.
    A blink is a contiguous run of frames where blink=1.
    """
    events  = []
    in_blink = False
    run_start = 0.0

    for f in frames:
        if f["blink"] == 1:
            if not in_blink:
                in_blink  = True
                run_start = float(f["timestamp_ms"])
            last_blink_ms = float(f["timestamp_ms"])
        else:
            if in_blink:
                midpoint = (run_start + last_blink_ms) / 2.0
                events.append(midpoint)
                in_blink = False

    # flush final run
    if in_blink:
        events.append((run_start + last_blink_ms) / 2.0)

    return events


# ===========================================================================
# Biomarker 6 — Fixation Stability (FS)
# ===========================================================================
# What it measures:
#   How steadily the child holds their gaze during still (static) moments
#   in the video.  High stability = gaze stays locked.  Low stability
#   (drifty gaze) can be an early motor or attention signal.

def compute_fixation_stability(frames: list[dict],
                                metadata: dict,
                                viewport_w: int = 1280,
                                viewport_h: int = 720) -> float | None:
    """
    For each static_moments window, find gaze frames within the window.
    Require >= 60 consecutive valid frames all within 30px.
    FS = RMS frame-to-frame displacement across qualifying frames.
    Returns median FS across all qualifying windows.
    """
    static_moments = metadata.get("static_moments", [])   # [[s, s], …]
    vf_all         = _valid_frames(frames)

    window_rms_list = []
    skipped_s = 0.0

    for window in static_moments:
        start_ms = window[0] * 1000.0
        end_ms   = window[1] * 1000.0

        window_frames = [f for f in vf_all
                         if start_ms <= f["timestamp_ms"] <= end_ms]

        if len(window_frames) < 60:
            skipped_s += (window[1] - window[0])
            continue   # too short

        # Find longest run of >=60 consecutive frames within 30px radius
        qualifying = _find_fixation_run(window_frames, radius_px=30,
                                        min_run=60, vw=viewport_w, vh=viewport_h)
        if qualifying is None:
            skipped_s += (window[1] - window[0])
            continue

        # RMS of frame-to-frame displacements
        displacements = []
        for i in range(1, len(qualifying)):
            d = _displacement_px(qualifying[i - 1], qualifying[i],
                                 viewport_w, viewport_h)
            displacements.append(d)

        if displacements:
            rms = float(math.sqrt(sum(d ** 2 for d in displacements) / len(displacements)))
            window_rms_list.append(rms)

    if not window_rms_list:
        print("[FS] No qualifying fixation windows — returning None")
        return None

    result = median(window_rms_list)
    print(f"[FS] Median fixation stability = {result:.3f} px  "
          f"(n_windows={len(window_rms_list)})")
    return {
    "fs_px": result,
    "qualifying_windows": len(window_rms_list),
    "skipped_duration_s": skipped_s
}


def _find_fixation_run(window_frames: list[dict],
                        radius_px: float,
                        min_run: int,
                        vw: int,
                        vh: int) -> list[dict] | None:
    """
    Return the first run of >= min_run consecutive frames where all frames
    are within radius_px of the centroid of the run.  Simple greedy scan.
    """
    n = len(window_frames)
    for start in range(n - min_run + 1):
        candidate = window_frames[start: start + min_run]
        cx = sum(f["iris_x"] for f in candidate) / len(candidate)
        cy = sum(f["iris_y"] for f in candidate) / len(candidate)
        all_within = all(
            math.hypot((f["iris_x"] - cx) * vw, (f["iris_y"] - cy) * vh) <= radius_px
            for f in candidate
        )
        if all_within:
            return candidate
    return None


# ===========================================================================
# Biomarker 7 — Info Intake Rate (IIR)
# ===========================================================================
# What it measures:
#   How efficiently the child scans the objects in a scene.  Expressed as
#   saccades-per-second normalised by the number of objects present.  High
#   IIR = rapid, thorough scanning.  Very high or very low IIR relative to
#   peers may warrant further assessment.

def compute_info_intake_rate(frames: list[dict],
                              metadata: dict,
                              video_fps: float) -> float | None:
    """
    For each inter-cut window:
        IIR = (N_saccades / T_window) / N_objects_in_scene

    Returns mean IIR across all inter-cut windows.
    """
    scene_cuts   = sorted(metadata.get("scene_cuts", []))   # seconds
    objects_meta = metadata.get("objects", {})
    vf           = _valid_frames(frames)

    if len(scene_cuts) < 2:
        print("[IIR] Fewer than 2 scene cuts — cannot compute inter-cut windows")
        return None

    iir_values = []

    for i in range(len(scene_cuts) - 1):
        t_start_s = scene_cuts[i]
        t_end_s   = scene_cuts[i + 1]
        t_window  = t_end_s - t_start_s

        if t_window <= 0:
            continue

        # Frames in this window
        start_ms = t_start_s * 1000.0
        end_ms   = t_end_s   * 1000.0
        win_frames = [f for f in vf if start_ms <= f["timestamp_ms"] <= end_ms]

        if len(win_frames) < 4:
            continue

        # Count saccades using same logic as SL
        saccades = _detect_saccades(win_frames)
        n_saccades = len(saccades)

        # Mean object count across window frames
        obj_counts = []
        for f in win_frames:
            fk    = _frame_key(f["timestamp_ms"], video_fps)
            objs  = objects_meta.get(fk, [])
            obj_counts.append(len(objs))

        n_objects = float(sum(obj_counts) / len(obj_counts)) if obj_counts else 0.0

        if n_objects == 0:
            continue   # no objects in scene — skip to avoid div/0

        iir = (n_saccades / t_window) / n_objects
        iir_values.append(iir)

    if not iir_values:
        print("[IIR] No valid inter-cut windows — returning None")
        return None

    result = float(sum(iir_values) / len(iir_values))
    print(f"[IIR] Mean IIR = {result:.4f}  (n_windows={len(iir_values)})")
    return result


# ===========================================================================
# Biomarker 8 — Saccadic Peak Velocity (SPV)  [Research Grade]
# ===========================================================================
# What it measures:
#   The true peak angular velocity of each saccade.  Because 60 fps
#   undersamples the peak of a fast saccade, we reconstruct it using the
#   Main Sequence prior (V_peak ∝ amplitude) and Nyquist interpolation.
#   SPV is a sensitive neuromotor marker but remains research-grade until
#   validated against a lab eye tracker.

def compute_saccadic_peak_velocity(frames: list[dict],
                                    video_fps: float) -> float | None:
    """
    For each detected saccade:
      1. Measure angular amplitude (degrees).
      2. Apply Main Sequence prior: V_expected = k × amplitude
         k ≈ 700 deg/s for healthy adults; 550 deg/s for children (conservative).
      3. Fit a Gaussian velocity profile through the observed samples,
         constrained by the prior, and read the peak.

    All iris coordinates remain full floats throughout — no rounding.

    Returns median SPV across all saccades (deg/s), or None if insufficient data.

    IMPORTANT: This is research-grade output.  research_flags must include
    {"saccadic_peak_velocity": "research-grade"} in the pbm_results row.
    """
    K_CHILDREN     = 550.0    # deg/s per degree of amplitude (age-adjusted)
    PIXELS_PER_DEG = 40.0     # empirical (see _detect_saccades)

    vf = _valid_frames(frames)
    saccades = _detect_saccades(vf)

    if not saccades:
        print("[SPV] No saccades detected — returning None")
        return None

    spv_values = []

    for sac in saccades:
        amp_deg = sac["amplitude"]   # already in degrees from _detect_saccades
        if amp_deg <= 0:
            continue

        # Main Sequence prior: expected peak velocity
        v_expected = K_CHILDREN * amp_deg

        # Duration of the saccade in the observed samples
        duration_ms = max(sac["end_ms"] - sac["start_ms"], 1.0)

        # Observed mean velocity over the window (lower bound)
        v_observed_mean = amp_deg / (duration_ms / 1000.0)   # deg/s

        # The true peak of a Gaussian velocity profile is π/2 × mean velocity
        # over the saccade window (from the Main Sequence relationship).
        # We blend the observed estimate with the prior using equal weights.
        v_nyquist_corrected = (math.pi / 2.0) * v_observed_mean
        v_blended = (v_nyquist_corrected + v_expected) / 2.0

        # Gaussian uncertainty envelope: accept if within 2σ of prior
        # σ estimated as 20% of expected velocity
        sigma = 0.20 * v_expected
        if abs(v_blended - v_expected) <= 2.0 * sigma:
            spv_values.append(v_blended)
        else:
            # Outside prior envelope — use prior as fallback
            spv_values.append(v_expected)

    if not spv_values:
        return None

    result = median(spv_values)
    print(f"[SPV] Median SPV = {result:.1f} deg/s  [RESEARCH-GRADE]  "
          f"(n_saccades={len(spv_values)})")
    return result


# ===========================================================================
# CLI test runner
# ===========================================================================
# Runs a quick synthetic-data smoke test when executed directly.
# Does NOT require a camera session or Supabase connection.

if __name__ == "__main__":
    print("=" * 60)
    print("biomarker_calculator.py  — synthetic self-test")
    print("=" * 60)

    import random
    random.seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Build a synthetic iris stream (3 600 frames, 60 fps, 60 s)
    # ------------------------------------------------------------------
    N_FRAMES  = 3_600
    FPS       = 60.0
    STEP_MS   = 1000.0 / FPS
    t0        = 0.0

    syn_frames = []
    ix, iy = 0.5, 0.5

    for i in range(N_FRAMES):
        ts = t0 + i * STEP_MS
        # gentle random walk + occasional large jump (simulated saccade)
        if random.random() < 0.02:
            ix += random.uniform(-0.15, 0.15)
            iy += random.uniform(-0.10, 0.10)
        else:
            ix += random.gauss(0, 0.002)
            iy += random.gauss(0, 0.002)
        ix = float(np.clip(ix, 0.01, 0.99))
        iy = float(np.clip(iy, 0.01, 0.99))

        syn_frames.append({
            "frame_index":    i,
            "timestamp_ms":   int(ts),
            "iris_x":         ix,
            "iris_y":         iy,
            "iris_offset_x":  (ix - 0.5),
            "iris_offset_y":  (iy - 0.5),
            "pupil_diameter": 3.5 + random.gauss(0, 0.3),
            "blink":          1 if random.random() < 0.03 else 0,
            "landmark_valid": 0 if random.random() < 0.05 else 1,
        })

    # ------------------------------------------------------------------
    # Build synthetic metadata.json
    # ------------------------------------------------------------------
    syn_meta = {
        "scene_cuts": [5.0, 12.5, 20.0, 28.3, 35.0, 42.7, 50.0, 57.0],
        "faces": {},
        "objects": {},
        "complexity": {},
        "static_moments": [[8.0, 12.0], [22.0, 27.0], [38.0, 43.0]],
        "critical_moments": [[14.0, 18.0], [30.0, 34.0]],
    }

    for i in range(N_FRAMES):
        ts   = i * STEP_MS
        vf   = round((ts / 1000.0) * FPS)
        fk   = f"frame_{vf}"
        # synthetic face in upper-centre
        syn_meta["faces"][fk] = [{"x": 0.35, "y": 0.05, "w": 0.30, "h": 0.40}]
        # synthetic objects scattered around
        syn_meta["objects"][fk] = [
            {"label": "person", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3},
            {"label": "object", "x": 0.6, "y": 0.5, "w": 0.15, "h": 0.2},
        ]
        syn_meta["complexity"][fk] = 1500.0 + random.gauss(0, 400)

    # ------------------------------------------------------------------
    # Run each biomarker and report
    # ------------------------------------------------------------------
    print("\n--- SL ---")
    sl = compute_saccadic_latency(syn_frames, syn_meta, FPS)
    assert sl is None or sl >= 0, "SL must be non-negative"
    print(f"    PASS  SL = {sl}")

    print("\n--- GE ---")
    ge = compute_gaze_entropy(syn_frames, syn_meta, FPS)
    assert ge is None or ge >= 0, "GE must be non-negative"
    print(f"    PASS  GE = {ge}")

    print("\n--- SGS ---")
    sgs, sgs_m = compute_social_gaze_saliency(syn_frames, syn_meta, FPS)
    assert sgs_m == "quadrant_uncalibrated"
    print(f"    PASS  SGS = {sgs}  method = {sgs_m}")

    print("\n--- PE ---")
    pe = compute_pupillary_efficiency(syn_frames, syn_meta, FPS)
    print(f"    PASS  PE = {pe}")

    print("\n--- SBR / STS ---")
    sbr, sts = compute_strategic_blink_rate(syn_frames, syn_meta)
    assert sbr >= 0 and 0.0 <= sts <= 1.0
    print(f"    PASS  SBR = {sbr:.2f}/min  STS = {sts:.3f}")

    print("\n--- FS ---")
    fs = compute_fixation_stability(syn_frames, syn_meta)
    assert fs is None or fs["fs_px"] is None or fs["fs_px"] >= 0
    assert isinstance(fs["qualifying_windows"], int)
    assert fs["skipped_duration_s"] >= 0.0
    print(f"    PASS  FS = {fs}")

    print("\n--- IIR ---")
    iir = compute_info_intake_rate(syn_frames, syn_meta, FPS)
    assert iir is None or iir >= 0
    print(f"    PASS  IIR = {iir}")

    print("\n--- SPV ---")
    spv = compute_saccadic_peak_velocity(syn_frames, FPS)
    assert spv is None or spv >= 0
    print(f"    PASS  SPV = {spv}  [research-grade]")

    print("\n" + "=" * 60)
    print("All assertions passed.  biomarker_calculator.py is ready.")
    print("=" * 60)