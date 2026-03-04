"""
optical_flow.py
---------------
Analyses per-frame motion magnitude in a video and identifies two types
of temporal windows:

  static_moments   — low-motion stretches used by the Fixation Stability
                     biomarker. We can only measure ocular tremor when
                     the screen itself is still; otherwise we can't
                     separate eye jitter from the child tracking motion.

  critical_moments — high-motion stretches used by the Strategic Blink
                     Rate biomarker to classify blinks as occurring at
                     safe vs. critical moments.

Both outputs are lists of [start_seconds, end_seconds] pairs.

The motion signal is computed with cv2.calcOpticalFlowFarneback(), the
same algorithm used in complexity_score.py. Here we use the raw mean
magnitude value (not the scaled version) so the thresholds (1.0 and 8.0
px/frame) are meaningful.
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — tunable without changing logic
# ---------------------------------------------------------------------------

# A frame is "static" if mean flow magnitude is below this value (px/frame)
STATIC_MAGNITUDE_THRESHOLD = 1.0

# A frame is "critical" if mean flow magnitude is above this value (px/frame)
CRITICAL_MAGNITUDE_THRESHOLD = 8.0

# Minimum contiguous run to count as a static moment (seconds)
STATIC_MIN_DURATION_SEC = 1.0

# Minimum contiguous run to count as a critical moment (seconds)
CRITICAL_MIN_DURATION_SEC = 0.5

# Adjacent windows closer than this are merged into one (seconds)
MERGE_GAP_SEC = 0.5


def detect_motion_windows(video_path: str, max_seconds: float) -> dict:
    """
    Returns a dict with two keys: static_moments and critical_moments.
    Each is a list of [start_seconds, end_seconds] pairs.

    Example:
    {
      "static_moments":   [[10.2, 12.4], [45.1, 46.8]],
      "critical_moments": [[5.0, 6.2],  [28.3, 30.1]]
    }

    Only processes the first max_seconds of the video.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return {"static_moments": [], "critical_moments": []}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        log.warning("Could not read FPS from video; defaulting to 24.")
        fps = 24.0

    max_frames = int(max_seconds * fps)
    log.info(f"optical_flow: fps={fps:.2f}, max_frames={max_frames}")

    # Minimum run lengths in frames (derived from seconds + fps)
    static_min_frames   = max(1, int(STATIC_MIN_DURATION_SEC  * fps))
    critical_min_frames = max(1, int(CRITICAL_MIN_DURATION_SEC * fps))
    merge_gap_frames    = max(1, int(MERGE_GAP_SEC * fps))

    # -----------------------------------------------------------------------
    # Step 1 — Read all frames and compute per-frame mean flow magnitude
    # -----------------------------------------------------------------------
    # We store mean_magnitude for every frame after the first.
    # Frame 0 has no previous frame, so it gets magnitude 0.0.

    magnitudes = []   # index = frame_number, value = mean magnitude (float)

    ret, prev_frame = cap.read()
    if not ret:
        log.warning("Could not read first frame.")
        cap.release()
        return {"static_moments": [], "critical_moments": []}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    magnitudes.append(0.0)  # Frame 0 — no prior frame to compare

    frame_num = 1
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # End of video before max_seconds — that's fine

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback dense optical flow — same params as complexity_score.py
        # Returns a 2-channel array of (dx, dy) per pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,   # image pyramid scale
            levels=3,        # pyramid levels
            winsize=15,      # averaging window size
            iterations=3,    # iterations per level
            poly_n=5,        # neighbourhood size for poly expansion
            poly_sigma=1.2,  # std dev for Gaussian used in poly expansion
            flags=0
        )

        # Magnitude of the (dx, dy) vector at each pixel
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag  = float(np.mean(magnitude))

        magnitudes.append(mean_mag)

        prev_gray = gray
        frame_num += 1

    cap.release()
    total_frames = len(magnitudes)
    log.info(f"optical_flow: computed magnitudes for {total_frames} frames")

    if total_frames == 0:
        return {"static_moments": [], "critical_moments": []}

    # -----------------------------------------------------------------------
    # Step 2 — Classify each frame as static, critical, or neither
    # -----------------------------------------------------------------------

    is_static   = [m < STATIC_MAGNITUDE_THRESHOLD   for m in magnitudes]
    is_critical = [m > CRITICAL_MAGNITUDE_THRESHOLD for m in magnitudes]

    # -----------------------------------------------------------------------
    # Step 3 — Extract contiguous runs that meet minimum duration
    # -----------------------------------------------------------------------

    def extract_runs(flags, min_run_frames):
        """
        Given a boolean list, find all contiguous True runs that are at
        least min_run_frames long. Return as list of [start_sec, end_sec].
        """
        runs = []
        in_run = False
        run_start = 0

        for i, flag in enumerate(flags):
            if flag and not in_run:
                in_run = True
                run_start = i
            elif not flag and in_run:
                in_run = False
                run_len = i - run_start
                if run_len >= min_run_frames:
                    start_sec = round(run_start / fps, 3)
                    end_sec   = round((i - 1) / fps, 3)
                    runs.append([start_sec, end_sec])

        # Handle run that extends to the last frame
        if in_run:
            run_len = len(flags) - run_start
            if run_len >= min_run_frames:
                start_sec = round(run_start / fps, 3)
                end_sec   = round((len(flags) - 1) / fps, 3)
                runs.append([start_sec, end_sec])

        return runs

    static_runs   = extract_runs(is_static,   static_min_frames)
    critical_runs = extract_runs(is_critical, critical_min_frames)

    # -----------------------------------------------------------------------
    # Step 4 — Merge adjacent windows separated by less than MERGE_GAP_SEC
    # -----------------------------------------------------------------------

    def merge_adjacent(runs, gap_sec):
        """
        Merge windows whose gap is smaller than gap_sec.
        Input is sorted by start time (extract_runs guarantees this).
        """
        if not runs:
            return []

        merged = [runs[0][:]]  # copy first window

        for start, end in runs[1:]:
            prev_end = merged[-1][1]
            if start - prev_end < gap_sec:
                # Extend the previous window to cover this one
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return merged

    merge_gap_sec = merge_gap_frames / fps

    static_moments   = merge_adjacent(static_runs,   merge_gap_sec)
    critical_moments = merge_adjacent(critical_runs, merge_gap_sec)

    log.info(f"optical_flow: {len(static_moments)} static moment(s) detected")
    log.info(f"optical_flow: {len(critical_moments)} critical moment(s) detected")

    if static_moments:
        log.info(f"  static sample (first 3): {static_moments[:3]}")
    if critical_moments:
        log.info(f"  critical sample (first 3): {critical_moments[:3]}")

    return {
        "static_moments":   static_moments,
        "critical_moments": critical_moments,
    }