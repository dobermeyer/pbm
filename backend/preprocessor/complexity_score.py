"""
complexity_score.py
-------------------
Assigns a single numeric "visual complexity" score to every frame of a video.

Score = edge_count + colour_variety + motion_magnitude

Used downstream by the Pupillary Efficiency (PE) biomarker:
the real-time eye-tracking layer correlates per-frame pupil diameter
against this score to detect cognitive load changes.

Unlike other modules (faces, objects), ALL frames are stored — even
zero-complexity frames are meaningful data points that anchor the
baseline for the pupil dilation calculation.

Function signature expected by main.py:
    compute_complexity(video_path: str, max_seconds: float) -> dict
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


def compute_complexity(video_path: str, max_seconds: float) -> dict:
    """
    Returns a dict of frame-keyed complexity scores.
    Example: {"frame_0": 1200, "frame_1": 1350, "frame_100": 2400}

    Every frame up to max_seconds is stored (including zero-score frames).
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # safe fallback if metadata is missing
    max_frames = int(fps * max_seconds)

    log.info(f"[complexity_score] fps={fps:.1f}, max_frames={max_frames}")

    complexity_scores = {}  # output: {"frame_NNN": score}

    frame_number = 0
    prev_gray = None  # previous frame in grayscale, needed for optical flow

    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            # End of video — stop cleanly
            break

        # ----------------------------------------------------------------
        # Component 1: Edge Count (Canny edge detection)
        #
        # Convert to grayscale, run Canny, count non-zero (white) pixels.
        # More edges = more visual detail = higher complexity.
        # Thresholds 100/200 are standard starting values for natural images;
        # they work well on animated content too.
        # ----------------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_count = int(np.count_nonzero(edges))

        # ----------------------------------------------------------------
        # Component 2: Colour Variety (HSV hue bucketing)
        #
        # Convert to HSV and look only at the Hue channel (0-179 in OpenCV).
        # Quantise into 16 equal bins. Count how many bins have at least one
        # pixel in them. Result is a number from 0 to 16.
        #
        # Multiply by a scaling factor (200) so the range is roughly
        # comparable in magnitude to the edge count, avoiding one component
        # from dominating the sum on most natural frames.
        # ----------------------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_channel = hsv[:, :, 0]  # hue values 0-179
        num_bins = 16
        bin_size = 180 // num_bins  # = 11 (close enough for 0-179 range)
        occupied_bins = 0
        for b in range(num_bins):
            lo = b * bin_size
            hi = lo + bin_size
            if np.any((hue_channel >= lo) & (hue_channel < hi)):
                occupied_bins += 1
        colour_variety = occupied_bins * 200  # scale to ~0-3200

        # ----------------------------------------------------------------
        # Component 3: Motion Magnitude (dense optical flow)
        #
        # Compare this frame to the previous one using Farneback optical flow.
        # Each pixel gets a flow vector (dx, dy). We take the mean magnitude
        # across all pixels. Multiplied by a scaling factor (100) to bring
        # its range in line with the other two components.
        #
        # For the very first frame there is no previous frame, so motion = 0.
        # ----------------------------------------------------------------
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,           # output flow array (None = allocate fresh)
                pyr_scale=0.5,  # pyramid scale between levels
                levels=3,       # number of pyramid levels
                winsize=15,     # averaging window size
                iterations=3,   # iterations per pyramid level
                poly_n=5,       # pixel neighbourhood size for poly expansion
                poly_sigma=1.2, # std dev of Gaussian for poly expansion
                flags=0
            )
            # flow shape: (H, W, 2) — channel 0 = dx, channel 1 = dy
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_magnitude = float(np.mean(magnitude)) * 100  # scale
        else:
            motion_magnitude = 0.0

        # ----------------------------------------------------------------
        # Final Score
        #
        # Simple unweighted sum. Each component contributes roughly equally
        # for typical animated content. Can be rebalanced later with
        # per-component weights if empirical calibration suggests it.
        # ----------------------------------------------------------------
        score = edge_count + colour_variety + motion_magnitude

        key = f"frame_{frame_number}"
        complexity_scores[key] = round(score, 2)

        # Keep this frame as "previous" for the next iteration's flow calc
        prev_gray = gray
        frame_number += 1

    cap.release()

    log.info(
        f"[complexity_score] Scored {len(complexity_scores)} frames. "
        f"Min={min(complexity_scores.values(), default=0):.0f}, "
        f"Max={max(complexity_scores.values(), default=0):.0f}"
    )

    return complexity_scores