"""
validate_subpixel.py
PBM Project — Sub-Pixel Precision Validation

Tests whether MediaPipe iris landmarks, with center-of-mass refinement applied,
produce centroids stable to sub-pixel resolution (<0.5px drift) during fixation.

Uses MediaPipe Tasks API (0.10.x) — mp.solutions is not available in this version.

Usage:
    python validate_subpixel.py --video Testvid1hor.MOV

Pass criterion:
    Median consecutive-frame centroid drift during fixation windows < 0.5px
    At least 5 fixation windows detected
"""

import argparse
import sys
import urllib.request
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIXATION_VELOCITY_THRESHOLD_PX = 2.0
MIN_FIXATION_WINDOW_FRAMES     = 30
SUBPIXEL_PASS_THRESHOLD_PX     = 0.5
MIN_WINDOWS_REQUIRED           = 5

# Left iris landmarks: 468 = centre, 469-472 = perimeter points
IRIS_LANDMARK_INDICES = [468, 469, 470, 471, 472]

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}")
    else:
        print(f"Model found: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Center-of-mass refinement (Amendment 3 method)
# ---------------------------------------------------------------------------

def refine_centroid(landmarks, frame_width, frame_height):
    """
    Weighted center-of-mass across the 5 iris landmark points.
    Coordinates stay as floats throughout — never rounded.
    Returns (centroid_x, centroid_y) in pixels, or None if confidence is low.
    """
    points  = []
    weights = []

    for idx in IRIS_LANDMARK_INDICES:
        lm     = landmarks[idx]
        px     = lm.x * frame_width
        py     = lm.y * frame_height
        weight = getattr(lm, 'presence', None)
        if weight is None:
            weight = 1.0
        if weight < 0.1:
            return None
        points.append((px, py))
        weights.append(weight)

    points  = np.array(points,  dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    return float(np.sum(points[:, 0] * weights)), float(np.sum(points[:, 1] * weights))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(video_path):
    print(f"\nPBM Sub-Pixel Precision Validation")
    print(f"Video: {video_path}")
    print("-" * 50)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resolution : {frame_width} x {frame_height}")
    print(f"FPS        : {fps:.1f}")
    print(f"Frames     : {total_frames} (~{total_frames/fps:.1f}s)")
    print("-" * 50)

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # --- Pass 1: Extract centroids ---
    print("\nPass 1: Extracting iris centroids...")

    centroids     = []
    frame_idx     = 0
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int((frame_idx / fps) * 1000)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        centroid = None
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            if len(lms) > max(IRIS_LANDMARK_INDICES):
                centroid = refine_centroid(lms, frame_width, frame_height)

        if centroid:
            centroids.append((frame_idx, centroid[0], centroid[1]))
        else:
            centroids.append(None)
            failed_frames += 1

        frame_idx += 1

    cap.release()
    landmarker.close()

    valid_count = sum(1 for c in centroids if c is not None)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Valid centroids  : {valid_count} ({100*valid_count/max(frame_idx,1):.1f}%)")
    print(f"  Failed frames    : {failed_frames}")

    if valid_count < 30:
        print("\nERROR: Too few valid frames. Ensure face is clearly visible.")
        sys.exit(1)

    # --- Pass 2: Detect fixation windows ---
    print("\nPass 2: Detecting fixation windows...")

    fixation_windows = []
    current_window   = []

    for i in range(1, len(centroids)):
        prev = centroids[i - 1]
        curr = centroids[i]

        if prev is None or curr is None:
            if len(current_window) >= MIN_FIXATION_WINDOW_FRAMES:
                fixation_windows.append(current_window)
            current_window = []
            continue

        dist = np.sqrt((curr[1]-prev[1])**2 + (curr[2]-prev[2])**2)

        if dist < FIXATION_VELOCITY_THRESHOLD_PX:
            if not current_window:
                current_window.append(prev)
            current_window.append(curr)
        else:
            if len(current_window) >= MIN_FIXATION_WINDOW_FRAMES:
                fixation_windows.append(current_window)
            current_window = []

    if len(current_window) >= MIN_FIXATION_WINDOW_FRAMES:
        fixation_windows.append(current_window)

    print(f"  Fixation windows found: {len(fixation_windows)}")

    if len(fixation_windows) < MIN_WINDOWS_REQUIRED:
        print(f"  WARNING: fewer than {MIN_WINDOWS_REQUIRED} windows — result may not be reliable")
        if len(fixation_windows) == 0:
            print("  No fixation windows detected at all. Check footage.")
            sys.exit(1)

    # --- Pass 3: Measure drift ---
    print("\nPass 3: Measuring centroid drift within fixation windows...")
    print()
    print(f"  {'Window':>6}  {'Frames':>6}  {'Median drift':>13}  {'Max drift':>10}  {'Result':>6}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*13}  {'-'*10}  {'-'*6}")

    all_median_drifts = []

    for w_idx, window in enumerate(fixation_windows):
        drifts = [
            np.sqrt((window[i][1]-window[i-1][1])**2 + (window[i][2]-window[i-1][2])**2)
            for i in range(1, len(window))
        ]
        if not drifts:
            continue

        med  = np.median(drifts)
        mx   = np.max(drifts)
        all_median_drifts.append(med)
        res  = "PASS" if med < SUBPIXEL_PASS_THRESHOLD_PX else "FAIL"

        print(f"  {w_idx+1:>6}  {len(window):>6}  {med:>12.4f}px  {mx:>9.4f}px  {res:>6}  (frame {window[0][0]})")

    # --- Verdict ---
    print()
    print("=" * 50)

    if not all_median_drifts:
        print("RESULT: INCONCLUSIVE")
        sys.exit(1)

    overall   = np.median(all_median_drifts)
    n_pass    = sum(1 for d in all_median_drifts if d < SUBPIXEL_PASS_THRESHOLD_PX)
    n_total   = len(all_median_drifts)
    pass_rate = 100 * n_pass / n_total

    print(f"Overall median drift : {overall:.4f}px")
    print(f"Pass threshold       : {SUBPIXEL_PASS_THRESHOLD_PX}px")
    print(f"Windows passed       : {n_pass}/{n_total} ({pass_rate:.0f}%)")
    print()

    if overall < SUBPIXEL_PASS_THRESHOLD_PX and pass_rate >= 80:
        print("RESULT: PASS")
        print("Sub-pixel precision confirmed. Proceed to build SPV biomarker.")
    else:
        print("RESULT: FAIL")
        print("Sub-pixel precision not confirmed.")
        print("Center-of-mass refinement needs adjustment before SPV is built.")

    print("=" * 50)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    run_validation(args.video)