"""
main.py — PBM Preprocessing Pipeline Orchestrator
===================================================
Takes an MP4 video (up to 60 seconds), runs six computer vision analysis
modules on it, and writes a single metadata.json file to backend/output/.

Usage:
    python backend/main.py --input path/to/video.mp4

Output:
    backend/output/metadata.json
"""

import argparse
import json
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Logging setup — prints timestamped progress to the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pbm.main")


# ---------------------------------------------------------------------------
# Module imports — each file in preprocessor/ owns one detection task
# ---------------------------------------------------------------------------
# These imports will work once the individual module files are written.
# For now they are commented out so main.py can be tested in isolation.
#
from preprocessor.scene_cut       import detect_scene_cuts
from preprocessor.face_detection  import detect_faces
from preprocessor.object_detection import detect_objects
from preprocessor.complexity_score import compute_complexity
from preprocessor.optical_flow    import detect_static_moments
# from preprocessor.audio_analysis  import detect_critical_moments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_input(video_path: str) -> None:
    """
    Confirms the file exists and is a .mp4 before we do any real work.
    Raises a clear error message if something is wrong.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.lower().endswith(".mp4"):
        raise ValueError(f"Expected an .mp4 file, got: {video_path}")


def enforce_60s_cap(video_path: str) -> float:
    """
    Opens the video with OpenCV just long enough to read its duration.
    If it is longer than 60 seconds, we log a warning — the individual
    modules will each respect the 60-second cap themselves when they read
    frames, so no trimming is needed here.

    Returns the actual duration in seconds.
    """
    import cv2  # imported here so the rest of the file stays importable without cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    if duration > 60.0:
        log.warning(
            "Video is %.1fs — only the first 60 seconds will be analysed.", duration
        )
    else:
        log.info("Video duration: %.1fs (within 60s cap).", duration)

    return duration


def build_empty_metadata() -> dict:
    """
    Returns the full metadata skeleton with empty containers for every key.
    Each module fills in its own section; if a module fails, its section
    stays empty rather than crashing the whole run.
    """
    return {
        "scene_cuts":       [],          # list of floats (seconds)
        "faces":            {},          # dict keyed by "frame_NNN"
        "objects":          {},          # dict keyed by "frame_NNN"
        "complexity":       {},          # dict keyed by "frame_NNN"
        "static_moments":  [],          # list of [start_s, end_s] pairs
        "critical_moments": [],          # list of [start_s, end_s] pairs
    }


def run_module(name: str, fn, *args, target: dict, key: str) -> None:
    """
    Calls one detection function inside a try/except so a single broken
    module never kills the whole pipeline.

    Arguments:
        name   — human-readable label used in log messages
        fn     — the detection function to call
        *args  — positional arguments forwarded to fn
        target — the metadata dict we write results into
        key    — the key inside target where results are stored
    """
    log.info("Starting: %s", name)
    t0 = time.perf_counter()
    try:
        result = fn(*args)
        target[key] = result
        elapsed = time.perf_counter() - t0
        log.info("Finished: %s (%.2fs)", name, elapsed)
    except Exception as exc:  # noqa: BLE001
        log.error("FAILED: %s — %s", name, exc)
        log.error("  Continuing with empty results for this module.")


def write_metadata(metadata: dict, output_dir: str) -> str:
    """
    Serialises the metadata dict to JSON and saves it.
    Creates the output directory if it does not exist yet.
    Returns the full path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metadata.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("metadata.json written to: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(video_path: str, output_dir: str = "backend/output") -> str:
    """
    Orchestrates the full preprocessing pipeline.

    Step 1  — Validate input (file exists, is .mp4)
    Step 2  — Check video length, warn if > 60s
    Step 3  — Build the empty metadata container
    Step 4  — Run each of the six detection modules in sequence
    Step 5  — Write the combined results to metadata.json
    Step 6  — Return the output path so callers can confirm where it landed

    Each module gets the same video_path and the same 60-second cap.
    Results flow into a shared metadata dict; failures are logged, not raised.
    """

    log.info("=== PBM Preprocessing Pipeline ===")
    log.info("Input video : %s", video_path)
    log.info("Output dir  : %s", output_dir)

    # ------------------------------------------------------------------
    # Step 1: Validate
    # ------------------------------------------------------------------
    log.info("--- Step 1: Validating input ---")
    validate_input(video_path)
    log.info("Input validated.")

    # ------------------------------------------------------------------
    # Step 2: Check duration
    # ------------------------------------------------------------------
    log.info("--- Step 2: Checking video duration ---")
    duration = enforce_60s_cap(video_path)

    # The cap passed to every module — even if the video is shorter,
    # this constant keeps behaviour consistent.
    MAX_SECONDS = 60.0

    # ------------------------------------------------------------------
    # Step 3: Build empty metadata container
    # ------------------------------------------------------------------
    log.info("--- Step 3: Initialising metadata container ---")
    metadata = build_empty_metadata()

    # ------------------------------------------------------------------
    # Step 4: Run each detection module
    #
    # When the individual module files exist, uncomment each run_module()
    # call below and remove the placeholder log line next to it.
    # ------------------------------------------------------------------
    log.info("--- Step 4: Running detection modules ---")

    # 4a. Scene cut detection
    #   Finds every moment where the camera angle or shot changes abruptly.
    #   Output: a list of timestamps in seconds, e.g. [0.5, 3.2, 7.8]
    run_module(
        "Scene Cut Detection",
        detect_scene_cuts,
        video_path, MAX_SECONDS,
        target=metadata, key="scene_cuts",
    )

    # 4b. Face detection
    #   For every frame, records whether a face is visible and where it is.
    #   Output: dict of {frame_id: {x, y, w, h}} bounding boxes
    run_module(
        "Face Detection",
        detect_faces,
        video_path, MAX_SECONDS,
        target=metadata, key="faces",
    )

    # 4c. Object detection
    #   Labels named characters / objects in each frame and their positions.
    #   Output: dict of {frame_id: [{label, x, y}, ...]}
    run_module(
        "Object Detection",
        detect_objects,
        video_path, MAX_SECONDS,
        target=metadata, key="objects",
    )

    # 4d. Complexity scoring
    #   Assigns a visual-complexity number to each frame (edge count +
    #   colour variety + motion magnitude). High score = busy scene.
    #   Output: dict of {frame_id: score}
    run_module(
        "Complexity Scoring",
        compute_complexity,
        video_path, MAX_SECONDS,
        target=metadata, key="complexity",
    )

    # 4e. Optical flow / static moment detection
    #   Identifies stretches of video where almost nothing is moving —
    #   these are the windows used to measure fixation stability.
    #   Output: list of [start_s, end_s] pairs
    run_module(
        "Static Moment Detection",
        detect_static_moments,
        video_path, MAX_SECONDS,
        target=metadata, key="static_moments",
    )

    # 4f. Audio analysis / critical moment detection
    #   Finds moments of loud, emphatic speech or high dramatic tension —
    #   the windows used to measure strategic blink timing.
    #   Output: list of [start_s, end_s] pairs
    # run_module(
    #     "Critical Moment Detection",
    #     detect_critical_moments,
    #     video_path, MAX_SECONDS,
    #     target=metadata, key="critical_moments",
    # )
    log.info("  [audio_analysis]  — module not yet implemented, skipping.")

    # ------------------------------------------------------------------
    # Step 5: Persist results
    # ------------------------------------------------------------------
    log.info("--- Step 5: Writing metadata.json ---")
    out_path = write_metadata(metadata, output_dir)

    # ------------------------------------------------------------------
    # Step 6: Done
    # ------------------------------------------------------------------
    log.info("=== Pipeline complete. Output: %s ===", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PBM Preprocessing Pipeline — analyse a video and output metadata.json"
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="VIDEO.MP4",
        help="Path to the input MP4 file (max 60 seconds).",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/output",
        metavar="DIR",
        help="Directory where metadata.json will be written (default: backend/output).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        result_path = run_pipeline(
            video_path=args.input,
            output_dir=args.output_dir,
        )
        sys.exit(0)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        log.error("Pipeline could not start: %s", err)
        sys.exit(1)