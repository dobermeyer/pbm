# backend/preprocessor/object_detection.py
#
# Detects objects in each frame of a video using YOLOv8.
# Output feeds Gaze Entropy and Info Intake Rate biomarkers.
#
# Output schema:
# {
#   "frame_800": [
#     {"label": "person", "x": 200, "y": 150, "w": 80, "h": 120},
#     {"label": "sports ball", "x": 410, "y": 300, "w": 45, "h": 45}
#   ],
#   ...
# }
# Frames with no detections are omitted.

import logging
import cv2
from ultralytics import YOLO

log = logging.getLogger(__name__)


def detect_objects(video_path: str, max_seconds: float) -> dict:
    """
    Returns a dict of frame-keyed object lists.
    Example: {"frame_800": [{"label": "person", "x": 200, "y": 150, "w": 80, "h": 120}]}
    Only processes the first max_seconds of the video.
    """

    # --- Load YOLOv8 nano model ---
    # 'yolov8n.pt' is the smallest/fastest variant — good for frame-by-frame
    # video processing where speed matters. ultralytics will auto-download it
    # on first run and cache it locally.
    log.info("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # --- Open the video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Get frames per second so we can convert frame numbers to timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # safe fallback if metadata is missing

    # Calculate the maximum frame index we should process
    max_frame = int(max_seconds * fps)

    log.info(f"Video FPS: {fps:.2f} | Processing up to frame {max_frame} ({max_seconds}s)")

    results_dict = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()

        # End of video or unreadable frame
        if not ret:
            break

        # Stop once we've passed the time cap
        if frame_number >= max_frame:
            break

        # --- Run YOLOv8 inference on this frame ---
        # verbose=False suppresses per-frame console output (very noisy otherwise)
        # conf=0.4 means we only keep detections with >= 40% confidence,
        # filtering out low-quality / ambiguous detections
        detections = model(frame, verbose=False, conf=0.4)

        # detections is a list with one Results object per image (we only sent one)
        result = detections[0]

        # result.boxes contains all bounding boxes for this frame
        # If nothing was detected, boxes will be empty — we skip the frame
        if result.boxes is None or len(result.boxes) == 0:
            frame_number += 1
            continue

        frame_objects = []

        for box in result.boxes:
            # box.xyxy gives [x1, y1, x2, y2] in pixel coordinates
            # We convert to {x, y, w, h} to match the face_detection.py schema
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            # box.cls gives the class index as a tensor — convert to int, then look up name
            class_index = int(box.cls[0].item())
            label = model.names[class_index]  # e.g. "person", "chair", "dog"

            frame_objects.append({
                "label": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

        # Only store this frame if we actually found something
        if frame_objects:
            frame_key = f"frame_{frame_number}"
            results_dict[frame_key] = frame_objects

        frame_number += 1

    cap.release()

    log.info(f"Object detection complete. {len(results_dict)} frames with detections "
             f"out of {frame_number} frames processed.")

# ---------------------------------------------------------------------------
# DOWNSTREAM CONSUMER NOTE — read before using "objects" data
# ---------------------------------------------------------------------------
# Multiple objects in the same frame may share the same label (e.g. two
# "person" entries). Consumers must NOT group or deduplicate by label alone.
#
# Each list entry is a spatially distinct object and must be treated as
# a unique entity. Use bounding box position to distinguish them:
#
#   "frame_250": [
#     {"label": "person", "x": 10,  "y": 20, "w": 80, "h": 120},  ← person_A
#     {"label": "person", "x": 200, "y": 20, "w": 80, "h": 120},  ← person_B
#   ]
#
# For Gaze Entropy: assign each object a spatial ID (e.g. index in list,
# or centroid coordinates) when building the gaze sequence. Two "person"
# entries = two separate fixation targets.
#
# For Info Intake Rate: count list entries, not unique labels. The above
# example = 2 fixation targets, not 1.
# ---------------------------------------------------------------------------
    return results_dict