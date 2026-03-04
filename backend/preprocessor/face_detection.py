"""
face_detection.py

Preprocessing module: Face Detection
-------------------------------------
Reads an MP4 video frame by frame and detects faces using the MediaPipe Tasks API
(FaceDetector). For every frame where at least one face is found, records the bounding
box in pixel coordinates. Output is used downstream to measure Social Gaze Saliency
(SGS) — whether the child's gaze fell inside a character's face region during viewing.

Requires:
    blaze_face_short_range.tflite must exist at:
    backend/preprocessor/blaze_face_short_range.tflite

Output format:
    {
        "frame_500": [{"x": 320, "y": 180, "w": 240, "h": 280}, {"x": 420, "y": 55, "w": 115, "h": 115}],
        "frame_501": [{"x": 318, "y": 182, "w": 242, "h": 276}],
        ...
    }

All detected faces per frame are stored as a list. Frames with no faces are omitted.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Path to the .tflite model file, relative to this file's location
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def detect_faces(video_path: str, max_seconds: float) -> dict:
    """
    Returns a dict of frame-keyed bounding boxes where faces were detected.
    Example: {"frame_500": [{"x": 320, "y": 180, "w": 240, "h": 280}, {"x": 420, "y": 55, "w": 115, "h": 115}]}
    Only processes the first max_seconds of the video.
    """

    # --- Validate model file exists ---
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Face detector model not found at: {_MODEL_PATH}\n"
            "Run the wget command in the module docstring to download it."
        )

    # --- Initialise MediaPipe Tasks FaceDetector ---
    # BaseOptions points to the .tflite model on disk.
    # min_detection_confidence=0.5 is a reasonable default — tune higher
    # (e.g. 0.7) if animated faces produce too many false positives.
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    # --- Open the video file ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback if metadata is missing

    max_frame = int(fps * max_seconds)

    # --- Result container ---
    faces = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number >= max_frame:
            break

        # OpenCV loads as BGR; MediaPipe Tasks expects RGB.
        # We wrap the numpy array in mp.Image using the SRGB format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect(mp_image)

        if result.detections:
            frame_height, frame_width = frame.shape[:2]
            frame_faces = []

            # Loop over all detected faces in this frame — not just the first.
            # In multi-character scenes, this captures every visible face so the
            # real-time SGS check can test whether gaze fell inside any of them.
            for detection in result.detections:
                bbox = detection.bounding_box
                x_px = bbox.origin_x
                y_px = bbox.origin_y
                w_px = bbox.width
                h_px = bbox.height

                # Clamp to frame boundaries
                x_px = max(0, x_px)
                y_px = max(0, y_px)
                w_px = min(w_px, frame_width - x_px)
                h_px = min(h_px, frame_height - y_px)

                frame_faces.append({"x": x_px, "y": y_px, "w": w_px, "h": h_px})

            key = f"frame_{frame_number}"
            faces[key] = frame_faces

        frame_number += 1

    # --- Clean up ---
    cap.release()
    detector.close()

    return faces