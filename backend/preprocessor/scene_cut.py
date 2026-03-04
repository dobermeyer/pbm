import cv2
import numpy as np


def detect_scene_cuts(video_path: str, max_seconds: float) -> list[float]:
    """
    Returns a list of timestamps (in seconds) where scene cuts occur.
    Example: [0.5, 3.2, 7.8, 14.1, ...]
    Only processes the first max_seconds of the video.

    Detection method:
    - Convert each frame to grayscale
    - Compute absolute pixel-level difference between consecutive frames
    - If >40% of pixels differ by more than a noise threshold, flag as a scene cut
    """

    # --- Open the video file ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # --- Get basic video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) reported for: {video_path}")

    # Maximum frame index we'll process (convert seconds -> frames)
    max_frames = int(max_seconds * fps)

    # --- Detection parameters ---
    # A pixel is considered "changed" if its grayscale value shifts by more than
    # this amount. Helps ignore minor lighting flicker, compression noise, etc.
    PIXEL_DIFF_THRESHOLD = 25   # out of 255

    # If the fraction of changed pixels exceeds this, it's a scene cut.
    # Matches the spec: >40% of pixels changing = cut.
    CUT_FRACTION_THRESHOLD = 0.40

    # --- State variables ---
    scene_cuts = []          # timestamps (seconds) of detected cuts
    prev_gray = None         # previous frame as grayscale numpy array
    frame_index = 0          # current frame counter

    while True:
        ret, frame = cap.read()

        # End of video or read failure
        if not ret:
            break

        # Stop once we've hit the max duration
        if frame_index >= max_frames:
            break

        # Convert current frame to grayscale (we don't need colour for cut detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Absolute per-pixel difference between this frame and the previous one
            diff = cv2.absdiff(gray, prev_gray)

            # Count pixels that changed more than the noise threshold
            changed_pixels = np.sum(diff > PIXEL_DIFF_THRESHOLD)
            total_pixels = gray.size   # height * width

            changed_fraction = changed_pixels / total_pixels

            if changed_fraction > CUT_FRACTION_THRESHOLD:
                # Convert frame index to timestamp in seconds
                timestamp = frame_index / fps
                scene_cuts.append(round(timestamp, 3))

        # This frame becomes the "previous" frame for the next iteration
        prev_gray = gray
        frame_index += 1

    cap.release()

    return scene_cuts