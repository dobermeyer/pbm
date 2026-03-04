## DEPENDENCIES.md

### Python Packages

| Package | What It Does |
|--------|-------------|
| **opencv-python** | Core computer vision library. Used for scene cut detection, optical flow analysis, and complexity scoring via frame-by-frame image processing |
| **mediapipe** | Google's framework for face detection and landmark tracking. Used to detect faces and generate bounding boxes per frame |
| **ultralytics** | Hosts the YOLOv8 object detection model. Used to identify and locate objects within each video frame |
| **librosa** | Audio analysis library. Used to measure amplitude levels and identify critical dialog moments in the video's audio track |
| **numpy** | Fundamental math and array processing library. Used throughout the pipeline for numerical calculations |

### System Libraries

| Library | What It Does |
|--------|-------------|
| **libgl1** | Low-level graphics library required by OpenCV to process video frames. Installed at the system level via `apt-get` |

### Development Environment

| Tool | What It Does |
|------|-------------|
| **GitHub Codespaces** | Cloud-based development environment. Hosts and runs the entire pipeline in the browser with no local setup required |

Got it! Here's the entry to paste into `DEPENDENCIES.md`:

---

**blaze_face_short_range.tflite** — MediaPipe face detector model (not included in repo)

This file must be downloaded manually before running the preprocessing pipeline. From the repo root:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite \
     -O backend/preprocessor/blaze_face_short_range.tflite
```

Expected location: `backend/preprocessor/blaze_face_short_range.tflite`
Expected size: ~225KB
Note: `*.tflite` is in `.gitignore` — this file will never be committed to the repo.

---

Paste that wherever makes sense in your existing `DEPENDENCIES.md` structure.