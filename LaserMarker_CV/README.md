# LaserMarker_CV — Computer Vision Subsystem

Python-based CV pipeline for detecting EP tube cap positions on an orange rack and converting them to physical millimeter coordinates for the laser gantry.

This is **Stage 1** of the project — the algorithm is validated here on PC before being ported to the iOS app (CoreML / on-device YOLO).

---

## What This Does

1. Takes a photo of a tube rack (phone camera or ESP32 stream)
2. Detects all tube cap centers (YOLO or classical OpenCV)
3. Corrects for camera perspective via homography (4 fiducial corner points)
4. Maps pixel positions → physical mm coordinates
5. Outputs `(x_mm, y_mm)` targets for the laser gantry

---

## File Overview

### Training

| File | Purpose |
|------|---------|
| `train_gpu.py` | Train YOLOv8n on GPU (100 epochs, batch 16) |
| `train_local.py` | Train on CPU (batch 4, slower) |
| `data.yaml` | Dataset config — single class `tube` |
| `fix_label.py` | Normalize all label files to class `0` |
| `check_result.py` | Quick inference test on a saved image |

### Calibration & Core

| File | Purpose |
|------|---------|
| `Homography_Calibration.py` | Click 4 rack corners → compute homography → show mm grid overlay |
| `laser_vision_core.py` | `get_physical_coords(px, py)` — pixel to mm conversion |

### Cloud / Backup

| File | Purpose |
|------|---------|
| `ai_detect.py` | Roboflow workflow inference (prototyping fallback) |

---

## OpenCV Detection Scripts

Parallel classical approaches. Useful for parameter tuning and understanding the image properties before committing to YOLO.

| Script | Method |
|--------|--------|
| `detect_ep_tubes_contour.py` | Contour area + solidity filter |
| `detect_ep_tubes_improved.py` | Contour + min/max circularity range |
| `detect_ep_tubes_watershed.py` | Watershed for touching/overlapping tubes |
| `detect_tubes_on_board.py` | Detect rack outline first, then Hough inside ROI |
| `detect_tubes_by_hole_map.py` | Grid of rack holes → per-hole occupancy check |
| `inner_circle_detect.py` | Loose Hough + blue-channel color filter |
| `find_tubes_Gemini.py` | Blue-channel threshold + blob stats (debug/analysis) |
| `analyze_tube.py` | Pixel statistics on a single tube image |
| `analyze_tube_shape.py` | Circularity, ellipse fit, solidity of one tube |
| `analyze_multi_tubes.py` | Suggest contour thresholds from a multi-tube image |
| `mark_tubes.py` | Click to label tubes manually, auto-suggest Hough params |

> **Key insight:** The orange rack has high R−B pixel values; white/pink EP tube caps have high blue-channel values and low R−B difference. The blue channel alone is highly discriminating for separating tubes from empty holes.

---

## Perspective Correction

A core challenge: the phone will never be held perfectly overhead. Even small tilts make pixel coordinates unreliable.

**Solution (Stage 4 of the full plan):**
- Place 4 colored fiducial markers (dots or mini QR codes) on the rack corners
- App detects these markers and calls `getPerspectiveTransform`
- The resulting homography "flattens" the image regardless of phone angle
- Physical tube coordinates become accurate to within 1–2 mm

`Homography_Calibration.py` implements the PC version of this:

```bash
python Homography_Calibration.py
# Click the 4 rack corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left
# Grid overlay appears showing the mm coordinate system
```

---

## Training Results

| Run | Epochs | mAP50 | Precision | Recall | Notes |
|-----|--------|-------|-----------|--------|-------|
| train1–2 | — | — | — | — | Config snapshots only |
| train3–6 | 100 | improving | — | — | Incremental dataset growth |
| **train7** | 100 | **~0.82** | **~1.0** | **~0.61** | Best so far, GPU |

Low recall (0.61) means some tubes are missed — primarily due to small dataset size (~12 labeled images). Target for production: 200+ images across varied lighting and angles.

---

## Setup

```bash
pip install ultralytics opencv-python numpy requests inference-sdk
```

### Train

```bash
python train_gpu.py     # requires CUDA GPU
# or
python train_local.py   # CPU only (slow)
```

### Calibrate

```bash
python Homography_Calibration.py
```

### Inference

```bash
python check_result.py
# Edit the image path inside the script first
```

---

## Dataset Structure

```
dataset/
├── train/
│   ├── images/     # not in repo — add your own images
│   └── labels/     # YOLO format: class cx cy w h (normalized)
└── val/
    ├── images/     # not in repo
    └── labels/
```

All labels use class `0` (tube). Run `fix_label.py` after labeling if your tool assigns different class IDs.

---

## Mobile Port Plan

Once the PC algorithm is solid, the detection will be ported to the Flutter app:

- Export the trained YOLO model to **TFLite** (Android) and **CoreML** (iOS)
- Run on-device inference — no server needed, works offline
- Perspective correction (using 4 rack corner markers) runs on the captured photo before inference
- Output: list of `(x_mm, y_mm)` coordinates sent to ESP32 over BLE

---

## Known Issues

| File | Issue |
|------|-------|
| `laser_vision_core.py` | Missing `import cv2`; calibration corner pixels are placeholders |
| `detect_ep_tubes.py` | Corrupted first line — won't run as-is |
| `ai_detect.py` | Hardcoded Roboflow API key — move to `.env` before committing |
| `dataset/*/images/` | Not in repo (too large) — scripts expect images beside them |
