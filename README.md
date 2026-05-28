# LaserMarker — Automated EP Tube Laser Engraver

> **Status: Prototype / Work in Progress**  
> A working gantry prototype exists. CV pipeline is validated on PC. The mobile app and BLE integration are the next major milestone.

![Prototype — XY gantry over orange EP tube rack](EP%20tube%20Marker%20Prototype.jpg)

---

## The Problem

In biology and chemistry labs, EP (microcentrifuge) tubes need to be individually labeled — by hand, with markers or stickers. For large batches this is slow, error-prone, and inconsistent. This project automates it.

---

## The Solution

1. **Aim your phone** at a rack of EP tubes and take a photo
2. **On-device AI (YOLO)** detects every tube cap and corrects for camera angle
3. **Tap a cap** on screen, type what to engrave
4. **Coordinates upload to the ESP32** via Bluetooth (BLE)
5. **The XY gantry moves** to each tube in sequence
6. **Laser fires** — permanently marked, no stickers, no sharpies

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Flutter App  (Android + iOS)                │
│                                                         │
│  📷  Camera → photograph rack                           │
│  🔲  Perspective correction (4 fiducial markers)        │
│  🤖  YOLO on-device → tube cap centers in mm            │
│  👆  User taps cap → enters label text                  │
│  📡  BLE → send X Y coordinates                         │
└───────────────────────────┬─────────────────────────────┘
                            │  "X1600 Y3200\n"  (BLE GATT)
                            ▼
┌─────────────────────────────────────────────────────────┐
│               XIAO ESP32-S3                             │
│                                                         │
│  📥  BLE GATT Server → receive coordinate list          │
│  ⚙️   FastAccelStepper → X/Y gantry (A4988 drivers)     │
│  🔴  Laser PWM (GPIO 44, LEDC, 12V module)              │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    X-axis stepper (42mm)       Y-axis stepper (42mm)
    200mm travel gantry         300mm travel gantry
```

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Controller | Seeed Studio XIAO ESP32-S3 |
| Gantry | XY synchronous-belt cross slide (龙门同步带十字滑台) |
| X travel | 200 mm stroke (140 mm usable) |
| Y travel | 300 mm stroke (240 mm usable) |
| Stepper motors | 2× NEMA17 42mm, 3.3 V, 1.5 A |
| Motor drivers | 2× A4988 on dual-axis expansion board |
| Power supply | 12 V DC, ≥5 A (≥60 W switching PSU) |
| Laser module | PWM-controlled diode laser, 12 V logic via GPIO 44 |
| Limit switches | Planned (currently open-loop with manual homing) |
| Tube rack | Standard 8×12 EP tube rack (orange, ~9–10 mm pitch) |
| Phone app | Flutter (Android + iOS) |

---

## Repository Structure

```
LaserMarker/
├── LaserMarker_CV/              # PC-side computer vision (Python)
│   ├── OpenCV/                  # Classical detection experiments
│   ├── dataset/                 # YOLO training labels
│   ├── runs/                    # Training results (train1–train7)
│   ├── laser_vision_core.py     # Pixel → mm homography
│   ├── Homography_Calibration.py
│   ├── train_gpu.py / train_local.py
│   ├── ai_detect.py             # Roboflow cloud API (backup)
│   └── data.yaml                # YOLOv8 dataset config
│
└── LaserMarker_for_EP_tube/     # ESP32 firmware + data collection
    └── EP_Tube_CV/
        ├── src/main.cpp         # Active firmware: serial X/Y gantry
        ├── src/cv_main          # Draft: camera stream + gantry (WIP)
        ├── laser_test.cpp       # Laser PWM test snippet
        ├── collect_data.py      # Capture training images from ESP32 stream
        ├── collect_force.py     # Robust MJPEG frame capture
        └── platformio.ini       # PlatformIO build config
```

---

## Development Roadmap

### Stage 1 — PC Algorithm Validation ✅ (current)
Get the math right on a PC first — much easier to debug than inside an app.

- [x] OpenCV tube detection (HoughCircles, contours, watershed)
- [x] YOLOv8 training pipeline (7 runs, best mAP50 ≈ 0.82)
- [x] Homography calibration: pixel coords → physical mm
- [x] ESP32 serial gantry control (`X… Y…` over USB)
- [ ] End-to-end PC test: click tube in image → gantry moves to it

### Stage 2 — Physical Calibration
Before building the app, nail the coordinate accuracy. Every other stage depends on this.

- [ ] Place 4 fiducial markers (colored dots or mini QR codes) on rack corners
- [ ] Implement `getPerspectiveTransform` on the captured image
- [ ] Validate: physical coordinates stay accurate across different camera angles and heights
- [ ] This is the gap between a garage demo and a reliable tool

### Stage 3 — Flutter App
With calibration proven, build the frontend.

- [ ] Flutter app (Android + iOS) with BLE support
- [ ] Camera viewfinder → prompt to photograph rack
- [ ] On-device YOLO inference → box all tube caps instantly
- [ ] Perspective correction using the 4 corner markers
- [ ] User taps cap → enters label text
- [ ] Compute physical (mm) coordinates → send via BLE

### Stage 4 — BLE Communication
Wire the app to the hardware.

- [ ] Rewrite ESP32 firmware: serial → BLE GATT server
- [ ] ESP32 advertises a writable characteristic
- [ ] Flutter app writes `X1600 Y3200\n` strings over BLE
- [ ] Laser fires at each confirmed position

---

## What Works Today

| Component | Status |
|-----------|--------|
| XY gantry (mechanical) | ✅ Assembled and moving |
| ESP32 serial firmware | ✅ Accepts `X… Y…` commands |
| Laser PWM test | ✅ GPIO 44, 12V, LEDC |
| OpenCV tube detection | ✅ Multiple algorithms explored |
| YOLOv8 training pipeline | ✅ 7 runs, mAP50 ≈ 0.82 |
| Homography calibration (PC) | ✅ 4-click rack → mm grid |
| BLE communication | 📋 Planned (Stage 2) |
| iOS app | 📋 Planned (Stage 3) |
| Fiducial calibration | 📋 Planned (Stage 4) |
| Limit switches / homing | 📋 Planned |

---

## Getting Started

### Flash the ESP32 gantry firmware

Requires [PlatformIO](https://platformio.org/).

```bash
cd LaserMarker_for_EP_tube/EP_Tube_CV
pio run --target upload
pio device monitor
```

Then type commands in the serial monitor (115200 baud):
```
X1600 Y3200
```

> **Important:** Manually move the gantry to the **bottom-left corner** before powering on. There are no limit switches yet — the firmware uses software limits only (X: 0–12000, Y: 0–9000 steps).

### Run the CV pipeline (PC)

Requires Python 3.10+.

```bash
pip install ultralytics opencv-python numpy requests

cd LaserMarker_CV

# Train YOLOv8 (needs GPU for reasonable speed)
python train_gpu.py

# Run inference on a test image
python check_result.py

# Interactive homography calibration
python Homography_Calibration.py
# Click the 4 rack corners in order: TL → TR → BR → BL
```

### Collect training images (via ESP32 camera)

```bash
cd LaserMarker_for_EP_tube/EP_Tube_CV

# Edit STREAM_URL in the script to match your ESP32's IP
python collect_force.py
# SPACE = save frame | Q = quit
```

---

## Computer Vision Notes

Two approaches were explored in parallel:

**Classical OpenCV** — good for understanding and fast to iterate:
- Hough circle transform on the blue channel (orange rack → blue is the discriminating channel)
- Contour analysis (area, circularity, solidity filters)
- Watershed segmentation for touching/overlapping tubes
- Interactive trackbar scripts for parameter tuning

**YOLOv8n** — better recall across lighting conditions:
- Trained on ~12 labeled images (needs more — target 200+)
- Single class: `tube`
- Best result after 7 training runs: mAP50 ≈ 0.82, Precision ≈ 1.0, Recall ≈ 0.61
- Will be quantized to CoreML for on-device iOS inference

---

## Motivation

This started as a real lab automation need and a personal challenge: how much of an intelligent hardware product can one person build from scratch with affordable components (target BOM under $50)?

It touches computer vision, embedded systems, mobile development, motion control, and mechanical design — which made it a worthwhile learning project even before it's fully finished.

---

## License

MIT — use freely, attribution appreciated.
