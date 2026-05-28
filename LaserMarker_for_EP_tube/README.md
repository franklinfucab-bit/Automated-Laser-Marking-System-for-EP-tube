# LaserMarker_for_EP_tube — ESP32 Firmware & Data Collection

Embedded firmware for the XIAO ESP32-S3 that controls a 2-axis stepper gantry and laser module, plus Python tools for collecting training images from the ESP32 camera stream.

---

## Hardware

### Motion Platform

| Part | Specification |
|------|--------------|
| Gantry | XY synchronous-belt cross slide (龙门同步带十字滑台铝型材) |
| X axis | 200 mm travel (140 mm usable stroke) |
| Y axis | 300 mm travel (240 mm usable stroke) |
| Steppers | 2× NEMA17 42mm, 3.3 V, 1.5 A |
| Drivers | 2× A4988 on dual-axis expansion board |
| Power supply | 12 V DC, ≥5 A (≥60 W switching PSU) |
| Controller | Seeed Studio XIAO ESP32-S3 |
| Laser | PWM diode laser module, 12 V, logic via GPIO 44 |

### GPIO Pin Map

| Pin | Signal | Component |
|-----|--------|-----------|
| GPIO 1 | EN_X | Stepper X enable |
| GPIO 2 | STEP_X | Stepper X step |
| GPIO 3 | DIR_X | Stepper X direction |
| GPIO 4 | EN_Y | Stepper Y enable |
| GPIO 5 | STEP_Y | Stepper Y step |
| GPIO 6 | DIR_Y | Stepper Y direction |
| GPIO 44 | LASER | Laser PWM (LEDC, 1 kHz, 8-bit) |

### Wiring Order

```
Switching PSU (12V, 5A+)
    │
    ├─→ A4988 Driver X  →  Stepper Motor X
    ├─→ A4988 Driver Y  →  Stepper Motor Y
    └─→ Laser Module
         └─→ Logic signal from ESP32 GPIO 44

ESP32 GPIO → A4988 (STEP / DIR / EN)
```

> **Home position:** Manually move the gantry to the **bottom-left corner** before powering on. No limit switches yet — software limits only.

**Software limits:**
- X: 0 – 12,000 steps
- Y: 0 – 9,000 steps
- Speed: 4,000 Hz | Acceleration: 15,000 steps/s²

---

## Project Structure

```
EP_Tube_CV/
├── platformio.ini         # Active build config (stepper only)
├── src/
│   ├── main.cpp           # ✅ Active firmware — serial X/Y control
│   └── cv_main            # 🔧 Draft — camera stream + gantry (WIP)
├── laser_test.cpp          # Laser PWM test (not built — outside src/)
├── collect_data.py         # Capture training images (OpenCV VideoCapture)
├── collect_force.py        # Capture training images (raw MJPEG parser, recommended)
└── src/origin_ini.ini      # Alternate config with esp32-camera + PSRAM
```

---

## Firmware: `src/main.cpp` (active)

Accepts `X… Y…` commands over Serial (115200 baud) and moves the gantry.

```bash
cd EP_Tube_CV
pio run --target upload
pio device monitor
```

**Example session:**
```
> X1600 Y3200    # move to position (1600, 3200) in steps
> X0 Y0          # return to origin
> X99999 Y0      # rejected — out of bounds
```

Both axes move simultaneously. Out-of-range commands are rejected with an error message (in Chinese).

---

## Planned: BLE GATT Firmware (Stage 2)

The next firmware milestone replaces the USB serial connection with Bluetooth Low Energy, so the iOS app can send coordinates wirelessly.

**Planned architecture:**
- ESP32 acts as a **BLE GATT Server**
- Advertises a custom characteristic (writable)
- Flutter app connects via BLE and writes `X1600 Y3200\n` strings
- ESP32 reads the characteristic → executes gantry move

---

## Laser: `laser_test.cpp`

Blinks the laser at ~2% power (duty 5/255) to verify PWM wiring. Not compiled by default (lives outside `src/`).

To test:
1. Move `laser_test.cpp` into `src/`
2. Rename `main.cpp` temporarily (e.g. `main.cpp.bak`)
3. Flash, verify laser blinks, then restore

```
GPIO 44 → LEDC channel → 1 kHz, 8-bit
Duty 5/255 for 2 s → off 2 s → repeat
```

> Note: the file is missing `#include <Arduino.h>` — add it before flashing.

---

## Camera Firmware: `src/cv_main` (draft, broken)

Intended to serve an MJPEG stream at `http://<ESP32-IP>/` so PC tools can capture training frames.

**Current state:** Two `setup()` / `loop()` definitions — won't compile. Missing includes and WiFi credentials. No `.cpp` extension so PlatformIO ignores it.

**To enable camera build:** use `src/origin_ini.ini` as the active `platformio.ini` (it includes `esp32-camera` and PSRAM flags).

---

## Data Collection Scripts

Both scripts connect to the ESP32 camera HTTP stream and let you save frames for the YOLO training dataset.

### `collect_force.py` (recommended)

Parses raw MJPEG bytes — more robust, handles corrupt frames.

```bash
python collect_force.py
# SPACE = save frame  |  Q = quit
# Saves to: dataset/raw_images/scan_N.jpg
```

### `collect_data.py`

Uses `cv2.VideoCapture` with auto-reconnect.

```bash
python collect_data.py
# SPACE = save frame  |  Q = quit
# Saves to: dataset/raw_images/rack_scan_N.jpg
```

**Stream URL** (edit in script to match your ESP32's IP):
```
http://100.100.22.159/
```

---

## Dependencies

```bash
pip install opencv-python requests numpy
```

PlatformIO handles C++ dependencies (`FastAccelStepper`) automatically on build.

---

## Roadmap

- [ ] Fix `cv_main` — unify camera + gantry into one firmware
- [ ] Replace serial with BLE GATT server (Stage 2)
- [ ] Receive and parse coordinate lists from iOS app over BLE
- [ ] Integrate laser PWM into main gantry firmware
- [ ] Add homing routine with physical limit switches
- [ ] Mount limit switches (MVP: hot glue; final: 3D printed bracket)
