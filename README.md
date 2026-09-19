# Jedi Lightsabers — Real-Time Computer-Vision

A real-time computer-vision application that turns detected hands into glowing Jedi-style lightsabers with fading luminous trails, procedural hilts, and smooth temporal filtering.

```
                     ┌──────────────────┐
                     │   Webcam Feed    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Camera Module   │ (Selfie horizontal flip)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   HandTracker    │ (MediaPipe Tasks VIDEO mode)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Saber Geometry  │ (Weighted MCP/Tip direction)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Temporal Filters │ (EMA smoother & grace period)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  SaberRenderer   │ (Trails, half-res bloom,
                     └────────┬─────────┘  core layers, procedural hilt)
                              │
                              ▼
                     ┌──────────────────┐
                     │  Display Window  │ (Interactive real-time 60 FPS)
                     └──────────────────┘
```

---

## Features

- **Dual Hand Tracking**: Independently tracks left and right hands simultaneously.
- **Iconic Dual Colors**: Left hand glows Jedi Guardian Blue/Cyan (`(255, 120, 30)`), Right hand glows Jedi Consular Emerald Green (`(30, 255, 120)`).
- **Procedural Luminous Blade**: Multi-layer procedural rendering with saturated outer glow, bright inner core, and white-hot center.
- **High-Performance Bloom**: Half-resolution Gaussian glow blurring and additive compositing for soft, luminous bloom at real-time speeds (30–60 FPS).
- **Smooth Motion Trails**: Fading polygon ribbons that trail behind blade swings and dissipate naturally over time.
- **Temporal Jitter Smoothing**: Exponential Moving Average (EMA) filtering for both pivot attachment and unit direction vectors, plus a grace-period fadeout when hands briefly drop out of view.
- **Procedural Hilts**: Directional metallic grips, emitter collars, and pommel caps rendered around the wrists.

---

## Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download MediaPipe Model Asset

Download Google's official `hand_landmarker.task` model (~7.5 MB):

```bash
python3 scripts/download_model.py
```

### 3. Run the Application

```bash
python3 -m src.main
# or
python3 src/main.py
```

---

## Keyboard Controls

| Key | Action |
|:---|:---|
| **`Q`** / **`ESC`** | Quit application cleanly |
| **`G`** | Toggle soft luminous bloom/glow effect |
| **`T`** | Toggle motion trails |
| **`D`** | Toggle debug mode (landmarks, skeleton, HUD status) |
| **`R`** | Reset tracking smoother and trail history |

---

## Project Structure

```
lightsabers/
├── assets/
│   └── hand_landmarker.task        # MediaPipe Hand Landmarker task model (auto-downloaded)
├── scripts/
│   └── download_model.py           # Model asset download helper
├── src/
│   ├── __init__.py
│   ├── camera.py                   # OpenCV VideoCapture wrapper with selfie flip
│   ├── config.py                   # Centralized dataclass configurations & visual settings
│   ├── geometry.py                 # Pure vector math, blade direction, resolution scaling
│   ├── hand_tracker.py             # MediaPipe Tasks HandLandmarker wrapper
│   ├── main.py                     # Main application entry point & event loop
│   ├── renderer.py                 # Procedural hilt, multi-layer blade, glow, and trails
│   ├── saber.py                    # SaberInstance state, trail history, and grace periods
│   └── smoothing.py                # PointSmoother and DirectionSmoother EMA filters
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py            # Unit tests for vector math and geometry
│   └── test_smoothing.py           # Unit tests for EMA smoothing and trail expiration
├── requirements.txt                # Dependencies (mediapipe, opencv-python, numpy)
└── README.md
```

---

## Running Unit Tests

Run the test suite without needing a camera:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

All 13 unit tests for vector normalization, zero-vector handling, direction estimation, resolution scaling, EMA smoothing, and trail expiration run in under 0.01 seconds.

---

## How the Computer Vision Pipeline Works

1. **Capture**: Frames are captured via OpenCV `VideoCapture(0)` at 1280x720 and mirrored horizontally (`cv2.flip(frame, 1)`) for a natural selfie-style mirror interaction.
2. **Hand Detection**: `HandTracker` runs Google's MediaPipe Tasks `HandLandmarker` in `RunningMode.VIDEO` with monotonically increasing millisecond timestamps. Up to 2 hands are detected and normalized coordinates are mapped to pixel coordinates.
3. **Orientation Geometry**: The blade pivot is anchored to the wrist landmark. The pointing direction combines `wrist -> index_mcp` (65% weight for structural hand bone orientation) and `wrist -> index_tip` (35% weight for intentional pointing), preventing wild flipping when curling fingers into a fist.
4. **Temporal Smoothing**:
   - Pivot coordinates use an EMA filter (`alpha = 0.65`).
   - Direction vectors use an EMA filter (`alpha = 0.60`) with unit-length normalization to avoid blade shrinkage during fast swings.
   - A 0.25-second grace period keeps the saber alive briefly if a hand is momentarily obstructed.
5. **Procedural Rendering**:
   - **Motion Ribbon**: Historical `(pivot, endpoint, timestamp)` pairs are connected into quadrilaterals with age-based alpha fading and rendered using additive blending (`cv2.add`).
   - **Bloom / Glow**: The blade is drawn to a half-resolution canvas, blurred with a Gaussian kernel, upscaled with bilinear interpolation, and composited additively for high frame rates.
   - **Blade Core**: Three concentric strokes (saturated outer, pastel inner, white-hot core) are rendered directly on the frame with rounded caps.
   - **Hilt**: A dark metallic cylinder with emitter and pommel rings is drawn extending backwards along the inverse blade direction.

---

## Troubleshooting

### Camera Not Found / Permission Denied
- On macOS, grant Terminal or your IDE camera access under **System Settings → Privacy & Security → Camera**.
- If multiple cameras are attached, change `camera_index = 0` in `src/config.py` to `1` or `2`.

### MediaPipe Model Missing
- Run `python3 scripts/download_model.py`.
- The model file must be located at `assets/hand_landmarker.task`.

---

## Future Hack Night Upgrades

- **Gesture-Based Activation**: Saber ignition / retraction on specific gestures (e.g. closed fist = ignite, open palm = extinguish).
- **Blade Clash Effects**: Detect intersections between the two blade line segments and draw bright sparks and flash bursts at the clash point.
- **Swing Velocity Streaks**: Modulate trail length and glow intensity based on angular velocity of the blade.
- **Audio Feedback**: Hum sound effect with pitch bending modulated by hand movement speed, plus clash sound effects on blade contact.
- **Background Segmentation**: Dim or tint the webcam background to make the glowing blades stand out dramatically in daylight.
