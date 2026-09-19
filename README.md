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

- **Single-Hand Tracking Per Person**: Detects hands across combatants; if one person raises two hands, only their primary hand holds a saber. If two distinct people enter frame, Person 1 gets Jedi Blue (`(255, 90, 20)`) and Person 2 gets Sith Red (`(30, 30, 255)`).
- **Line Segment Blade Collision Physics**: Continuous 2D intersection testing between active blade segments.
- **Elastic Blade Recoil**: Damped harmonic oscillator (`omega=38.0`, `zeta=0.75`) providing tactile spring recoil deflection when sabers clash.
- **Newtonian Spark Particle Physics**: High-velocity sparks spray outward from clashes with gravitational acceleration ($850\text{px/s}^2$), exponential air drag, thermal color decay (white-hot $\to$ electric yellow $\to$ amber $\to$ cooling red embers), and motion streak rendering.
- **Floor-Bouncing Disarm Mechanics**: Fast, forceful strikes disarm the defending combatant. The dropped lightsaber tumbles through the air, bounces off the floor with restitution and friction, emits ground sparks, and smoothly retracts back into the hilt before re-arming.
- **Procedural Luminous Blade**: Multi-layer procedural rendering with saturating outer bloom, bright inner core, and white-hot center using saturating arithmetic (`cv2.add`) to eliminate overflow artifacts.
- **High-Performance Zero-Latency Bloom**: Half-resolution Gaussian glow blurring, zero-allocation pre-allocated canvas buffers, and camera buffer queue optimization (`CAP_PROP_BUFFERSIZE=1`).
- **Dynamic Fading Motion Trails**: Tuned fast dissipation ($0.22\text{s}$) with natural exponential decay.
- **Temporal Jitter Smoothing**: Exponential Moving Average (EMA) filtering for pivot points and unit direction vectors, plus grace period.
- **Procedural Hilts**: Directional metallic grips, emitter collars, and pommel caps rendered around the hand grips.

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
| **`C`** | Spawn test clash sparks at screen center |
| **`K`** | Knock out / disarm Person 2 (test demonstration) |
| **`R`** | Reset tracking smoother, particles, and physics states |

---

## Project Structure

```
lightsabers/
├── assets/
│   └── hand_landmarker.task        # MediaPipe Hand Landmarker task model (auto-downloaded)
├── scripts/
│   ├── demo_collision_physics.py  # Simulation visual generator for collision & disarm
│   └── download_model.py           # Model asset download helper
├── src/
│   ├── __init__.py
│   ├── camera.py                   # OpenCV VideoCapture wrapper with selfie flip & queue optimization
│   ├── collision.py                # Line intersection, damped harmonic recoil, disarm trigger
│   ├── config.py                   # Centralized dataclass configurations & visual settings
│   ├── falling_saber.py            # Newtonian tumbling, floor bounces, ground sparks, blade retract
│   ├── geometry.py                 # Vector math, arm extension geometry, resolution scaling
│   ├── hand_tracker.py             # MediaPipe Tasks HandLandmarker + single-hand filter per person
│   ├── main.py                     # Main application entry point, event loop & physics integration
│   ├── particles.py                # Newtonian spark particle system with thermal color decay
│   ├── renderer.py                 # Multi-pass saturating bloom, procedural hilt, trails, canvas buffer
│   ├── saber.py                    # SaberInstance state, velocity tracking, recoil & disarm cooldown
│   └── smoothing.py                # PointSmoother and DirectionSmoother EMA filters
├── tests/
│   ├── __init__.py
│   ├── test_collision.py           # Unit tests for collision, recoil, sparks, and falling saber
│   ├── test_geometry.py            # Unit tests for vector math and geometry
│   └── test_smoothing.py           # Unit tests for EMA smoothing, single-hand filter, and canvas buffer
├── requirements.txt                # Dependencies (mediapipe, opencv-python, numpy)
└── README.md
```

---

## Running Unit Tests

Run the test suite without needing a camera:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

All 25 unit tests for collision detection, spring recoil physics, spark particle dynamics, falling saber floor bounces, vector normalization, single-hand filtering, and canvas buffer reuse run in under 0.02 seconds.

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
