# Jedi Lightsabers — Real-Time Computer-Vision

A real-time computer-vision application that turns detected hands into glowing Jedi-style lightsabers with gesture ignition, dynamic distance scaling, blade recoil physics, spark showers, and competitive duel parry combat.

```
                     ┌──────────────────┐
                     │   Webcam Feed    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Camera Module   │ (Selfie horizontal flip, OS-agnostic)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   HandTracker    │ (MediaPipe Tasks VIDEO mode)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Gesture Ignition │ (Two-Finger Blue / Force Push Red)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Saber Geometry  │ (Calibrated 3D perspective scaling)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Combat & Recoil  │ (Parry/disarm logic, harmonic recoil)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  SaberRenderer   │ (Additive bloom, glowing trails,
                     └────────┬─────────┘  scaled core layers, procedural hilt)
                              │
                              ▼
                     ┌──────────────────┐
                     │  Display Window  │ (Interactive real-time 60 FPS)
                     └──────────────────┘
```

---

## Features

- **Gesture-Based Ignition & Hand Binding**:
  - **Jedi Blue**: Two-Finger Focus Pose (index and middle fingers extended toward the camera, ring and pinky curled into palm).
  - **Sith Red**: Force Push Pose (open palm facing camera with all fingers extended and spread).
  - **Direct Hand Binding**: Whichever hand performs the gesture is bound to that lightsaber with a smooth plasma ignition animation.
- **Calibrated 3D Perspective Scaling**:
  - Dynamic scaling based on hand distance relative to reference hand span.
  - Blade length, plasma core thickness, multi-layer bloom flare, and hilt radius scale proportionally to preserve 3D perspective without bloating into thick sausages.
- **Pre-Duel Countdown & Readiness Gate**:
  - 3-second animated on-screen countdown (`"DUEL IN: 3... 2... 1... ENGAGE!"`) initiates when two combatants ready their sabers.
  - Disarms remain locked during countdown so fighters can spar without early knockouts.
- **Strict Two-Person Limit with Spatial Hysteresis**:
  - Strictly at most 2 lightsabers on screen, each assigned to a different person.
  - If 1 person raises two hands, only their dominant hand holds a saber.
  - Spatial hysteresis prevents combatants from merging into a single entity when crossing blades close together.
- **Parry Combat & Disarm Mechanics**:
  - High-speed strikes evaluate defender leverage and blade angle.
  - Blocking with the **forte** (lower 72% of the blade) at a crossing angle $\ge 30^\circ$ executes a successful **Parry** (`"PARRIED!"`).
  - Strikers hitting the weak **foible** (tip) or slipping past poor angles trigger a **Missed Parry Disarm** (`"DISARMED! (MISSED PARRY: WEAK TIP)"`).
- **Floor-Bouncing Disarm Physics**:
  - When disarmed, the saber tumbles along a parabolic arc under gravity ($950\text{px/s}^2$), bounces off the floor with restitution and friction, emits ground sparks, and retracts into the hilt before re-arming.
- **Elastic Blade Recoil Physics**:
  - Damped harmonic oscillator (`omega=38.0`, `zeta=0.75`) providing tactile spring recoil deflection when blades clash.
- **Newtonian Spark Particle Physics**:
  - High-velocity sparks spray along contact planes with gravity ($850\text{px/s}^2$), air drag, thermal color decay (white-hot $\to$ electric yellow $\to$ amber $\to$ cooling red embers), and motion streak rendering.
- **Procedural Luminous Blade**:
  - Multi-pass additive rendering (`cv2.add`) with saturating outer bloom, bright inner core, and white-hot center.
- **High-Performance Low-Latency Pipeline**:
  - Half-resolution Gaussian glow blurring, pre-allocated canvas buffers, and camera buffer queue optimization (`CAP_PROP_BUFFERSIZE=1`).

---

## Quick Start (OS-Agnostic)

### Prerequisites

- **Python**: Python 3.9, 3.10, 3.11, or 3.12 installed.
- **Webcam**: Built-in or external USB camera.
- Supported Operating Systems: **macOS**, **Linux**, **Windows 10/11**.

---

### Step 1: Clone Repository

```bash
git clone https://github.com/shivamjkulkarni/lightsabers.git
cd lightsabers
```

---

### Step 2: Environment Setup & Activation

Choose the instructions for your operating system and shell:

#### macOS / Linux

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# If script execution is restricted, allow it for the current process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Windows (Command Prompt / CMD)

```cmd
:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
.venv\Scripts\activate.bat

:: Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Download MediaPipe Model Asset

Download Google's official `hand_landmarker.task` model asset (~7.5 MB) using the cross-platform download script:

**macOS / Linux:**
```bash
python3 scripts/download_model.py
```

**Windows (PowerShell or CMD):**
```bash
python scripts/download_model.py
```

*(Alternatively, use Python module syntax on any OS: `python -m scripts.download_model`)*

---

### Step 4: Run the Application

Launch the real-time lightsaber application:

**Cross-Platform (All Operating Systems):**
```bash
python -m src.main
```

*(On macOS/Linux you can also use `python3 -m src.main` or `python3 src/main.py`)*

---

## How to Play

1. **Ignite Blue Lightsaber (Jedi)**:
   - Hold up your hand facing the camera in a **Two-Finger Focus Pose** (index and middle fingers extended straight up/forward, ring and pinky curled into your palm).
   - Hold for ~0.15s to ignite.
2. **Ignite Red Lightsaber (Sith)**:
   - Hold up your hand facing the camera in an **Open Palm Force Push** (all 4 fingers extended and spread forward).
   - Hold for ~0.15s to ignite.
3. **Duel Countdown**:
   - When two combatants ignite their sabers, a 3-second countdown (`3... 2... 1... ENGAGE!`) will appear on screen.
4. **Parrying & Clashing**:
   - Swing your blade into your opponent's blade.
   - **Successful Parry**: Intercept your opponent's attack with the lower 70% (forte) of your blade at an angle $\ge 30^\circ$.
   - **Disarm**: If you get struck on the weak tip (foible) of your blade or slip past parallel, you will be disarmed! Your saber will fly out of your hand, bounce on the floor with sparks, and retract.

---

## Keyboard Controls

| Key | Action |
|:---|:---|
| **`Q`** / **`ESC`** | Quit application cleanly |
| **`G`** | Toggle soft luminous bloom/glow effect |
| **`T`** | Toggle motion trails |
| **`D`** | Toggle debug HUD (landmarks, skeleton, tracking state) |
| **`C`** | Spawn test clash sparks at screen center |
| **`K`** | Test disarm trigger on Person 2 |
| **`R`** | Reset duel countdown, smoother, and particle physics |

---

## Running Automated Tests

Run the full automated test suite without requiring a camera or GUI window:

**Cross-Platform (macOS / Linux / Windows):**
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All **38 unit tests** execute in under 0.01 seconds across:
- `test_gestures.py`: Two-finger focus pose, force push recognition, fist rejection, and debounce state machine.
- `test_duel_mechanics.py`: Distance scaling calibration, single-person filter, clash proximity hysteresis, and parry/disarm combat rules.
- `test_collision.py`: Line intersection math, harmonic recoil oscillator, spark dynamics, and floor bounce physics.
- `test_geometry.py`: Vector normalization, saber angles, endpoint projection, and arm extension.
- `test_smoothing.py`: Exponential moving average (EMA) filters and canvas buffer reuse.

---

## Project Structure

```
lightsabers/
├── assets/
│   └── hand_landmarker.task        # MediaPipe Hand Landmarker task model (auto-downloaded)
├── scripts/
│   ├── demo_collision_physics.py   # Simulation visual generator for collision & disarm
│   ├── demo_duel_mechanics.py      # Simulation visual generator for duel countdown & parries
│   ├── demo_gesture_and_scaling.py # Simulation visual generator for gestures & 3D scaling
│   └── download_model.py           # Cross-platform model asset downloader
├── src/
│   ├── __init__.py
│   ├── camera.py                   # OpenCV VideoCapture wrapper (selfie flip & queue optimization)
│   ├── collision.py                # Line intersection, recoil oscillator, parry & disarm evaluator
│   ├── config.py                   # Centralized configuration dataclasses & combat settings
│   ├── falling_saber.py            # Newtonian tumbling, floor bounce mechanics, ground sparks
│   ├── geometry.py                 # Distance-based blade scaling, perspective scaling, vector math
│   ├── gestures.py                 # Two-finger focus & force push recognition with temporal debouncing
│   ├── hand_tracker.py             # MediaPipe HandLandmarker + TwoPersonTracker spatial hysteresis
│   ├── main.py                     # Application entry point, duel state machine, HUD & event loop
│   ├── particles.py                # Newtonian spark particle system with thermal color decay
│   ├── renderer.py                 # Additive bloom, procedural hilt, scaled core layers, trails
│   ├── saber.py                    # SaberInstance state, dynamic scaling smoothing, ignition progress
│   └── smoothing.py                # PointSmoother and DirectionSmoother EMA filters
├── tests/
│   ├── __init__.py
│   ├── test_collision.py           # Tests for collision, recoil, sparks, and falling saber
│   ├── test_duel_mechanics.py      # Tests for distance scaling, two-person tracker & parry rules
│   ├── test_geometry.py            # Tests for vector math, scaling ratios, and geometry
│   ├── test_gestures.py            # Tests for Two-Finger pose, Force Push pose, and debounce
│   └── test_smoothing.py           # Tests for EMA filters, single-hand filter, canvas buffer
├── requirements.txt                # Dependencies (mediapipe, opencv-python, numpy)
└── README.md
```

---

## Platform-Specific Troubleshooting

### macOS

- **Camera Permission Denied**:
  - macOS requires explicit camera permissions for terminal applications.
  - Navigate to **System Settings → Privacy & Security → Camera**.
  - Ensure the toggle is switched ON for your terminal or IDE (e.g., **Terminal**, **iTerm2**, **VS Code**, or **Cursor**).
- **Multiple Webcams / Continuity Camera**:
  - If macOS defaults to an iPhone Continuity Camera instead of your built-in webcam, modify `camera_index = 0` in [`src/config.py`](file:///Users/shivam/Desktop/workspace./lightsabers/src/config.py) to `1` or `2`.

### Windows

- **Camera Permission Denied**:
  - Navigate to **Settings → Privacy & Security → Camera**.
  - Ensure **Camera access** is turned ON.
  - Ensure **Let desktop apps access your camera** is turned ON.
- **DirectShow Camera Backend**:
  - If the camera fails to open with default MSMF on older Windows versions or USB capture cards, update `src/camera.py`:
    ```python
    self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
    ```
- **PowerShell Execution Policy**:
  - If `.venv\Scripts\Activate.ps1` gives an execution policy error:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ```

### Linux (Ubuntu / Debian / Fedora / Arch)

- **Video Device Permissions**:
  - Ensure your user account belongs to the `video` group to access `/dev/video*`:
    ```bash
    sudo usermod -aG video $USER
    ```
    *(Log out and log back in for changes to take effect).*
- **Missing OpenCV GUI / OpenGL Libraries**:
  - On minimal or server Linux installations, OpenCV requires X11/Wayland and OpenGL runtime libraries:
    - **Ubuntu / Debian**:
      ```bash
      sudo apt update && sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev
      ```
    - **Fedora / RHEL**:
      ```bash
      sudo dnf install -y mesa-libGL glib2 libSM libXext libXrender
      ```
    - **Arch Linux**:
      ```bash
      sudo pacman -S --needed glibc libglvnd libsm libxext libxrender
      ```

---

## License

MIT License. Open source for hack nights, Jedi training, and computer-vision experimentation.
