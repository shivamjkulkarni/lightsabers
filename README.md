# Jedi Lightsabers — Real-Time Computer-Vision

A real-time computer-vision application that turns detected hands into glowing Jedi-style lightsabers with gesture ignition in dual chambers, 3-round competitive duel matches, dynamic distance scaling, spark showers, parry combat, and persistent winner victory screens.

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
                     │ Dual Chambers    │ (Left Jedi Blue / Right Sith Red)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Saber Geometry  │ (Calibrated 3D perspective scaling)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Combat & Parries │ (Parry/disarm logic, 3-round match)
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

- **Dual Independent Ignition Chambers & Spatial Segregation**:
  - **Left Chamber**: Dedicated exclusively to Jedi Blue. Place your hand inside the left holographic box and form a **Two-Finger Focus Pose**.
  - **Right Chamber**: Dedicated exclusively to Sith Red. Place your hand inside the right holographic box and form a **Force Push Pose**.
  - **Independent Chamber Removal**: Each chamber disappears from the screen as soon as its respective lightsaber ignites.
- **3-Round Match System (Best-of-3)**:
  - Matches are contested over up to 3 rounds (first to 2 round wins).
  - Real-time scoreboard tracked at the top of the HUD: `ROUND 1/3   [ JEDI BLUE: 0  |  SITH RED: 0 ]`.
  - **Mandatory Fresh Re-Ignition Every Round**: When a round concludes via disarm, both sabers extinguish, and combatants must place their hands back into their respective chambers and re-form their poses to begin the next round.
- **Persistent Match Winner Screen**:
  - When a combatant secures 2 round wins, a persistent victory screen appears over a frosted dark backdrop.
  - Displays the victor banner (`*** JEDI BLUE WINS THE MATCH! ***`), final score, and remains on screen indefinitely until an explicit user input is received.
  - Press **`R`** to start a fresh rematch, or **`Q`** / **`ESC`** to quit.
- **Rock-Solid Winning Saber & Recoil Elimination**:
  - Oscillation recoil flutter has been completely eliminated so the winning saber remains constant and stable.
- **Pre-Duel Countdown & State Continuity**:
  - 3-second animated on-screen countdown (`"DUEL IN: 3... 2... 1... ENGAGE!"`) initiates once both combatants ignite their sabers.
  - **Monotonic Continuity**: Once ignited, sabers remain ignited and the countdown proceeds monotonically without resetting if a hand temporarily drifts off-screen or swings rapidly.
- **Spatial Hand Re-Acquisition**:
  - Person 1 (Jedi Blue) occupies the left screen partition; Person 2 (Sith Red) occupies the right.
  - Off-screen hands immediately re-bind to their respective lightsabers the moment they re-enter camera view without any arbitrary distance boundaries.
- **Dynamic Guard Poise System**:
  - Each combatant begins each round with **2 Guard Points** displayed in the HUD: `JEDI GUARD: [ + + ]   SITH GUARD: [ + + ]`.
  - Imperfect blocks (parallel blade slips or weak tip contacts) deplete guard poise (`! GUARD SHAKEN! [1 POISE LEFT] !`) rather than causing instant 1-hit round finishes.
  - A second broken block shatters guard poise completely, triggering a dramatic **DISARM**!
- **Active Deflection ("Perfect Parry") & Counter-Strike Advantage**:
  - Actively snapping your blade into an incoming strike at $\ge 35^\circ$ with the forte executes a **Perfect Parry**!
  - Emits a radiant 50-spark supernova burst and an expanding radial plasma shockwave ring.
  - Restores +1 Guard Poise and grants a **1.2-second Counter-Strike Window** (+35% blade speed priority).
- **Radial Plasma Shockwaves**:
  - High-energy clashes and parries generate expanding circular plasma wavefront rings rendered directly through additive Gaussian bloom.
- **Saber Locks & Rally Combos**:
  - Sustained blade contact (> 0.22s) triggers a **Saber Lock** with crackling electrical sparks and a push-off prompt.
  - Rapid back-and-forth clashes build **Rally Streaks** displayed on-screen (`>> RALLY x3! <<`, `*** RALLY x5! EPIC CLASH! ***`).
- **Floor-Bouncing Disarm Physics**:
  - When disarmed, the saber tumbles along a parabolic arc under gravity ($950\text{px/s}^2$), bounces off the floor with restitution and friction, emits ground sparks, and retracts into the hilt before re-arming.
- **Calibrated 3D Perspective Scaling**:
  - Dynamic scaling based on hand distance relative to reference hand span.
  - Blade length, plasma core thickness, multi-layer bloom flare, and hilt radius scale proportionally to preserve 3D perspective without bloating into thick sausages.
- **Strict Two-Person Limit with Spatial Hysteresis**:
  - Strictly at most 2 lightsabers on screen, each assigned to a different person.
  - If 1 person raises two hands, only their dominant hand holds a saber.
  - Spatial hysteresis prevents combatants from merging into a single entity when crossing blades close together.
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

## Gameplay & Match Rules

1. **Ignite Jedi Blue (Left Chamber)**:
   - Place your hand within the **Left Holographic Box** on screen.
   - Form a **Two-Finger Focus Pose** (index and middle fingers extended straight toward camera, ring and pinky curled into palm).
   - Once ignited, the Left chamber disappears.
2. **Ignite Sith Red (Right Chamber)**:
   - Place your hand within the **Right Holographic Box** on screen.
   - Form an **Open Palm Force Push** (all fingers extended and spread forward toward camera).
   - Once ignited, the Right chamber disappears.
3. **Pre-Duel Countdown**:
   - Once both sabers are ignited, the countdown begins (`3... 2... 1... ENGAGE!`).
4. **Parrying & Clashing**:
   - Attack with velocity. The defender must parry with their forte (lower blade) at $\ge 30^\circ$ to deflect attacks.
   - If a parry is missed (hit on the weak foible tip or slipped), the defender is disarmed!
5. **Round Progression & Victory**:
   - The victor receives a point. Both sabers extinguish, and players must re-enter their chambers to ignite for the next round.
   - First combatant to win 2 rounds wins the match! A persistent winner screen displays until **`R`** (rematch) or **`Q`** (quit) is pressed.

---

## Keyboard Controls

| Key | Action |
|:---|:---|
| **`Q`** / **`ESC`** | Quit application cleanly |
| **`R`** | Rematch / Reset match, scores, and duel state |
| **`G`** | Toggle soft luminous bloom/glow effect |
| **`T`** | Toggle motion trails |
| **`D`** | Toggle debug HUD (landmarks, skeleton, tracking state) |
| **`C`** | Spawn test clash sparks at screen center |
| **`K`** | Demo disarm Person 2 (Sith Red) -> Awards round win to Jedi |
| **`J`** | Demo disarm Person 1 (Jedi Blue) -> Awards round win to Sith |

---

## Running Automated Tests

Run the full automated test suite without requiring a camera or GUI window:

**Cross-Platform (macOS / Linux / Windows):**
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All **44 unit tests** execute in under 0.01 seconds across:
- `test_duel_mechanics.py`: 3-round match scoring, round saber extinction, dual chamber isolation, winner screen rendering, distance scaling, and parry combat rules.
- `test_gestures.py`: Two-finger focus pose, force push recognition, box containment checks, and temporal debounce.
- `test_collision.py`: Line intersection math, spark dynamics, and floor bounce physics.
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
│   ├── demo_ignition_box.py        # Simulation visual generator for ignition chamber
│   ├── demo_match_system.py        # Simulation visual generator for 3-round match system
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
│   ├── renderer.py                 # Additive bloom, procedural hilt, scaled core layers, trails, winner screen
│   ├── saber.py                    # SaberInstance state, dynamic scaling smoothing, ignition progress
│   └── smoothing.py                # PointSmoother and DirectionSmoother EMA filters
├── tests/
│   ├── __init__.py
│   ├── test_collision.py           # Tests for collision, sparks, and falling saber
│   ├── test_duel_mechanics.py      # Tests for match scoring, dual chambers, winner screen, scaling & parries
│   ├── test_geometry.py            # Tests for vector math, scaling ratios, and geometry
│   ├── test_gestures.py            # Tests for Two-Finger pose, Force Push pose, box containment, debounce
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
