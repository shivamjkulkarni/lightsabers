"""Main entry point for Lightsabers application."""

import collections
import pathlib
import sys
import time

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.camera import Camera
from src.geometry import calculate_saber_direction, calculate_saber_endpoint
from src.hand_tracker import HandTracker, draw_hand_landmarks
from src.renderer import render_blade, render_hilt, render_trail
from src.smoothing import DirectionSmoother, PointSmoother


def main() -> None:
    print("Starting Lightsabers — Visual Prototype...")
    print("Controls: 'q'/ESC: quit | 'g': toggle glow | 't': toggle trail | 'd': toggle debug | 'r': reset")

    window_name = "Jedi Lightsabers"
    prev_time = time.time()
    fps = 0.0
    blade_length = 320.0
    trail_duration = 0.35  # seconds

    show_glow = True
    show_trail = True
    show_debug = False

    pivot_smoother = PointSmoother(alpha=0.65)
    direction_smoother = DirectionSmoother(alpha=0.60)
    trail_history: collections.deque = collections.deque()

    # Saber blade color in BGR (electric blue/cyan)
    blade_color = (255, 120, 30)

    try:
        with Camera(camera_index=0, width=1280, height=720, flip_horizontal=True) as camera, \
             HandTracker() as tracker:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: Received empty frame from camera, skipping...")
                    continue

                # Compute frame FPS
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
                prev_time = curr_time

                # Detect hands
                observations = tracker.detect(frame)

                # Smooth and calculate saber for the primary detected hand
                current_direction = None
                if observations:
                    obs = observations[0]
                    smooth_pivot = pivot_smoother.update(obs.wrist)
                    raw_direction = calculate_saber_direction(obs.wrist, obs.index_mcp, obs.index_tip)
                    current_direction = direction_smoother.update(raw_direction)
                    endpoint = calculate_saber_endpoint(smooth_pivot, current_direction, blade_length)

                    # Store in trail history: (pivot, endpoint, timestamp)
                    trail_history.append((smooth_pivot, endpoint, curr_time))
                else:
                    pivot_smoother.reset()
                    direction_smoother.reset()

                # Expire old trail points
                while trail_history and (curr_time - trail_history[0][2] > trail_duration):
                    trail_history.popleft()

                # 1. Render motion trail
                if show_trail and len(trail_history) >= 2:
                    render_trail(frame, trail_history, blade_color, trail_duration, curr_time)

                # 2. Render hilt & luminous blade
                if observations and trail_history and current_direction is not None:
                    curr_pivot, curr_endpoint, _ = trail_history[-1]
                    render_hilt(frame, curr_pivot, current_direction, hilt_length=50.0)
                    render_blade(frame, curr_pivot, curr_endpoint, blade_color, show_glow=show_glow)

                # 3. Optional Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, observations)
                    status_text = (
                        f"FPS: {fps:.1f} | Hands: {len(observations)} | "
                        f"Glow: {'ON' if show_glow else 'OFF'} | "
                        f"Trail: {'ON' if show_trail else 'OFF'}"
                    )
                    cv2.putText(
                        frame,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, frame)

                # Check window close button (X)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    break
                elif key in (ord("g"), ord("G")):  # 'g' to toggle glow
                    show_glow = not show_glow
                elif key in (ord("t"), ord("T")):  # 't' to toggle trail
                    show_trail = not show_trail
                elif key in (ord("d"), ord("D")):  # 'd' to toggle debug
                    show_debug = not show_debug
                elif key in (ord("r"), ord("R")):  # 'r' to reset
                    pivot_smoother.reset()
                    direction_smoother.reset()
                    trail_history.clear()

    except FileNotFoundError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nCamera Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting upon user interrupt...")
    finally:
        cv2.destroyAllWindows()
        print("Application stopped cleanly.")


if __name__ == "__main__":
    main()
