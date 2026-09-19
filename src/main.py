"""Main entry point for Lightsabers application."""

import pathlib
import sys
import time

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.camera import Camera
from src.hand_tracker import HandTracker, draw_hand_landmarks
from src.renderer import render_blade, render_hilt, render_trail
from src.saber import SaberInstance


def main() -> None:
    print("Starting Lightsabers — Dual Saber Mode...")
    print("Controls: 'q'/ESC: quit | 'g': toggle glow | 't': toggle trail | 'd': toggle debug | 'r': reset")

    window_name = "Jedi Lightsabers"
    prev_time = time.time()
    fps = 0.0
    blade_length = 320.0
    trail_duration = 0.35  # seconds

    show_glow = True
    show_trail = True
    show_debug = False

    # Independent sabers for left and right hands
    # Left hand: Jedi Guardian Blue/Cyan | Right hand: Jedi Consular Emerald Green
    sabers = {
        "Left": SaberInstance(handedness="Left", color=(255, 120, 30)),
        "Right": SaberInstance(handedness="Right", color=(30, 255, 120)),
    }

    try:
        with Camera(camera_index=0, width=1280, height=720, flip_horizontal=True) as camera, \
             HandTracker(num_hands=2) as tracker:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: Received empty frame from camera, skipping...")
                    continue

                # Compute frame FPS
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
                prev_time = curr_time

                # Detect up to 2 hands
                observations = tracker.detect(frame)

                # Match detected hands to sabers by handedness
                matched_obs = {}
                unassigned = []
                for obs in observations:
                    if obs.handedness in sabers and obs.handedness not in matched_obs:
                        matched_obs[obs.handedness] = obs
                    else:
                        unassigned.append(obs)

                # Assign any unassigned hand observation to free saber slot
                for hand_key in ("Left", "Right"):
                    if hand_key not in matched_obs and unassigned:
                        matched_obs[hand_key] = unassigned.pop(0)

                # Update each saber instance
                for hand_key, saber in sabers.items():
                    saber.update(
                        matched_obs.get(hand_key),
                        curr_time=curr_time,
                        blade_length=blade_length,
                        trail_duration=trail_duration,
                    )

                # 1. Render motion trails for all sabers
                if show_trail:
                    for saber in sabers.values():
                        if len(saber.trail_history) >= 2:
                            render_trail(
                                frame,
                                saber.trail_history,
                                saber.color,
                                trail_duration,
                                curr_time,
                            )

                # 2. Render hilts and luminous blades for active sabers
                for saber in sabers.values():
                    if saber.is_active and saber.current_pivot and saber.current_endpoint and saber.current_direction:
                        render_hilt(frame, saber.current_pivot, saber.current_direction, hilt_length=50.0)
                        render_blade(frame, saber.current_pivot, saber.current_endpoint, saber.color, show_glow=show_glow)

                # 3. Optional Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Hands: {len(observations)} | "
                        f"Sabers: {', '.join(active_sabers) or 'None'} | "
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
                    for saber in sabers.values():
                        saber.reset()

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
