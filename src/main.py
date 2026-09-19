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


def main() -> None:
    print("Starting Lightsabers — Hand Tracking Mode...")
    print("Press 'q' or ESC to exit.")

    window_name = "Jedi Lightsabers"
    prev_time = time.time()
    fps = 0.0

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

                # Draw tracked hand landmarks
                draw_hand_landmarks(frame, observations)

                # Overlay status
                status_text = f"FPS: {fps:.1f} | Hands: {len(observations)}"
                cv2.putText(
                    frame,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
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
