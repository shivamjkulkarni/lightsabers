"""Main entry point for Lightsabers application."""

import pathlib
import sys
import time

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.camera import Camera
from src.config import AppConfig
from src.geometry import scale_length_for_resolution
from src.hand_tracker import HandTracker, draw_hand_landmarks, filter_one_hand_per_person
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    draw_trail_on_light_canvas,
    render_hilt,
)
from src.saber import SaberInstance


def main() -> None:
    config = AppConfig()

    print("=" * 60)
    print("      JEDI LIGHTSABERS — REAL-TIME COMPUTER VISION")
    print("=" * 60)
    print("Controls:")
    print("  'q' / ESC : Quit")
    print("  'g'       : Toggle glow effect")
    print("  't'       : Toggle motion trail")
    print("  'd'       : Toggle debug landmarks & HUD")
    print("  'r'       : Reset saber tracking states")
    print("=" * 60)

    window_name = "Jedi Lightsabers"
    prev_time = time.time()
    fps = 0.0

    show_glow = config.show_glow
    show_trail = config.show_trail
    show_debug = config.debug_mode

    # Pre-allocated canvas buffer to eliminate memory allocations and GC latency
    canvas_buffer = CanvasBuffer(config.camera.width, config.camera.height)

    # Saber slots: Person 1 (Jedi Blue) and Person 2 (Sith Red)
    sabers = {
        "Person1": SaberInstance(
            handedness="Person1",
            color=config.saber.jedi_blue,
            alpha_pivot=config.saber.alpha_pivot,
            alpha_direction=config.saber.alpha_direction,
            grace_period=config.saber.grace_period,
        ),
        "Person2": SaberInstance(
            handedness="Person2",
            color=config.saber.sith_red,
            alpha_pivot=config.saber.alpha_pivot,
            alpha_direction=config.saber.alpha_direction,
            grace_period=config.saber.grace_period,
        ),
    }

    try:
        with Camera(
            camera_index=config.camera.camera_index,
            width=config.camera.width,
            height=config.camera.height,
            flip_horizontal=config.camera.flip_horizontal,
        ) as camera, HandTracker(
            model_path=config.tracker.model_path,
            num_hands=config.tracker.max_hands,
            min_detection_confidence=config.tracker.min_detection_confidence,
            min_presence_confidence=config.tracker.min_presence_confidence,
            min_tracking_confidence=config.tracker.min_tracking_confidence,
        ) as tracker:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: Received empty frame from camera, skipping...")
                    continue

                frame_height, frame_width = frame.shape[:2]
                blade_length = scale_length_for_resolution(
                    config.saber.base_blade_length,
                    frame_height,
                    config.saber.reference_height,
                )

                # Compute frame FPS
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
                prev_time = curr_time

                # Detect hands
                raw_observations = tracker.detect(frame)

                # Filter so strictly one hand per person is tracked
                observations = filter_one_hand_per_person(
                    raw_observations,
                    same_person_max_distance_ratio=config.tracker.same_person_max_distance_ratio,
                )

                # Map observations to saber slots (Person 1 = Blue, Person 2 = Red)
                if len(observations) == 1:
                    sabers["Person1"].update(
                        observations[0],
                        curr_time=curr_time,
                        blade_length=blade_length,
                        trail_duration=config.saber.trail_duration,
                    )
                    sabers["Person2"].update(
                        None,
                        curr_time=curr_time,
                        blade_length=blade_length,
                        trail_duration=config.saber.trail_duration,
                    )
                elif len(observations) >= 2:
                    # Two distinct people in frame: sort left-to-right
                    obs_sorted = sorted(observations, key=lambda o: o.wrist[0])
                    sabers["Person1"].update(
                        obs_sorted[0],
                        curr_time=curr_time,
                        blade_length=blade_length,
                        trail_duration=config.saber.trail_duration,
                    )
                    sabers["Person2"].update(
                        obs_sorted[1],
                        curr_time=curr_time,
                        blade_length=blade_length,
                        trail_duration=config.saber.trail_duration,
                    )
                else:
                    for saber in sabers.values():
                        saber.update(
                            None,
                            curr_time=curr_time,
                            blade_length=blade_length,
                            trail_duration=config.saber.trail_duration,
                        )

                # 1. Render physical hilts inside hand grips
                for saber in sabers.values():
                    if (
                        saber.is_active
                        and saber.current_hilt_start
                        and saber.current_hilt_end
                        and saber.current_direction
                    ):
                        render_hilt(
                            frame,
                            saber.current_hilt_start,
                            saber.current_hilt_end,
                            saber.current_direction,
                        )

                # 2. Reusable pre-allocated light canvas (zero-allocation)
                light_canvas = canvas_buffer.reset_and_get(frame.shape)

                # Motion trails (faster fade, 20% lower opacity)
                if show_trail:
                    for saber in sabers.values():
                        if len(saber.trail_history) >= 2:
                            draw_trail_on_light_canvas(
                                light_canvas,
                                saber.trail_history,
                                saber.color,
                                config.saber.trail_duration,
                                curr_time,
                            )

                # Luminous blades extending outward from hands
                for saber in sabers.values():
                    if (
                        saber.is_active
                        and saber.current_emitter
                        and saber.current_endpoint
                    ):
                        draw_blade_on_light_canvas(
                            light_canvas,
                            saber.current_emitter,
                            saber.current_endpoint,
                            saber.color,
                            curr_time=curr_time,
                        )

                # 3. Additive bloom and compositing with saturating addition
                composite_light_layer(frame, light_canvas, show_glow=show_glow)

                # 4. Optional Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, raw_observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Raw Hands: {len(raw_observations)} | "
                        f"Filtered: {len(observations)} | Active: {', '.join(active_sabers) or 'None'}"
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
        print("Tip: Run 'python3 scripts/download_model.py' to download the model asset.", file=sys.stderr)
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
