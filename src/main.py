"""Main entry point for Lightsabers application."""

import math
import pathlib
import sys
import time
from typing import List

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.camera import Camera
from src.collision import BladeRecoilController, check_blade_collision, check_disarm_trigger
from src.config import AppConfig
from src.falling_saber import FallingSaber
from src.geometry import scale_length_for_resolution
from src.hand_tracker import HandTracker, draw_hand_landmarks, filter_one_hand_per_person
from src.particles import ParticleSystem
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
    print("  'c'       : Spawn test clash sparks")
    print("  'k'       : Knock out / disarm Person 2 (test demo)")
    print("  'r'       : Reset saber tracking & physics states")
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

    # Physical simulations: sparks, elastic recoil, falling sabers
    particle_system = ParticleSystem(gravity=850.0, drag=0.93)
    recoil_controller = BladeRecoilController(omega=38.0, zeta=0.75)
    falling_sabers: List[FallingSaber] = []

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

                # Compute frame delta time and FPS
                curr_time = time.time()
                dt = max(1e-4, min(curr_time - prev_time, 0.05))
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
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

                # Update elastic blade recoil physics
                recoil_controller.update(dt)

                # Apply recoil deflection to active sabers
                for slot_name, saber in sabers.items():
                    if saber.is_active:
                        angle = recoil_controller.get_recoil_angle(slot_name)
                        saber.apply_recoil(angle, blade_length)

                # Check blade-on-blade collision and disarm mechanics
                s1 = sabers["Person1"]
                s2 = sabers["Person2"]
                if (
                    s1.is_active
                    and s2.is_active
                    and s1.current_emitter
                    and s1.current_endpoint
                    and s2.current_emitter
                    and s2.current_endpoint
                ):
                    collision = check_blade_collision(
                        s1.current_emitter,
                        s1.current_endpoint,
                        s2.current_emitter,
                        s2.current_endpoint,
                    )
                    if collision is not None:
                        # 1. Spawn radiant clash sparks along contact plane
                        particle_system.spawn_clash_sparks(
                            collision.clash_point[0],
                            collision.clash_point[1],
                            collision.normal[0],
                            collision.normal[1],
                            count=28,
                        )

                        # 2. Apply spring-damper recoil impulses
                        if s1.current_direction and s2.current_direction:
                            recoil_controller.apply_clash_impulse(
                                "Person1",
                                "Person2",
                                collision.normal,
                                s1.current_direction,
                                s2.current_direction,
                                impulse_magnitude=0.45,
                            )

                        # 3. Check for high-velocity disarm strike
                        v_rel = s1.tip_speed + s2.tip_speed
                        disarmed_slot = check_disarm_trigger(
                            collision,
                            v_rel=v_rel,
                            s1_speed=s1.tip_speed,
                            s2_speed=s2.tip_speed,
                            disarm_threshold=480.0,
                        )
                        if disarmed_slot is not None:
                            victim = sabers[disarmed_slot]
                            floor_y = frame_height - 20.0
                            knock_vx = 220.0 if disarmed_slot == "Person2" else -220.0
                            knock_vy = -260.0
                            rot_speed = 7.5 if disarmed_slot == "Person2" else -7.5
                            init_angle = (
                                math.atan2(victim.current_direction[1], victim.current_direction[0])
                                if victim.current_direction
                                else 0.0
                            )
                            hilt_pos = (
                                victim.current_hilt_end
                                or victim.current_emitter
                                or (frame_width / 2.0, frame_height / 2.0)
                            )

                            falling_sabers.append(
                                FallingSaber(
                                    hilt_pos=hilt_pos,
                                    initial_velocity=(knock_vx, knock_vy),
                                    initial_angle=init_angle,
                                    angular_velocity=rot_speed,
                                    color=victim.color,
                                    blade_length=blade_length,
                                    floor_y=floor_y,
                                    floor_width=float(frame_width),
                                )
                            )
                            victim.disarm(curr_time, duration=3.5)

                # Advance falling sabers and spark particles
                alive_falling: List[FallingSaber] = []
                for fs in falling_sabers:
                    fs.update(dt, particle_system)
                    if not fs.is_dead:
                        alive_falling.append(fs)
                falling_sabers = alive_falling

                particle_system.update(dt)

                # 1. Render physical hilts for active hand grips
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

                # Render tumbling falling sabers (hilts on frame, blades on light_canvas)
                for fs in falling_sabers:
                    fs.draw(frame, light_canvas, curr_time)

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

                # Render Newtonian sparks with thermal color decay
                particle_system.draw(light_canvas)

                # 3. Additive bloom and compositing with saturating addition
                composite_light_layer(frame, light_canvas, show_glow=show_glow)

                # 4. Optional Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, raw_observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Sparks: {len(particle_system.sparks)} | "
                        f"Falling: {len(falling_sabers)} | Active: {', '.join(active_sabers) or 'None'}"
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
                elif key in (ord("c"), ord("C")):  # 'c' to trigger test clash sparks
                    particle_system.spawn_clash_sparks(frame_width / 2.0, frame_height / 2.0, count=35)
                elif key in (ord("k"), ord("K")):  # 'k' to knock out / disarm Person2 for demo
                    victim = sabers["Person2"]
                    floor_y = frame_height - 20.0
                    init_angle = (
                        math.atan2(victim.current_direction[1], victim.current_direction[0])
                        if victim.current_direction
                        else -0.6
                    )
                    hilt_pos = (
                        victim.current_hilt_end
                        or victim.current_emitter
                        or (frame_width * 0.7, frame_height * 0.4)
                    )
                    falling_sabers.append(
                        FallingSaber(
                            hilt_pos=hilt_pos,
                            initial_velocity=(180.0, -260.0),
                            initial_angle=init_angle,
                            angular_velocity=8.0,
                            color=victim.color,
                            blade_length=blade_length,
                            floor_y=floor_y,
                            floor_width=float(frame_width),
                        )
                    )
                    victim.disarm(curr_time, duration=3.5)
                elif key in (ord("r"), ord("R")):  # 'r' to reset
                    for saber in sabers.values():
                        saber.reset()
                    recoil_controller.reset()
                    particle_system.clear()
                    falling_sabers.clear()

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
