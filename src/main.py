"""Main entry point for Lightsabers application."""

import math
import pathlib
import sys
import time
from typing import List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.camera import Camera
from src.collision import (
    BladeRecoilController,
    check_blade_collision,
    evaluate_duel_clash,
)
from src.config import AppConfig
from src.falling_saber import FallingSaber
from src.geometry import calculate_distance_scaled_blade_length
from src.hand_tracker import HandTracker, TwoPersonTracker, draw_hand_landmarks
from src.particles import ParticleSystem
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    draw_trail_on_light_canvas,
    render_hilt,
)
from src.saber import SaberInstance


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 3,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 6,
) -> None:
    """Draw centered text with a dark shadow/outline for maximum readability."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(10, (frame.shape[1] - text_w) // 2)
    # Dark outline
    cv2.putText(frame, text, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    # Main colored text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    config = AppConfig()

    print("=" * 65)
    print("      JEDI LIGHTSABERS — REAL-TIME COMPUTER VISION DUEL")
    print("=" * 65)
    print("Controls:")
    print("  'q' / ESC : Quit")
    print("  'g'       : Toggle glow effect")
    print("  't'       : Toggle motion trail")
    print("  'd'       : Toggle debug landmarks & HUD")
    print("  'c'       : Spawn test clash sparks")
    print("  'k'       : Knock out / disarm Person 2 (test demo)")
    print("  'r'       : Reset duel, tracking & physics states")
    print("=" * 65)

    window_name = "Jedi Lightsabers"
    prev_time = time.time()
    fps = 0.0

    show_glow = config.show_glow
    show_trail = config.show_trail
    show_debug = config.debug_mode

    # Pre-allocated canvas buffer to eliminate memory allocations and GC latency
    canvas_buffer = CanvasBuffer(config.camera.width, config.camera.height)

    # Two-Person Tracker with spatial hysteresis (prevents clash-proximity merging)
    person_tracker = TwoPersonTracker(
        same_person_max_distance_ratio=config.tracker.same_person_max_distance_ratio
    )

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

    # Duel state machine
    # States: "AWAITING_PLAYERS" -> "COUNTDOWN" -> "DUEL_ACTIVE" -> "ROUND_OVER"
    duel_state = "AWAITING_PLAYERS"
    countdown_time_left = config.duel.countdown_seconds
    active_banner_text: Optional[str] = None
    active_banner_color: Tuple[int, int, int] = (0, 255, 128)
    active_banner_expiry: float = 0.0
    was_colliding_last_frame: bool = False
    post_disarm_timer: float = 0.0

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

                # Compute frame delta time and FPS
                curr_time = time.time()
                dt = max(1e-4, min(curr_time - prev_time, 0.05))
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
                prev_time = curr_time

                # Detect hands
                raw_observations = tracker.detect(frame)

                # Filter so at most 2 distinct people are tracked (with spatial hysteresis)
                observations = person_tracker.update(raw_observations, curr_time)

                # Dynamic distance-based blade scaling computed per hand
                if len(observations) == 1:
                    obs = observations[0]
                    len1 = calculate_distance_scaled_blade_length(
                        obs.hand_size,
                        hand_to_blade_ratio=config.saber.hand_to_blade_ratio,
                        min_length=config.saber.min_blade_length,
                        max_length=config.saber.max_blade_length,
                    )
                    sabers["Person1"].update(
                        obs,
                        curr_time=curr_time,
                        blade_length=len1,
                        trail_duration=config.saber.trail_duration,
                    )
                    sabers["Person2"].update(
                        None,
                        curr_time=curr_time,
                        blade_length=config.saber.base_blade_length,
                        trail_duration=config.saber.trail_duration,
                    )
                elif len(observations) >= 2:
                    obs1, obs2 = observations[0], observations[1]
                    len1 = calculate_distance_scaled_blade_length(
                        obs1.hand_size,
                        hand_to_blade_ratio=config.saber.hand_to_blade_ratio,
                        min_length=config.saber.min_blade_length,
                        max_length=config.saber.max_blade_length,
                    )
                    len2 = calculate_distance_scaled_blade_length(
                        obs2.hand_size,
                        hand_to_blade_ratio=config.saber.hand_to_blade_ratio,
                        min_length=config.saber.min_blade_length,
                        max_length=config.saber.max_blade_length,
                    )
                    sabers["Person1"].update(
                        obs1,
                        curr_time=curr_time,
                        blade_length=len1,
                        trail_duration=config.saber.trail_duration,
                    )
                    sabers["Person2"].update(
                        obs2,
                        curr_time=curr_time,
                        blade_length=len2,
                        trail_duration=config.saber.trail_duration,
                    )
                else:
                    for saber in sabers.values():
                        saber.update(
                            None,
                            curr_time=curr_time,
                            blade_length=config.saber.base_blade_length,
                            trail_duration=config.saber.trail_duration,
                        )

                # Update elastic blade recoil physics
                recoil_controller.update(dt)

                # Apply recoil deflection to active sabers
                for slot_name, saber in sabers.items():
                    if saber.is_active:
                        angle = recoil_controller.get_recoil_angle(slot_name)
                        saber.apply_recoil(angle)

                # Duel state machine updates
                active_saber_count = sum(1 for s in sabers.values() if s.is_active)

                if duel_state == "AWAITING_PLAYERS":
                    if active_saber_count == 2:
                        duel_state = "COUNTDOWN"
                        countdown_time_left = config.duel.countdown_seconds
                elif duel_state == "COUNTDOWN":
                    if active_saber_count < 2:
                        # Fighter lowered hand during countdown
                        duel_state = "AWAITING_PLAYERS"
                    else:
                        countdown_time_left -= dt
                        if countdown_time_left <= 0.0:
                            duel_state = "DUEL_ACTIVE"
                            active_banner_text = "ENGAGE! DUEL ACTIVE"
                            active_banner_color = (0, 255, 255)
                            active_banner_expiry = curr_time + 1.2
                elif duel_state == "ROUND_OVER":
                    if (
                        curr_time >= post_disarm_timer
                        and not sabers["Person1"].is_disarmed
                        and not sabers["Person2"].is_disarmed
                    ):
                        if active_saber_count == 2:
                            duel_state = "COUNTDOWN"
                            countdown_time_left = 2.0  # Quick 2-second countdown for next round

                # Collision detection and Duel Combat mechanics
                s1 = sabers["Person1"]
                s2 = sabers["Person2"]
                is_currently_colliding = False

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
                        is_currently_colliding = True
                        is_new_clash = not was_colliding_last_frame

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

                        # 3. Evaluate duel combat mechanics (parry vs missed parry)
                        if s1.current_direction and s2.current_direction:
                            clash_eval = evaluate_duel_clash(
                                collision=collision,
                                s1_dir=s1.current_direction,
                                s2_dir=s2.current_direction,
                                s1_speed=s1.tip_speed,
                                s2_speed=s2.tip_speed,
                                is_new_clash=is_new_clash,
                                is_duel_active=(duel_state == "DUEL_ACTIVE"),
                                strike_min_speed=config.duel.strike_min_speed,
                                strike_speed_ratio=config.duel.strike_speed_ratio,
                                parry_min_angle_deg=config.duel.parry_min_angle_deg,
                                parry_max_foible_ratio=config.duel.parry_max_foible_ratio,
                                mutual_clash_min_speed=config.duel.mutual_clash_min_speed,
                                mutual_clash_ratio=config.duel.mutual_clash_ratio,
                            )

                            if clash_eval.banner_text:
                                active_banner_text = clash_eval.banner_text
                                active_banner_color = clash_eval.banner_color
                                active_banner_expiry = curr_time + 1.8

                            # Trigger disarm if a combatant missed a parry
                            if clash_eval.disarmed_slot is not None:
                                victim_slot = clash_eval.disarmed_slot
                                victim = sabers[victim_slot]
                                floor_y = frame_height - 20.0
                                knock_vx = 230.0 if victim_slot == "Person2" else -230.0
                                knock_vy = -270.0
                                rot_speed = 7.5 if victim_slot == "Person2" else -7.5
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
                                        blade_length=victim.current_blade_length,
                                        floor_y=floor_y,
                                        floor_width=float(frame_width),
                                    )
                                )
                                victim.disarm(curr_time, duration=3.5)
                                duel_state = "ROUND_OVER"
                                post_disarm_timer = curr_time + config.duel.post_disarm_cooldown

                was_colliding_last_frame = is_currently_colliding

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

                # Motion trails
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

                # 4. Cinematic Duel HUD & Countdown Display
                if duel_state == "AWAITING_PLAYERS":
                    status_str = f"AWAITING 2 JEDI... ({active_saber_count}/2 READY)"
                    draw_centered_text(frame, status_str, y=60, font_scale=0.85, color=(160, 200, 255))
                elif duel_state == "COUNTDOWN":
                    sec_num = int(math.ceil(countdown_time_left))
                    cd_text = f"DUEL IN: {sec_num}" if sec_num > 0 else "ENGAGE!"
                    draw_centered_text(frame, cd_text, y=100, font_scale=1.4, color=(0, 230, 255), thickness=4)
                elif duel_state == "DUEL_ACTIVE":
                    draw_centered_text(
                        frame,
                        "[ DUEL ACTIVE: PARRY OR BE DISARMED ]",
                        y=50,
                        font_scale=0.75,
                        color=(0, 255, 180),
                        thickness=2,
                    )
                elif duel_state == "ROUND_OVER":
                    draw_centered_text(
                        frame,
                        "ROUND OVER — DISARM RECORDED",
                        y=50,
                        font_scale=0.8,
                        color=(30, 100, 255),
                        thickness=2,
                    )

                # Active combat banner feedback (Parry / Disarm notification)
                if active_banner_text and curr_time < active_banner_expiry:
                    draw_centered_text(
                        frame,
                        active_banner_text,
                        y=150,
                        font_scale=1.0,
                        color=active_banner_color,
                        thickness=3,
                        outline_thickness=7,
                    )

                # 5. Optional Technical Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, raw_observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Duel: {duel_state} | Sabers: {len(active_sabers)}/2 | "
                        f"Sparks: {len(particle_system.sparks)} | S1 Spd: {s1.tip_speed:.0f} | S2 Spd: {s2.tip_speed:.0f}"
                    )
                    cv2.putText(
                        frame,
                        status_text,
                        (20, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
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
                            blade_length=victim.current_blade_length,
                            floor_y=floor_y,
                            floor_width=float(frame_width),
                        )
                    )
                    victim.disarm(curr_time, duration=3.5)
                    duel_state = "ROUND_OVER"
                    post_disarm_timer = curr_time + config.duel.post_disarm_cooldown
                elif key in (ord("r"), ord("R")):  # 'r' to reset
                    for saber in sabers.values():
                        saber.reset()
                    person_tracker.reset()
                    recoil_controller.reset()
                    particle_system.clear()
                    falling_sabers.clear()
                    duel_state = "AWAITING_PLAYERS"
                    countdown_time_left = config.duel.countdown_seconds
                    active_banner_text = None

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
