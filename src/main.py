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
    check_blade_collision,
    evaluate_duel_clash,
)
from src.config import AppConfig
from src.falling_saber import FallingSaber
from src.geometry import (
    calculate_distance_scaled_blade_length,
    calculate_perspective_scale_factor,
)
from src.gestures import is_force_push_pose, is_hand_in_box, is_two_finger_pose
from src.hand_tracker import HandTracker, TwoPersonTracker, draw_hand_landmarks
from src.particles import ParticleSystem
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    draw_trail_on_light_canvas,
    render_hilt,
    render_ignition_box,
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
    print("Gesture Recognition:")
    print("  Jedi Blue : Two-Finger Focus / Peace sign toward camera")
    print("  Sith Red  : Force Push / Open Palm toward camera")
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
            is_ignited=False,
        ),
        "Person2": SaberInstance(
            handedness="Person2",
            color=config.saber.sith_red,
            alpha_pivot=config.saber.alpha_pivot,
            alpha_direction=config.saber.alpha_direction,
            grace_period=config.saber.grace_period,
            is_ignited=False,
        ),
    }

    # Physical simulations: sparks, falling sabers
    particle_system = ParticleSystem(gravity=850.0, drag=0.93)
    falling_sabers: List[FallingSaber] = []

    # Duel state machine: "AWAITING_IGNITION" -> "COUNTDOWN" -> "DUEL_ACTIVE" -> "ROUND_OVER"
    duel_state = "AWAITING_IGNITION"
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

                # -------------------------------------------------------------
                # GESTURE RECOGNITION & HAND-TO-SABER BINDING (IGNITION BOX)
                # -------------------------------------------------------------
                s_blue = sabers["Person1"]
                s_red = sabers["Person2"]

                box_x1 = int(config.box.x_min * frame_width)
                box_y1 = int(config.box.y_min * frame_height)
                box_x2 = int(config.box.x_max * frame_width)
                box_y2 = int(config.box.y_max * frame_height)
                ignition_box_rect = (box_x1, box_y1, box_x2, box_y2)

                has_hand_in_box = False
                detected_box_gesture = None

                # Check unassigned hands for ignition gestures ONLY within the ignition box
                for obs in observations:
                    is_near_blue = (
                        s_blue.is_ignited
                        and s_blue.assigned_wrist_pos is not None
                        and math.hypot(obs.wrist[0] - s_blue.assigned_wrist_pos[0], obs.wrist[1] - s_blue.assigned_wrist_pos[1]) < 220.0
                    )
                    is_near_red = (
                        s_red.is_ignited
                        and s_red.assigned_wrist_pos is not None
                        and math.hypot(obs.wrist[0] - s_red.assigned_wrist_pos[0], obs.wrist[1] - s_red.assigned_wrist_pos[1]) < 220.0
                    )

                    # Hand is free to ignite a lightsaber
                    if not is_near_blue and not is_near_red:
                        # Strictly require hand to be inside the designated Ignition Box
                        if is_hand_in_box(obs.wrist, obs.knuckles_center, ignition_box_rect):
                            has_hand_in_box = True
                            lms = obs.landmarks_3d if obs.landmarks_3d else obs.landmarks

                            # 1. Two-Finger Pose -> Ignites Jedi Blue on THIS hand
                            if not s_blue.is_ignited and not s_blue.is_disarmed:
                                if is_two_finger_pose(lms):
                                    detected_box_gesture = "JediBlue"
                                    s_blue.ignite()
                                    s_blue.assigned_wrist_pos = obs.wrist
                                    particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=30)
                                    active_banner_text = "JEDI BLUE IGNITED!"
                                    active_banner_color = (255, 140, 0)
                                    active_banner_expiry = curr_time + 1.5

                            # 2. Force Push / Open Palm -> Ignites Sith Red on THIS hand
                            if not s_red.is_ignited and not s_red.is_disarmed:
                                if is_force_push_pose(lms):
                                    detected_box_gesture = "SithRed"
                                    s_red.ignite()
                                    s_red.assigned_wrist_pos = obs.wrist
                                    particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=30)
                                    active_banner_text = "SITH RED IGNITED!"
                                    active_banner_color = (30, 40, 255)
                                    active_banner_expiry = curr_time + 1.5

                # -------------------------------------------------------------
                # TRACK & UPDATE MATCHING HANDS FOR EACH SABER
                # -------------------------------------------------------------
                matched_blue_obs = None
                matched_red_obs = None

                if s_blue.is_ignited and s_blue.assigned_wrist_pos:
                    best_dist = 280.0
                    for obs in observations:
                        dist = math.hypot(obs.wrist[0] - s_blue.assigned_wrist_pos[0], obs.wrist[1] - s_blue.assigned_wrist_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            matched_blue_obs = obs

                if s_red.is_ignited and s_red.assigned_wrist_pos:
                    best_dist = 280.0
                    for obs in observations:
                        if obs is matched_blue_obs:
                            continue
                        dist = math.hypot(obs.wrist[0] - s_red.assigned_wrist_pos[0], obs.wrist[1] - s_red.assigned_wrist_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            matched_red_obs = obs

                # Update Blue Saber with dynamic distance scaling
                if matched_blue_obs is not None:
                    len_blue = calculate_distance_scaled_blade_length(
                        matched_blue_obs.hand_size,
                        base_length=config.saber.base_blade_length,
                        reference_hand_size=config.saber.reference_hand_size,
                        min_length=config.saber.min_blade_length,
                        max_length=config.saber.max_blade_length,
                    )
                    scale_blue = calculate_perspective_scale_factor(
                        matched_blue_obs.hand_size,
                        reference_hand_size=config.saber.reference_hand_size,
                    )
                    s_blue.update(
                        matched_blue_obs,
                        curr_time=curr_time,
                        blade_length=len_blue,
                        trail_duration=config.saber.trail_duration,
                        scale_factor=scale_blue,
                    )
                else:
                    s_blue.update(
                        None,
                        curr_time=curr_time,
                        blade_length=config.saber.base_blade_length,
                        trail_duration=config.saber.trail_duration,
                    )

                # Update Red Saber with dynamic distance scaling
                if matched_red_obs is not None:
                    len_red = calculate_distance_scaled_blade_length(
                        matched_red_obs.hand_size,
                        base_length=config.saber.base_blade_length,
                        reference_hand_size=config.saber.reference_hand_size,
                        min_length=config.saber.min_blade_length,
                        max_length=config.saber.max_blade_length,
                    )
                    scale_red = calculate_perspective_scale_factor(
                        matched_red_obs.hand_size,
                        reference_hand_size=config.saber.reference_hand_size,
                    )
                    s_red.update(
                        matched_red_obs,
                        curr_time=curr_time,
                        blade_length=len_red,
                        trail_duration=config.saber.trail_duration,
                        scale_factor=scale_red,
                    )
                else:
                    s_red.update(
                        None,
                        curr_time=curr_time,
                        blade_length=config.saber.base_blade_length,
                        trail_duration=config.saber.trail_duration,
                    )

                # -------------------------------------------------------------
                # DUEL STATE MACHINE UPDATES (COUNTDOWN -> DUEL_ACTIVE)
                # -------------------------------------------------------------
                active_saber_count = sum(1 for s in sabers.values() if s.is_active and s.is_ignited)

                if duel_state == "AWAITING_IGNITION":
                    if active_saber_count == 2:
                        duel_state = "COUNTDOWN"
                        countdown_time_left = config.duel.countdown_seconds
                elif duel_state == "COUNTDOWN":
                    if active_saber_count < 2:
                        # Combatant lowered hand or extinguished blade
                        duel_state = "AWAITING_IGNITION"
                    else:
                        countdown_time_left -= dt
                        if countdown_time_left <= 0.0:
                            duel_state = "DUEL_ACTIVE"
                            active_banner_text = "ENGAGE! DUEL ACTIVE"
                            active_banner_color = (0, 255, 255)
                            active_banner_expiry = curr_time + 1.2
                elif duel_state == "DUEL_ACTIVE":
                    if active_saber_count < 2 and not s_blue.is_disarmed and not s_red.is_disarmed:
                        duel_state = "AWAITING_IGNITION"
                elif duel_state == "ROUND_OVER":
                    if (
                        curr_time >= post_disarm_timer
                        and not s_blue.is_disarmed
                        and not s_red.is_disarmed
                    ):
                        if active_saber_count == 2:
                            duel_state = "COUNTDOWN"
                            countdown_time_left = 2.0  # Quick 2-second countdown for next round
                        else:
                            duel_state = "AWAITING_IGNITION"

                # -------------------------------------------------------------
                # BLADE COLLISION & PARRY COMBAT EVALUATION
                # -------------------------------------------------------------
                is_currently_colliding = False

                if (
                    s_blue.is_active
                    and s_red.is_active
                    and s_blue.current_emitter
                    and s_blue.current_endpoint
                    and s_red.current_emitter
                    and s_red.current_endpoint
                ):
                    collision = check_blade_collision(
                        s_blue.current_emitter,
                        s_blue.current_endpoint,
                        s_red.current_emitter,
                        s_red.current_endpoint,
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

                        # 2. Evaluate duel combat mechanics (parry vs missed parry)
                        if s_blue.current_direction and s_red.current_direction:
                            clash_eval = evaluate_duel_clash(
                                collision=collision,
                                s1_dir=s_blue.current_direction,
                                s2_dir=s_red.current_direction,
                                s1_speed=s_blue.tip_speed,
                                s2_speed=s_red.tip_speed,
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

                # 1. Render physical hilts for active hand grips (with dynamic perspective scaling)
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
                            scale_factor=saber.scale_factor,
                        )

                # 2. Reusable pre-allocated light canvas (zero-allocation)
                light_canvas = canvas_buffer.reset_and_get(frame.shape)

                # Render tumbling falling sabers (hilts on frame, blades on light_canvas)
                for fs in falling_sabers:
                    fs.draw(frame, light_canvas, curr_time)

                # Motion trails
                if show_trail:
                    for saber in sabers.values():
                        if len(saber.trail_history) >= 2 and saber.is_active:
                            draw_trail_on_light_canvas(
                                light_canvas,
                                saber.trail_history,
                                saber.color,
                                config.saber.trail_duration,
                                curr_time,
                            )

                # Luminous blades extending outward from hands (with dynamic perspective scaling)
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
                            scale_factor=saber.scale_factor,
                        )

                # Render Newtonian sparks with thermal color decay
                particle_system.draw(light_canvas)

                # Render holographic on-screen Ignition Chamber (completely removed after both ignite)
                if not (s_blue.is_ignited and s_red.is_ignited):
                    render_ignition_box(
                        frame,
                        light_canvas,
                        ignition_box_rect,
                        has_hand_inside=has_hand_in_box,
                        detected_gesture=detected_box_gesture,
                        is_blue_ignited=s_blue.is_ignited,
                        is_red_ignited=s_red.is_ignited,
                        curr_time=curr_time,
                    )

                # 3. Additive bloom and compositing with saturating addition
                composite_light_layer(frame, light_canvas, show_glow=show_glow)

                # 4. Duel HUD & Countdown Display
                if duel_state == "AWAITING_IGNITION":
                    status_str = f"AWAITING JEDI & SITH IGNITION ({active_saber_count}/2 ACTIVE)"
                    draw_centered_text(frame, status_str, y=50, font_scale=0.75, color=(160, 200, 255))
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
                        "ROUND OVER — PLACE HAND IN BOX TO RE-IGNITE",
                        y=50,
                        font_scale=0.75,
                        color=(30, 100, 255),
                        thickness=2,
                    )

                # Active combat banner feedback (Ignition / Parry / Disarm notification)
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

                # 6. Optional Technical Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, raw_observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Duel: {duel_state} | Sabers: {len(active_sabers)}/2 | "
                        f"Sparks: {len(particle_system.sparks)} | Blue Spd: {s_blue.tip_speed:.0f} | Red Spd: {s_red.tip_speed:.0f}"
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
                    if victim.is_active:
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
                    particle_system.clear()
                    falling_sabers.clear()
                    duel_state = "AWAITING_IGNITION"
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
