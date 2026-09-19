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
    render_match_winner_screen,
    render_side_chamber,
    render_tutorial_card,
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
    print("  SPACE / N : Next tutorial card")
    print("  's'       : Skip tutorial")
    print("  'h'       : Open/reopen tutorial cards")
    print("  'g'       : Toggle glow effect")
    print("  't'       : Toggle motion trail")
    print("  'd'       : Toggle debug landmarks & HUD")
    print("  'c'       : Spawn test clash sparks")
    print("  'k' / 'j' : Knock out Sith / Jedi (demo disarm)")
    print("  'r'       : Reset match & scores")
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

    # Duel state machine: "TUTORIAL" -> "AWAITING_IGNITION" -> "COUNTDOWN" -> "DUEL_ACTIVE" -> "ROUND_OVER" -> "MATCH_OVER"
    duel_state = "TUTORIAL" if config.duel.enable_tutorial else "AWAITING_IGNITION"
    tutorial_slide: int = 1
    countdown_time_left = config.duel.countdown_seconds
    current_round: int = 1
    score_blue: int = 0
    score_red: int = 0
    match_winner_name: Optional[str] = None
    match_winner_color: Tuple[int, int, int] = config.saber.jedi_blue
    active_banner_text: Optional[str] = None
    active_banner_color: Tuple[int, int, int] = (0, 255, 128)
    was_colliding_last_frame: bool = False
    post_disarm_timer: float = 0.0

    # Combat engagement: Guard Poise, Rally Streaks, and Active Deflections
    poise_blue: int = 2
    poise_red: int = 2
    rally_streak: int = 0
    last_clash_time: float = 0.0
    clash_duration: float = 0.0
    counter_strike_slot: Optional[str] = None
    counter_strike_expiry: float = 0.0

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

                # Compute Left (Jedi) and Right (Sith) ignition chamber bounds
                left_box_rect = (
                    int(config.box.left_x_min * frame_width),
                    int(config.box.left_y_min * frame_height),
                    int(config.box.left_x_max * frame_width),
                    int(config.box.left_y_max * frame_height),
                )
                right_box_rect = (
                    int(config.box.right_x_min * frame_width),
                    int(config.box.right_y_min * frame_height),
                    int(config.box.right_x_max * frame_width),
                    int(config.box.right_y_max * frame_height),
                )

                has_hand_in_left = False
                has_hand_in_right = False

                # Check unassigned hands for ignition gestures within their respective chambers
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
                        # 1. Left Chamber -> Two-Finger Focus ignites Jedi Blue
                        if not s_blue.is_ignited and not s_blue.is_disarmed:
                            if is_hand_in_box(obs.wrist, obs.knuckles_center, left_box_rect, palm_center=obs.palm_center):
                                has_hand_in_left = True
                                is_jedi = is_two_finger_pose(obs.landmarks_3d) or is_two_finger_pose(obs.landmarks)
                                if is_jedi:
                                    s_blue.ignite()
                                    s_blue.assigned_wrist_pos = obs.wrist
                                    particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=30)
                                    active_banner_text = "JEDI BLUE IGNITED!"
                                    active_banner_color = (255, 140, 0)
                                    active_banner_expiry = curr_time + 1.5

                        # 2. Right Chamber -> Force Push ignites Sith Red
                        if not s_red.is_ignited and not s_red.is_disarmed:
                            if is_hand_in_box(obs.wrist, obs.knuckles_center, right_box_rect, palm_center=obs.palm_center):
                                has_hand_in_right = True
                                is_sith = is_force_push_pose(obs.landmarks_3d) or is_force_push_pose(obs.landmarks)
                                if is_sith:
                                    s_red.ignite()
                                    s_red.assigned_wrist_pos = obs.wrist
                                    particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=30)
                                    active_banner_text = "SITH RED IGNITED!"
                                    active_banner_color = (30, 40, 255)
                                    active_banner_expiry = curr_time + 1.5

                # -------------------------------------------------------------
                # TRACK & UPDATE MATCHING HANDS FOR EACH SABER (SPATIAL PARTITION)
                # -------------------------------------------------------------
                matched_blue_obs = None
                matched_red_obs = None

                # Spatial hand assignment: Person 1 (Blue) is on the left; Person 2 (Red) is on the right.
                # When hands leave the frame and re-enter, spatial matching ensures immediate re-acquisition
                # without any arbitrary distance clamps.
                if len(observations) >= 2:
                    sorted_obs = sorted(observations, key=lambda o: o.wrist[0])
                    obs_left = sorted_obs[0]
                    obs_right = sorted_obs[-1]

                    if s_blue.is_ignited and not s_blue.is_disarmed:
                        matched_blue_obs = obs_left
                        s_blue.assigned_wrist_pos = obs_left.wrist

                    if s_red.is_ignited and not s_red.is_disarmed:
                        matched_red_obs = obs_right
                        s_red.assigned_wrist_pos = obs_right.wrist

                elif len(observations) == 1:
                    single_obs = observations[0]
                    mid_x = frame_width * 0.50

                    if s_blue.is_ignited and not s_red.is_ignited and not s_blue.is_disarmed:
                        matched_blue_obs = single_obs
                        s_blue.assigned_wrist_pos = single_obs.wrist
                    elif s_red.is_ignited and not s_blue.is_ignited and not s_red.is_disarmed:
                        matched_red_obs = single_obs
                        s_red.assigned_wrist_pos = single_obs.wrist
                    elif s_blue.is_ignited and s_red.is_ignited:
                        # Both are ignited - match based on disarm status, distance, or screen half
                        if s_blue.is_disarmed:
                            matched_red_obs = single_obs
                            s_red.assigned_wrist_pos = single_obs.wrist
                        elif s_red.is_disarmed:
                            matched_blue_obs = single_obs
                            s_blue.assigned_wrist_pos = single_obs.wrist
                        elif s_blue.assigned_wrist_pos and s_red.assigned_wrist_pos:
                            d_blue = math.hypot(single_obs.wrist[0] - s_blue.assigned_wrist_pos[0], single_obs.wrist[1] - s_blue.assigned_wrist_pos[1])
                            d_red = math.hypot(single_obs.wrist[0] - s_red.assigned_wrist_pos[0], single_obs.wrist[1] - s_red.assigned_wrist_pos[1])
                            if d_blue < d_red:
                                matched_blue_obs = single_obs
                                s_blue.assigned_wrist_pos = single_obs.wrist
                            else:
                                matched_red_obs = single_obs
                                s_red.assigned_wrist_pos = single_obs.wrist
                        elif single_obs.wrist[0] < mid_x:
                            matched_blue_obs = single_obs
                            s_blue.assigned_wrist_pos = single_obs.wrist
                        else:
                            matched_red_obs = single_obs
                            s_red.assigned_wrist_pos = single_obs.wrist

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
                    if s_blue.is_ignited and s_red.is_ignited:
                        duel_state = "COUNTDOWN"
                        countdown_time_left = config.duel.countdown_seconds
                elif duel_state == "COUNTDOWN":
                    # Monotonic countdown: do NOT cancel back to AWAITING_IGNITION if a hand moves off-screen!
                    countdown_time_left -= dt
                    if countdown_time_left <= 0.0:
                        duel_state = "DUEL_ACTIVE"
                        active_banner_text = "ENGAGE! DUEL ACTIVE"
                        active_banner_color = (0, 255, 255)
                        active_banner_expiry = curr_time + 1.2
                elif duel_state == "DUEL_ACTIVE":
                    # Rally streak cools down after 3.0s of inactivity
                    if curr_time - last_clash_time > 3.0:
                        rally_streak = 0
                elif duel_state == "ROUND_OVER":
                    if curr_time >= post_disarm_timer:
                        # Check match victory condition: first to wins_to_win (2) or completed max_rounds (3)
                        if score_blue >= config.duel.wins_to_win:
                            duel_state = "MATCH_OVER"
                            match_winner_name = "Jedi Blue"
                            match_winner_color = config.saber.jedi_blue
                        elif score_red >= config.duel.wins_to_win:
                            duel_state = "MATCH_OVER"
                            match_winner_name = "Sith Red"
                            match_winner_color = config.saber.sith_red
                        elif current_round >= config.duel.max_rounds:
                            duel_state = "MATCH_OVER"
                            if score_blue > score_red:
                                match_winner_name = "Jedi Blue"
                                match_winner_color = config.saber.jedi_blue
                            elif score_red > score_blue:
                                match_winner_name = "Sith Red"
                                match_winner_color = config.saber.sith_red
                            else:
                                match_winner_name = "Draw"
                                match_winner_color = (255, 255, 0)
                        else:
                            # Advance to next round - both sabers extinguish for fresh re-ignition!
                            current_round += 1
                            s_blue.reset()
                            s_red.reset()
                            poise_blue = 2
                            poise_red = 2
                            rally_streak = 0
                            clash_duration = 0.0
                            counter_strike_slot = None
                            counter_strike_expiry = 0.0
                            duel_state = "AWAITING_IGNITION"
                            countdown_time_left = config.duel.countdown_seconds
                            active_banner_text = f"ROUND {current_round} - RE-IGNITE SABERS!"
                            active_banner_color = (0, 255, 255)
                            active_banner_expiry = curr_time + 2.5

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

                        if is_new_clash:
                            clash_duration = 0.0
                            last_clash_time = curr_time
                            rally_streak += 1
                        else:
                            clash_duration += dt

                        # Continuous electrical crackle sparks during sustained blade bind (Saber Lock)
                        if clash_duration > 0.22 and duel_state == "DUEL_ACTIVE":
                            particle_system.spawn_clash_sparks(
                                collision.clash_point[0],
                                collision.clash_point[1],
                                collision.normal[0],
                                collision.normal[1],
                                count=8,
                                speed_min=80.0,
                                speed_max=220.0,
                            )

                        # Evaluate duel combat mechanics
                        if s_blue.current_direction and s_red.current_direction:
                            # Apply counter-strike advantage (+35% speed boost for 1.2s after a perfect parry)
                            v1 = s_blue.tip_speed * (1.35 if counter_strike_slot == "Person1" and curr_time < counter_strike_expiry else 1.0)
                            v2 = s_red.tip_speed * (1.35 if counter_strike_slot == "Person2" and curr_time < counter_strike_expiry else 1.0)

                            clash_eval = evaluate_duel_clash(
                                collision=collision,
                                s1_dir=s_blue.current_direction,
                                s2_dir=s_red.current_direction,
                                s1_speed=v1,
                                s2_speed=v2,
                                is_new_clash=is_new_clash,
                                is_duel_active=(duel_state == "DUEL_ACTIVE"),
                                s1_poise=poise_blue,
                                s2_poise=poise_red,
                                strike_min_speed=config.duel.strike_min_speed,
                                strike_speed_ratio=config.duel.strike_speed_ratio,
                                parry_min_angle_deg=config.duel.parry_min_angle_deg,
                                parry_max_foible_ratio=config.duel.parry_max_foible_ratio,
                                mutual_clash_min_speed=config.duel.mutual_clash_min_speed,
                                mutual_clash_ratio=config.duel.mutual_clash_ratio,
                            )

                            # Spawn appropriate sparks and plasma shockwaves
                            if is_new_clash:
                                if clash_eval.shockwave_type == "perfect":
                                    particle_system.spawn_shockwave(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        max_radius=190.0,
                                        speed=520.0,
                                        lifetime=0.40,
                                        color=(255, 255, 120),
                                        thickness=5,
                                    )
                                    particle_system.spawn_clash_sparks(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        collision.normal[0],
                                        collision.normal[1],
                                        count=48,
                                        speed_min=240.0,
                                        speed_max=580.0,
                                    )
                                elif clash_eval.shockwave_type == "heavy":
                                    particle_system.spawn_shockwave(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        max_radius=145.0,
                                        speed=440.0,
                                        lifetime=0.34,
                                        color=(100, 200, 255),
                                        thickness=4,
                                    )
                                    particle_system.spawn_clash_sparks(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        collision.normal[0],
                                        collision.normal[1],
                                        count=36,
                                        speed_min=200.0,
                                        speed_max=480.0,
                                    )
                                else:
                                    particle_system.spawn_shockwave(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        max_radius=95.0,
                                        speed=360.0,
                                        lifetime=0.25,
                                        color=(220, 240, 255),
                                        thickness=2,
                                    )
                                    particle_system.spawn_clash_sparks(
                                        collision.clash_point[0],
                                        collision.clash_point[1],
                                        collision.normal[0],
                                        collision.normal[1],
                                        count=24,
                                        speed_min=160.0,
                                        speed_max=400.0,
                                    )

                            # 1. Handle Perfect Parry
                            if clash_eval.is_perfect_parry:
                                if clash_eval.banner_text and "Person1" in clash_eval.banner_text:
                                    poise_blue = min(2, poise_blue + 1)
                                    counter_strike_slot = "Person1"
                                    active_banner_text = ">> JEDI PERFECT PARRY! (COUNTER READY) <<"
                                    active_banner_color = (255, 220, 0)
                                else:
                                    poise_red = min(2, poise_red + 1)
                                    counter_strike_slot = "Person2"
                                    active_banner_text = ">> SITH PERFECT PARRY! (COUNTER READY) <<"
                                    active_banner_color = (255, 220, 0)
                                counter_strike_expiry = curr_time + 1.2
                                active_banner_expiry = curr_time + 1.5

                            # 2. Handle Guard Poise Damage (Guard Shaken)
                            elif clash_eval.poise_damaged_slot is not None:
                                if clash_eval.poise_damaged_slot == "Person1":
                                    poise_blue = max(1, poise_blue - 1)
                                    active_banner_text = "! JEDI GUARD SHAKEN! [1 POISE LEFT] !"
                                else:
                                    poise_red = max(1, poise_red - 1)
                                    active_banner_text = "! SITH GUARD SHAKEN! [1 POISE LEFT] !"
                                active_banner_color = (0, 165, 255)
                                active_banner_expiry = curr_time + 1.6

                            # 3. Handle Disarm (Guard Broken -> Round Win)
                            elif clash_eval.disarmed_slot is not None:
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
                                rally_streak = 0
                                if victim_slot == "Person2":
                                    score_blue += 1
                                    active_banner_text = f">>> SITH GUARD BROKEN! JEDI WINS ROUND {current_round}! <<<"
                                    active_banner_color = config.saber.jedi_blue
                                else:
                                    score_red += 1
                                    active_banner_text = f">>> JEDI GUARD BROKEN! SITH WINS ROUND {current_round}! <<<"
                                    active_banner_color = config.saber.sith_red
                                active_banner_expiry = curr_time + 2.8
                                duel_state = "ROUND_OVER"
                                post_disarm_timer = curr_time + config.duel.post_disarm_cooldown

                            # 4. Handle Rally Streaks and other banners
                            elif is_new_clash:
                                if clash_eval.is_mutual:
                                    active_banner_text = "== MUTUAL CLASH! =="
                                    active_banner_color = (0, 255, 255)
                                    active_banner_expiry = curr_time + 1.2
                                elif rally_streak >= 2:
                                    if rally_streak >= 4:
                                        active_banner_text = f"*** RALLY x{rally_streak}! EPIC CLASH! ***"
                                        active_banner_color = (0, 215, 255)
                                    else:
                                        active_banner_text = f">> RALLY x{rally_streak}! <<"
                                        active_banner_color = (0, 255, 255)
                                    active_banner_expiry = curr_time + 1.1
                                elif clash_eval.banner_text:
                                    active_banner_text = clash_eval.banner_text
                                    active_banner_color = clash_eval.banner_color
                                    active_banner_expiry = curr_time + 1.3

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

                # Render holographic on-screen Ignition Chambers (disappear independently once ignited)
                if duel_state != "MATCH_OVER":
                    if not s_blue.is_ignited:
                        render_side_chamber(
                            frame,
                            light_canvas,
                            left_box_rect,
                            title="[ JEDI BLUE CHAMBER ]",
                            prompt="TWO-FINGER FOCUS TO IGNITE",
                            border_color=(255, 180, 50),
                            glow_color=config.saber.jedi_blue,
                            has_hand_inside=has_hand_in_left,
                            curr_time=curr_time,
                        )

                    if not s_red.is_ignited:
                        render_side_chamber(
                            frame,
                            light_canvas,
                            right_box_rect,
                            title="[ SITH RED CHAMBER ]",
                            prompt="FORCE PUSH / PALM TO IGNITE",
                            border_color=(50, 50, 255),
                            glow_color=config.saber.sith_red,
                            has_hand_inside=has_hand_in_right,
                            curr_time=curr_time,
                        )

                # Render holographic tutorial briefing cards (Cards 1, 2, 3)
                if duel_state == "TUTORIAL":
                    render_tutorial_card(
                        frame,
                        light_canvas,
                        slide_index=tutorial_slide,
                        curr_time=curr_time,
                    )

                # Render dramatic victory screen when match is over (stays indefinitely until 'R' or 'Q')
                if duel_state == "MATCH_OVER":
                    render_match_winner_screen(
                        frame,
                        light_canvas,
                        winner_name=match_winner_name or "Jedi Blue",
                        winner_color=match_winner_color,
                        score_blue=score_blue,
                        score_red=score_red,
                        curr_time=curr_time,
                    )

                # 3. Additive bloom and compositing with saturating addition
                composite_light_layer(frame, light_canvas, show_glow=show_glow)

                # 4. Duel HUD & Countdown Display
                if duel_state not in ("MATCH_OVER", "TUTORIAL"):
                    # Match Scoreboard
                    scoreboard_str = f"ROUND {current_round}/{config.duel.max_rounds}   [ JEDI BLUE: {score_blue}  |  SITH RED: {score_red} ]"
                    draw_centered_text(frame, scoreboard_str, y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)

                    # Guard Poise Status Bar
                    if s_blue.is_ignited or s_red.is_ignited:
                        blue_sh = "+ " * poise_blue + "- " * (2 - poise_blue)
                        red_sh = "+ " * poise_red + "- " * (2 - poise_red)
                        poise_hud = f"JEDI GUARD: [ {blue_sh.strip()} ]         SITH GUARD: [ {red_sh.strip()} ]"
                        draw_centered_text(frame, poise_hud, y=60, font_scale=0.55, color=(0, 230, 255), thickness=2)

                    if duel_state == "AWAITING_IGNITION":
                        status_str = f"AWAITING JEDI & SITH IGNITION ({active_saber_count}/2 ACTIVE)"
                        draw_centered_text(frame, status_str, y=88, font_scale=0.52, color=(160, 200, 255), thickness=2)
                    elif duel_state == "COUNTDOWN":
                        sec_num = int(math.ceil(countdown_time_left))
                        cd_text = f"DUEL IN: {sec_num}" if sec_num > 0 else "ENGAGE!"
                        draw_centered_text(frame, cd_text, y=115, font_scale=1.4, color=(0, 230, 255), thickness=4)
                    elif duel_state == "DUEL_ACTIVE":
                        if clash_duration > 0.22:
                            draw_centered_text(
                                frame,
                                ">> [ SABER LOCK! PUSH OFF! ] <<",
                                y=88,
                                font_scale=0.58,
                                color=(0, 255, 255),
                                thickness=2,
                            )
                        elif counter_strike_slot and curr_time < counter_strike_expiry:
                            c_name = "JEDI" if counter_strike_slot == "Person1" else "SITH"
                            draw_centered_text(
                                frame,
                                f">> {c_name} COUNTER-STRIKE READY! (+35% SPEED) <<",
                                y=88,
                                font_scale=0.55,
                                color=(255, 220, 0),
                                thickness=2,
                            )
                        else:
                            draw_centered_text(
                                frame,
                                "[ DUEL ACTIVE: PARRY TO PRESERVE GUARD POISE ]",
                                y=88,
                                font_scale=0.52,
                                color=(0, 255, 180),
                                thickness=2,
                            )
                    elif duel_state == "ROUND_OVER":
                        draw_centered_text(
                            frame,
                            f"ROUND {current_round} OVER - PREPARING NEXT ROUND...",
                            y=88,
                            font_scale=0.56,
                            color=(30, 100, 255),
                            thickness=2,
                        )

                    # Active combat banner feedback (Ignition / Parry / Disarm notification)
                    if active_banner_text and curr_time < active_banner_expiry:
                        draw_centered_text(
                            frame,
                            active_banner_text,
                            y=155,
                            font_scale=0.95,
                            color=active_banner_color,
                            thickness=3,
                            outline_thickness=7,
                        )

                # 6. Optional Technical Debug Overlay
                if show_debug:
                    draw_hand_landmarks(frame, raw_observations)
                    active_sabers = [k for k, s in sabers.items() if s.is_active]
                    status_text = (
                        f"FPS: {fps:.1f} | Duel: {duel_state} | Round: {current_round}/3 | "
                        f"Blue: {score_blue} | Red: {score_red} | Sabers: {len(active_sabers)}/2"
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
                elif key in (ord("k"), ord("K")):  # 'k' to knock out / disarm Person2 (Sith Red) for demo
                    victim = sabers["Person2"]
                    if victim.is_active and duel_state != "MATCH_OVER":
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
                        score_blue += 1
                        active_banner_text = f"JEDI WINS ROUND {current_round}!"
                        active_banner_color = config.saber.jedi_blue
                        active_banner_expiry = curr_time + 2.8
                        duel_state = "ROUND_OVER"
                        post_disarm_timer = curr_time + config.duel.post_disarm_cooldown
                elif key in (ord("j"), ord("J")):  # 'j' to disarm Person1 (Jedi Blue) for demo
                    victim = sabers["Person1"]
                    if victim.is_active and duel_state != "MATCH_OVER":
                        floor_y = frame_height - 20.0
                        init_angle = (
                            math.atan2(victim.current_direction[1], victim.current_direction[0])
                            if victim.current_direction
                            else 0.6
                        )
                        hilt_pos = (
                            victim.current_hilt_end
                            or victim.current_emitter
                            or (frame_width * 0.3, frame_height * 0.4)
                        )
                        falling_sabers.append(
                            FallingSaber(
                                hilt_pos=hilt_pos,
                                initial_velocity=(-180.0, -260.0),
                                initial_angle=init_angle,
                                angular_velocity=-8.0,
                                color=victim.color,
                                blade_length=victim.current_blade_length,
                                floor_y=floor_y,
                                floor_width=float(frame_width),
                            )
                        )
                        victim.disarm(curr_time, duration=3.5)
                        score_red += 1
                        active_banner_text = f"SITH WINS ROUND {current_round}!"
                        active_banner_color = config.saber.sith_red
                        active_banner_expiry = curr_time + 2.8
                        duel_state = "ROUND_OVER"
                        post_disarm_timer = curr_time + config.duel.post_disarm_cooldown
                elif key in (ord("r"), ord("R")):  # 'r' to reset / rematch
                    for saber in sabers.values():
                        saber.reset()
                    person_tracker.reset()
                    particle_system.clear()
                    falling_sabers.clear()
                    current_round = 1
                    score_blue = 0
                    score_red = 0
                    poise_blue = 2
                    poise_red = 2
                    rally_streak = 0
                    clash_duration = 0.0
                    counter_strike_slot = None
                    counter_strike_expiry = 0.0
                    match_winner_name = None
                    duel_state = "AWAITING_IGNITION"
                    countdown_time_left = config.duel.countdown_seconds
                    active_banner_text = "MATCH RESET - IGNITE SABERS!"
                    active_banner_color = (0, 255, 255)
                    active_banner_expiry = curr_time + 2.0
                elif key in (ord("s"), ord("S")):  # 's' to skip tutorial
                    if duel_state == "TUTORIAL":
                        duel_state = "AWAITING_IGNITION"
                        active_banner_text = "TUTORIAL SKIPPED - IGNITE SABERS!"
                        active_banner_color = (0, 255, 255)
                        active_banner_expiry = curr_time + 1.8
                elif key in (32, 13, ord("n"), ord("N")):  # SPACE / ENTER / 'n' to advance tutorial slide
                    if duel_state == "TUTORIAL":
                        tutorial_slide += 1
                        if tutorial_slide > 3:
                            duel_state = "AWAITING_IGNITION"
                            active_banner_text = "BRIEFING COMPLETE - IGNITE SABERS!"
                            active_banner_color = (0, 255, 255)
                            active_banner_expiry = curr_time + 2.0
                elif key in (ord("h"), ord("H")):  # 'h' to open/reopen tutorial cards
                    duel_state = "TUTORIAL"
                    tutorial_slide = 1

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
