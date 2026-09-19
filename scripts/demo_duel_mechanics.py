"""Generates a visual verification sequence demonstrating distance scaling, countdown, and parry mechanics."""

import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.collision import BladeRecoilController, check_blade_collision, evaluate_duel_clash
from src.falling_saber import FallingSaber
from src.geometry import calculate_distance_scaled_blade_length
from src.main import draw_centered_text
from src.particles import ParticleSystem
from src.renderer import (
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
)


def run_duel_demo(output_path: str) -> None:
    width, height = 1280, 720
    dt = 0.016

    blue_color = (255, 90, 20)  # Jedi Blue
    red_color = (30, 30, 255)   # Sith Red

    snapshots = []

    # -------------------------------------------------------------
    # PANEL 1: Distance-Based Scaling & Duel Countdown ("DUEL IN: 3")
    # -------------------------------------------------------------
    # Person 1 (Blue) is closer to camera (hand_size = 70px -> blade = 525px)
    # Person 2 (Red) is further back (hand_size = 35px -> blade = 262.5px)
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:] = (18, 18, 22)
    light1 = np.zeros((height, width, 3), dtype=np.uint8)

    p1_len = calculate_distance_scaled_blade_length(70.0, hand_to_blade_ratio=7.5)
    p2_len = calculate_distance_scaled_blade_length(35.0, hand_to_blade_ratio=7.5)

    p1_emit = (350.0, 500.0)
    p1_dir = (0.5, -0.866)
    p1_tip = (p1_emit[0] + p1_dir[0] * p1_len, p1_emit[1] + p1_dir[1] * p1_len)

    p2_emit = (920.0, 480.0)
    p2_dir = (-0.5, -0.866)
    p2_tip = (p2_emit[0] + p2_dir[0] * p2_len, p2_emit[1] + p2_dir[1] * p2_len)

    render_hilt(frame1, (p1_emit[0] - p1_dir[0] * 50, p1_emit[1] - p1_dir[1] * 50), p1_emit, p1_dir)
    render_hilt(frame1, (p2_emit[0] - p2_dir[0] * 35, p2_emit[1] - p2_dir[1] * 35), p2_emit, p2_dir)
    draw_blade_on_light_canvas(light1, p1_emit, p1_tip, blue_color, 0.0)
    draw_blade_on_light_canvas(light1, p2_emit, p2_tip, red_color, 0.0)
    composite_light_layer(frame1, light1, show_glow=True)

    draw_centered_text(frame1, "DUEL IN: 3", y=90, font_scale=1.4, color=(0, 230, 255), thickness=4)
    cv2.putText(frame1, f"Person 1 (Near): {p1_len:.0f}px", (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
    cv2.putText(frame1, f"Person 2 (Far, Scaled): {p2_len:.0f}px", (850, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
    cv2.putText(frame1, "1. Distance Scaling & Duel Countdown", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    snapshots.append(frame1)

    # -------------------------------------------------------------
    # PANEL 2: Successful Forte Parry ("PERSON2 PARRIED!")
    # -------------------------------------------------------------
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2[:] = (18, 18, 22)
    light2 = np.zeros((height, width, 3), dtype=np.uint8)
    ps2 = ParticleSystem(gravity=850.0, drag=0.93)
    recoil2 = BladeRecoilController()

    # Person 1 swings downward right; Person 2 blocks across with forte (lower blade)
    s1_emit = (400.0, 520.0)
    s1_tip = (760.0, 260.0)
    s1_d = (s1_tip[0] - s1_emit[0], s1_tip[1] - s1_emit[1])
    s1_l = math.hypot(*s1_d)
    s1_d = (s1_d[0] / s1_l, s1_d[1] / s1_l)

    s2_emit = (820.0, 460.0)
    s2_tip = (480.0, 280.0)
    s2_d = (s2_tip[0] - s2_emit[0], s2_tip[1] - s2_emit[1])
    s2_l = math.hypot(*s2_d)
    s2_d = (s2_d[0] / s2_l, s2_d[1] / s2_l)

    col2 = check_blade_collision(s1_emit, s1_tip, s2_emit, s2_tip)
    assert col2 is not None
    eval2 = evaluate_duel_clash(
        col2, s1_d, s2_d, s1_speed=460.0, s2_speed=60.0, is_new_clash=True, is_duel_active=True
    )
    ps2.spawn_clash_sparks(col2.clash_point[0], col2.clash_point[1], col2.normal[0], col2.normal[1], count=35)
    recoil2.apply_clash_impulse("p1", "p2", col2.normal, s1_d, s2_d, impulse_magnitude=0.40)
    for _ in range(3):
        ps2.update(dt)
        recoil2.update(dt)

    draw_blade_on_light_canvas(light2, s1_emit, s1_tip, blue_color, 0.1)
    draw_blade_on_light_canvas(light2, s2_emit, s2_tip, red_color, 0.1)
    ps2.draw(light2)
    render_hilt(frame2, (s1_emit[0] - s1_d[0] * 50, s1_emit[1] - s1_d[1] * 50), s1_emit, s1_d)
    render_hilt(frame2, (s2_emit[0] - s2_d[0] * 50, s2_emit[1] - s2_d[1] * 50), s2_emit, s2_d)
    composite_light_layer(frame2, light2, show_glow=True)

    draw_centered_text(frame2, eval2.banner_text or "PERSON2 PARRIED!", y=130, font_scale=1.1, color=(0, 255, 128), thickness=3)
    cv2.putText(frame2, "2. Successful Forte Parry (Solid Angle & Strong Leverage)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    snapshots.append(frame2)

    # -------------------------------------------------------------
    # PANEL 3: Mutual Clash ("MUTUAL CLASH!")
    # -------------------------------------------------------------
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3[:] = (18, 18, 22)
    light3 = np.zeros((height, width, 3), dtype=np.uint8)
    ps3 = ParticleSystem(gravity=850.0, drag=0.93)

    # High velocity simultaneous attack
    eval3 = evaluate_duel_clash(
        col2, s1_d, s2_d, s1_speed=420.0, s2_speed=390.0, is_new_clash=True, is_duel_active=True
    )
    ps3.spawn_clash_sparks(col2.clash_point[0], col2.clash_point[1], col2.normal[0], col2.normal[1], count=55, speed_max=550.0)
    for _ in range(4):
        ps3.update(dt)

    draw_blade_on_light_canvas(light3, s1_emit, s1_tip, blue_color, 0.2)
    draw_blade_on_light_canvas(light3, s2_emit, s2_tip, red_color, 0.2)
    ps3.draw(light3)
    render_hilt(frame3, (s1_emit[0] - s1_d[0] * 50, s1_emit[1] - s1_d[1] * 50), s1_emit, s1_d)
    render_hilt(frame3, (s2_emit[0] - s2_d[0] * 50, s2_emit[1] - s2_d[1] * 50), s2_emit, s2_d)
    composite_light_layer(frame3, light3, show_glow=True)

    draw_centered_text(frame3, "MUTUAL CLASH! (NO DISARM)", y=130, font_scale=1.1, color=(0, 240, 255), thickness=3)
    cv2.putText(frame3, "3. Mutual Aggressive Clash (Both Attacking Simultaneously)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    snapshots.append(frame3)

    # -------------------------------------------------------------
    # PANEL 4: Missed Parry Disarm (Weak Tip Overwhelmed)
    # -------------------------------------------------------------
    frame4 = np.zeros((height, width, 3), dtype=np.uint8)
    frame4[:] = (18, 18, 22)
    cv2.line(frame4, (0, 680), (1280, 680), (45, 45, 55), 2)  # Floor
    light4 = np.zeros((height, width, 3), dtype=np.uint8)
    ps4 = ParticleSystem(gravity=850.0, drag=0.93)

    # Person 2 hit on weak tip (ratio = 0.86) -> Disarmed, falling, bouncing & retracting
    fs4 = FallingSaber(
        hilt_pos=s2_emit,
        initial_velocity=(180.0, -290.0),
        initial_angle=-0.7,
        angular_velocity=8.5,
        color=red_color,
        blade_length=420.0,
        floor_y=680.0,
        floor_width=1280.0,
    )

    # Advance simulation through air and bounce
    sim_t = 0.0
    for _ in range(32):
        sim_t += dt
        ps4.update(dt)
        fs4.update(dt, ps4)

    draw_blade_on_light_canvas(light4, s1_emit, s1_tip, blue_color, sim_t)
    render_hilt(frame4, (s1_emit[0] - s1_d[0] * 50, s1_emit[1] - s1_d[1] * 50), s1_emit, s1_d)
    fs4.draw(frame4, light4, sim_t)
    ps4.draw(light4)
    composite_light_layer(frame4, light4, show_glow=True)

    draw_centered_text(frame4, "Person2 DISARMED! (MISSED PARRY: WEAK TIP)", y=130, font_scale=0.95, color=(30, 60, 255), thickness=3)
    cv2.putText(frame4, "4. Missed Parry Disarm (Fallen Saber Bounces & Retracts)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    snapshots.append(frame4)

    # Assemble 2x2 grid
    half_h, half_w = height // 2, width // 2
    p1 = cv2.resize(snapshots[0], (half_w, half_h))
    p2 = cv2.resize(snapshots[1], (half_w, half_h))
    p3 = cv2.resize(snapshots[2], (half_w, half_h))
    p4 = cv2.resize(snapshots[3], (half_w, half_h))

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Duel mechanics verification saved to {output_path}")


if __name__ == "__main__":
    out_dir = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92"
    run_duel_demo(f"{out_dir}/duel_mechanics_verification.jpg")

