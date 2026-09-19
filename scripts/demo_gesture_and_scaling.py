"""Demonstrates Two-Finger Blue and Force-Push Red gesture ignition and proper distance scaling."""

import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.geometry import (
    calculate_distance_scaled_blade_length,
    calculate_perspective_scale_factor,
)
from src.gestures import is_force_push_pose, is_two_finger_pose
from src.main import draw_centered_text
from src.particles import ParticleSystem
from src.renderer import (
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
)


def run_demo(output_path: str) -> None:
    width, height = 1280, 720
    blue_color = (255, 90, 20)
    red_color = (30, 30, 255)

    snapshots = []

    # -------------------------------------------------------------
    # PANEL 1: Two-Finger Pose -> Jedi Blue Ignition
    # -------------------------------------------------------------
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:] = (18, 18, 22)
    light1 = np.zeros((height, width, 3), dtype=np.uint8)
    ps1 = ParticleSystem()

    emit1 = (450.0, 520.0)
    dir1 = (0.35, -0.93)
    tip1 = (emit1[0] + dir1[0] * 420.0, emit1[1] + dir1[1] * 420.0)

    render_hilt(frame1, (emit1[0] - dir1[0] * 50, emit1[1] - dir1[1] * 50), emit1, dir1, scale_factor=1.0)
    draw_blade_on_light_canvas(light1, emit1, tip1, blue_color, scale_factor=1.0)
    ps1.spawn_clash_sparks(emit1[0], emit1[1], count=30)
    ps1.update(0.02)
    ps1.draw(light1)
    composite_light_layer(frame1, light1, show_glow=True)

    draw_centered_text(frame1, "JEDI BLUE IGNITED! (TWO-FINGER POSE)", y=100, font_scale=1.0, color=(255, 140, 0), thickness=3)
    cv2.putText(frame1, "1. Two-Finger Focus Sign Held Toward Camera -> Blue Saber Awakens", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    snapshots.append(frame1)

    # -------------------------------------------------------------
    # PANEL 2: Force Push Pose -> Sith Red Ignition
    # -------------------------------------------------------------
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2[:] = (18, 18, 22)
    light2 = np.zeros((height, width, 3), dtype=np.uint8)
    ps2 = ParticleSystem()

    emit2 = (800.0, 520.0)
    dir2 = (-0.35, -0.93)
    tip2 = (emit2[0] + dir2[0] * 420.0, emit2[1] + dir2[1] * 420.0)

    render_hilt(frame2, (emit2[0] - dir2[0] * 50, emit2[1] - dir2[1] * 50), emit2, dir2, scale_factor=1.0)
    draw_blade_on_light_canvas(light2, emit2, tip2, red_color, scale_factor=1.0)
    ps2.spawn_clash_sparks(emit2[0], emit2[1], count=30)
    ps2.update(0.02)
    ps2.draw(light2)
    composite_light_layer(frame2, light2, show_glow=True)

    draw_centered_text(frame2, "SITH RED IGNITED! (FORCE PUSH POSE)", y=100, font_scale=1.0, color=(30, 60, 255), thickness=3)
    cv2.putText(frame2, "2. Force Push / Open Palm Toward Camera -> Red Saber Awakens", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    snapshots.append(frame2)

    # -------------------------------------------------------------
    # PANEL 3: Proper 3D Distance Perspective Scaling (Length + Thickness)
    # -------------------------------------------------------------
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3[:] = (18, 18, 22)
    light3 = np.zeros((height, width, 3), dtype=np.uint8)

    # Near hand (hand_size = 95px): Full scale
    len_near = calculate_distance_scaled_blade_length(95.0)
    scale_near = calculate_perspective_scale_factor(95.0)
    e_near = (350.0, 560.0)
    d_near = (0.45, -0.89)
    t_near = (e_near[0] + d_near[0] * len_near, e_near[1] + d_near[1] * len_near)
    render_hilt(frame3, (e_near[0] - d_near[0] * 50, e_near[1] - d_near[1] * 50), e_near, d_near, scale_factor=scale_near)
    draw_blade_on_light_canvas(light3, e_near, t_near, blue_color, scale_factor=scale_near)

    # Far hand (hand_size = 35px): Proportional distance scaling
    len_far = calculate_distance_scaled_blade_length(35.0)
    scale_far = calculate_perspective_scale_factor(35.0)
    e_far = (900.0, 520.0)
    d_far = (-0.45, -0.89)
    t_far = (e_far[0] + d_far[0] * len_far, e_far[1] + d_far[1] * len_far)
    render_hilt(frame3, (e_far[0] - d_far[0] * 30, e_far[1] - d_far[1] * 30), e_far, d_far, scale_factor=scale_far)
    draw_blade_on_light_canvas(light3, e_far, t_far, red_color, scale_factor=scale_far)

    composite_light_layer(frame3, light3, show_glow=True)
    cv2.putText(frame3, f"Near Hand (95px): Blade {len_near:.0f}px | Thickness {scale_near:.2f}x", (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
    cv2.putText(frame3, f"Far Hand (35px): Blade {len_far:.0f}px | Thickness {scale_far:.2f}x", (780, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
    cv2.putText(frame3, "3. Proper Distance Scaling: Proportional Length & Sleek Thickness", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    snapshots.append(frame3)

    # -------------------------------------------------------------
    # PANEL 4: Both Sabers Ignited & Ready for Duel Countdown
    # -------------------------------------------------------------
    frame4 = np.zeros((height, width, 3), dtype=np.uint8)
    frame4[:] = (18, 18, 22)
    light4 = np.zeros((height, width, 3), dtype=np.uint8)

    e1 = (420.0, 520.0)
    d1 = (0.6, -0.8)
    t1 = (e1[0] + d1[0] * 400.0, e1[1] + d1[1] * 400.0)
    render_hilt(frame4, (e1[0] - d1[0] * 50, e1[1] - d1[1] * 50), e1, d1, scale_factor=1.0)
    draw_blade_on_light_canvas(light4, e1, t1, blue_color, scale_factor=1.0)

    e2 = (860.0, 520.0)
    d2 = (-0.6, -0.8)
    t2 = (e2[0] + d2[0] * 400.0, e2[1] + d2[1] * 400.0)
    render_hilt(frame4, (e2[0] - d2[0] * 50, e2[1] - d2[1] * 50), e2, d2, scale_factor=1.0)
    draw_blade_on_light_canvas(light4, e2, t2, red_color, scale_factor=1.0)

    composite_light_layer(frame4, light4, show_glow=True)
    draw_centered_text(frame4, "DUEL IN: 3", y=90, font_scale=1.4, color=(0, 230, 255), thickness=4)
    draw_centered_text(frame4, "BOTH JEDI RECOGNIZED & IGNITED", y=150, font_scale=0.85, color=(0, 255, 180), thickness=2)
    cv2.putText(frame4, "4. Duel Countdown Armed (Crossing Blades Without Disarming)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    snapshots.append(frame4)

    # 2x2 grid
    half_h, half_w = height // 2, width // 2
    p1 = cv2.resize(snapshots[0], (half_w, half_h))
    p2 = cv2.resize(snapshots[1], (half_w, half_h))
    p3 = cv2.resize(snapshots[2], (half_w, half_h))
    p4 = cv2.resize(snapshots[3], (half_w, half_h))

    grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Gesture & scaling demonstration saved to {output_path}")


if __name__ == "__main__":
    out_dir = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92"
    run_demo(f"{out_dir}/gesture_and_scaling_verification.jpg")
