"""Generates visual verification sequence of collision physics, sparks, and falling saber."""

import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.collision import BladeRecoilController, check_blade_collision
from src.falling_saber import FallingSaber
from src.particles import ParticleSystem
from src.renderer import (
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
)


def run_collision_demo(output_path: str = "artifacts/collision_physics_demo.jpg") -> None:
    width, height = 1280, 720
    dt = 0.016
    sim_time = 0.0

    particle_system = ParticleSystem(gravity=850.0, drag=0.93)
    recoil = BladeRecoilController(omega=38.0, zeta=0.75)
    falling_sabers = []

    # Frame 1: Pre-clash crossing blades
    # Frame 2: Mid-clash impact with radiant sparks spraying and blade recoil
    # Frame 3: Disarmed red saber tumbling mid-air
    # Frame 4: Falling saber bouncing on the floor with upward ground sparks and retraction

    snapshots = []

    # Simulation setup
    # Saber 1 (Jedi Blue): striking downwards-right
    s1_emitter = (380.0, 480.0)
    s1_tip = (780.0, 240.0)
    s1_dir = (s1_tip[0] - s1_emitter[0], s1_tip[1] - s1_emitter[1])
    s1_len = math.hypot(*s1_dir)
    s1_dir = (s1_dir[0] / s1_len, s1_dir[1] / s1_len)

    # Saber 2 (Sith Red): blocking upwards-left
    s2_emitter = (900.0, 480.0)
    s2_tip = (500.0, 250.0)
    s2_dir = (s2_tip[0] - s2_emitter[0], s2_tip[1] - s2_emitter[1])
    s2_len = math.hypot(*s2_dir)
    s2_dir = (s2_dir[0] / s2_len, s2_dir[1] / s2_len)

    blue_color = (255, 90, 20)
    red_color = (30, 30, 255)

    # Snapshot 1: Approach / Crossing
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:] = (18, 18, 22)  # Dark room atmosphere
    light1 = np.zeros((height, width, 3), dtype=np.uint8)
    draw_blade_on_light_canvas(light1, s1_emitter, s1_tip, blue_color, 0.0)
    draw_blade_on_light_canvas(light1, s2_emitter, s2_tip, red_color, 0.0)
    render_hilt(frame1, (s1_emitter[0] - s1_dir[0] * 50, s1_emitter[1] - s1_dir[1] * 50), s1_emitter, s1_dir)
    render_hilt(frame1, (s2_emitter[0] - s2_dir[0] * 50, s2_emitter[1] - s2_dir[1] * 50), s2_emitter, s2_dir)
    composite_light_layer(frame1, light1, show_glow=True)
    cv2.putText(frame1, "1. Crossing Blades Contact", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    snapshots.append(frame1)

    # Clash & Sparks
    col = check_blade_collision(s1_emitter, s1_tip, s2_emitter, s2_tip)
    assert col is not None
    clash_x, clash_y = col.clash_point
    particle_system.spawn_clash_sparks(clash_x, clash_y, col.normal[0], col.normal[1], count=45)
    recoil.apply_clash_impulse("p1", "p2", col.normal, s1_dir, s2_dir, impulse_magnitude=0.55)

    # Advance a few steps to let sparks fan out and recoil deflect
    for _ in range(3):
        sim_time += dt
        particle_system.update(dt)
        recoil.update(dt)

    # Snapshot 2: Clash Impact & Spraying Newtonian Sparks
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2[:] = (18, 18, 22)
    light2 = np.zeros((height, width, 3), dtype=np.uint8)
    # Recoiled blades
    r1 = recoil.get_recoil_angle("p1")
    r2 = recoil.get_recoil_angle("p2")
    c1, s1 = math.cos(r1), math.sin(r1)
    d1 = (c1 * s1_dir[0] - s1 * s1_dir[1], s1 * s1_dir[0] + c1 * s1_dir[1])
    tip1 = (s1_emitter[0] + d1[0] * s1_len, s1_emitter[1] + d1[1] * s1_len)
    c2, s2 = math.cos(r2), math.sin(r2)
    d2 = (c2 * s2_dir[0] - s2 * s2_dir[1], s2 * s2_dir[0] + c2 * s2_dir[1])
    tip2 = (s2_emitter[0] + d2[0] * s2_len, s2_emitter[1] + d2[1] * s2_len)

    draw_blade_on_light_canvas(light2, s1_emitter, tip1, blue_color, sim_time)
    draw_blade_on_light_canvas(light2, s2_emitter, tip2, red_color, sim_time)
    particle_system.draw(light2)
    render_hilt(frame2, (s1_emitter[0] - d1[0] * 50, s1_emitter[1] - d1[1] * 50), s1_emitter, d1)
    render_hilt(frame2, (s2_emitter[0] - d2[0] * 50, s2_emitter[1] - d2[1] * 50), s2_emitter, d2)
    composite_light_layer(frame2, light2, show_glow=True)
    cv2.putText(frame2, "2. Clash Sparks & Elastic Recoil", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    snapshots.append(frame2)

    # Disarm Red Saber: spawn FallingSaber
    fs = FallingSaber(
        hilt_pos=s2_emitter,
        initial_velocity=(190.0, -320.0),
        initial_angle=math.atan2(s2_dir[1], s2_dir[0]),
        angular_velocity=8.2,
        color=red_color,
        blade_length=s2_len,
        floor_y=680.0,
        floor_width=1280.0,
    )
    falling_sabers.append(fs)

    # Advance until tumbling mid-air
    for _ in range(16):
        sim_time += dt
        particle_system.update(dt)
        fs.update(dt, particle_system)

    # Snapshot 3: Mid-Air Disarm & Tumble
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3[:] = (18, 18, 22)
    light3 = np.zeros((height, width, 3), dtype=np.uint8)
    draw_blade_on_light_canvas(light3, s1_emitter, s1_tip, blue_color, sim_time)
    render_hilt(frame3, (s1_emitter[0] - s1_dir[0] * 50, s1_emitter[1] - s1_dir[1] * 50), s1_emitter, s1_dir)
    fs.draw(frame3, light3, sim_time)
    particle_system.draw(light3)
    composite_light_layer(frame3, light3, show_glow=True)
    cv2.putText(frame3, "3. Disarmed Saber Tumbling Mid-Air", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    snapshots.append(frame3)

    # Advance until floor bounces and retracting
    for _ in range(35):
        sim_time += dt
        particle_system.update(dt)
        fs.update(dt, particle_system)

    # Snapshot 4: Floor Impact, Ground Sparks & Retraction
    frame4 = np.zeros((height, width, 3), dtype=np.uint8)
    frame4[:] = (18, 18, 22)
    # Floor line
    cv2.line(frame4, (0, 680), (1280, 680), (45, 45, 55), 2)
    light4 = np.zeros((height, width, 3), dtype=np.uint8)
    draw_blade_on_light_canvas(light4, s1_emitter, s1_tip, blue_color, sim_time)
    render_hilt(frame4, (s1_emitter[0] - s1_dir[0] * 50, s1_emitter[1] - s1_dir[1] * 50), s1_emitter, s1_dir)
    fs.draw(frame4, light4, sim_time)
    particle_system.draw(light4)
    composite_light_layer(frame4, light4, show_glow=True)
    cv2.putText(frame4, "4. Floor Impact Bounce, Ground Sparks & Retraction", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    snapshots.append(frame4)

    # Combine 4 panels into a 2x2 grid (1280x720 total)
    half_h, half_w = height // 2, width // 2
    p1 = cv2.resize(snapshots[0], (half_w, half_h))
    p2 = cv2.resize(snapshots[1], (half_w, half_h))
    p3 = cv2.resize(snapshots[2], (half_w, half_h))
    p4 = cv2.resize(snapshots[3], (half_w, half_h))

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    # Ensure output dir exists
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Collision physics demonstration saved to {output_path}")


if __name__ == "__main__":
    import os
    out_dir = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92"
    run_collision_demo(os.path.join(out_dir, "collision_physics_verification.jpg"))

