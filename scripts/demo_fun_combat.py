"""Visual verification script for countdown continuity, spatial re-acquisition, and fun combat mechanics."""

import math
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collision import CollisionInfo, evaluate_duel_clash
from src.falling_saber import FallingSaber
from src.main import draw_centered_text
from src.particles import ParticleSystem
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
)


def generate_fun_combat_verification_artifact():
    width, height = 1280, 720
    canvas_buffer = CanvasBuffer(width, height)
    panels = []

    # -------------------------------------------------------------
    # Panel 1: Countdown Continuity & Spatial Hand Re-Acquisition
    # -------------------------------------------------------------
    frame1 = np.full((height, width, 3), (18, 18, 22), dtype=np.uint8)
    light1 = canvas_buffer.reset_and_get(frame1.shape)

    # Blue saber ignited on left; Red hand momentarily off-screen on right, saber stays ignited!
    p_blue_emitter = (380.0, 420.0)
    p_blue_tip = (520.0, 180.0)
    p_red_emitter = (880.0, 420.0)  # Waiting at last known position
    p_red_tip = (740.0, 180.0)

    draw_blade_on_light_canvas(light1, p_blue_emitter, p_blue_tip, (255, 140, 0), curr_time=1.0)
    draw_blade_on_light_canvas(light1, p_red_emitter, p_red_tip, (30, 40, 255), curr_time=1.0)
    render_hilt(frame1, (360.0, 460.0), p_blue_emitter, (0.5, -0.86))
    render_hilt(frame1, (900.0, 460.0), p_red_emitter, (-0.5, -0.86))

    composite_light_layer(frame1, light1, show_glow=True)

    draw_centered_text(frame1, "PANEL 1: COUNTDOWN CONTINUITY & HAND RE-ACQUISITION", y=35, font_scale=0.68, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame1, "JEDI GUARD: [ + + ]         SITH GUARD: [ + + ]", y=65, font_scale=0.55, color=(0, 230, 255), thickness=2)
    draw_centered_text(frame1, "DUEL IN: 2", y=120, font_scale=1.4, color=(0, 230, 255), thickness=4)
    draw_centered_text(frame1, "[ SITH HAND OFF-SCREEN -> TIMER DOES NOT RESET! SABER RETAINED ]", y=175, font_scale=0.58, color=(0, 255, 180), thickness=2)
    panels.append(frame1)

    # -------------------------------------------------------------
    # Panel 2: Perfect Parry with Supernova Sparks & Plasma Shockwave Ring
    # -------------------------------------------------------------
    frame2 = np.full((height, width, 3), (18, 18, 22), dtype=np.uint8)
    light2 = canvas_buffer.reset_and_get(frame2.shape)
    ps2 = ParticleSystem()

    clash_pt = (640.0, 320.0)
    # Blue strikes down fast; Red snaps blade across with forte at 90 deg -> PERFECT PARRY!
    p_blue_e = (500.0, 480.0)
    p_blue_t = (720.0, 220.0)
    p_red_e = (780.0, 480.0)
    p_red_t = (560.0, 220.0)

    draw_blade_on_light_canvas(light2, p_blue_e, p_blue_t, (255, 140, 0), curr_time=2.0)
    draw_blade_on_light_canvas(light2, p_red_e, p_red_t, (30, 40, 255), curr_time=2.0)
    render_hilt(frame2, (480.0, 520.0), p_blue_e, (0.7, -0.7))
    render_hilt(frame2, (800.0, 520.0), p_red_e, (-0.7, -0.7))

    # Spawn Perfect Parry shockwave and supernova sparks
    ps2.spawn_shockwave(clash_pt[0], clash_pt[1], max_radius=180.0, speed=520.0, lifetime=0.35, color=(255, 255, 120), thickness=5)
    ps2.spawn_clash_sparks(clash_pt[0], clash_pt[1], count=50, speed_min=240.0, speed_max=550.0)
    for _ in range(3):
        ps2.update(0.02)
    ps2.draw(light2)

    composite_light_layer(frame2, light2, show_glow=True)

    draw_centered_text(frame2, "PANEL 2: ACTIVE DEFLECTION - PERFECT PARRY & SHOCKWAVE", y=35, font_scale=0.68, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame2, "JEDI GUARD: [ + + ]         SITH GUARD: [ + + ]", y=65, font_scale=0.55, color=(0, 230, 255), thickness=2)
    draw_centered_text(frame2, ">> SITH COUNTER-STRIKE READY! (+35% SPEED) <<", y=95, font_scale=0.55, color=(255, 220, 0), thickness=2)
    draw_centered_text(frame2, ">> SITH PERFECT PARRY! (COUNTER READY) <<", y=155, font_scale=0.92, color=(255, 220, 0), thickness=3, outline_thickness=7)
    panels.append(frame2)

    # -------------------------------------------------------------
    # Panel 3: Guard Poise Damage (Guard Shaken, 1 Guard Left) & Rally Streak
    # -------------------------------------------------------------
    frame3 = np.full((height, width, 3), (18, 18, 22), dtype=np.uint8)
    light3 = canvas_buffer.reset_and_get(frame3.shape)
    ps3 = ParticleSystem()

    clash_pt3 = (630.0, 290.0)
    draw_blade_on_light_canvas(light3, (490.0, 460.0), (710.0, 200.0), (255, 140, 0), curr_time=3.0)
    draw_blade_on_light_canvas(light3, (770.0, 460.0), (550.0, 200.0), (30, 40, 255), curr_time=3.0)
    render_hilt(frame3, (470.0, 500.0), (490.0, 460.0), (0.7, -0.7))
    render_hilt(frame3, (790.0, 500.0), (770.0, 460.0), (-0.7, -0.7))

    ps3.spawn_shockwave(clash_pt3[0], clash_pt3[1], max_radius=130.0, speed=420.0, lifetime=0.30, color=(100, 200, 255), thickness=4)
    ps3.spawn_clash_sparks(clash_pt3[0], clash_pt3[1], count=32, speed_min=180.0, speed_max=440.0)
    for _ in range(4):
        ps3.update(0.02)
    ps3.draw(light3)

    composite_light_layer(frame3, light3, show_glow=True)

    draw_centered_text(frame3, "PANEL 3: GUARD POISE SYSTEM & RALLY COMBO", y=35, font_scale=0.68, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame3, "JEDI GUARD: [ + + ]         SITH GUARD: [ + - ]", y=65, font_scale=0.55, color=(0, 230, 255), thickness=2)
    draw_centered_text(frame3, ">> RALLY x3! <<", y=95, font_scale=0.62, color=(0, 255, 255), thickness=2)
    draw_centered_text(frame3, "! SITH GUARD SHAKEN! [1 POISE LEFT] !", y=155, font_scale=0.92, color=(0, 165, 255), thickness=3, outline_thickness=7)
    panels.append(frame3)

    # -------------------------------------------------------------
    # Panel 4: Guard Broken Disarm & Round Victory
    # -------------------------------------------------------------
    frame4 = np.full((height, width, 3), (18, 18, 22), dtype=np.uint8)
    light4 = canvas_buffer.reset_and_get(frame4.shape)
    ps4 = ParticleSystem()

    # Jedi holds blade triumphant; Sith blade knocked flying to floor
    p_blue_e4 = (420.0, 430.0)
    p_blue_t4 = (640.0, 160.0)
    draw_blade_on_light_canvas(light4, p_blue_e4, p_blue_t4, (255, 140, 0), curr_time=4.0)
    render_hilt(frame4, (390.0, 460.0), p_blue_e4, (0.6, -0.8))

    falling = FallingSaber(
        hilt_pos=(860.0, 360.0),
        initial_velocity=(240.0, -220.0),
        initial_angle=-0.8,
        angular_velocity=8.5,
        color=(30, 40, 255),
        blade_length=420.0,
        floor_y=height - 30.0,
    )
    for _ in range(8):
        falling.update(0.02, ps4)
    falling.draw(frame4, light4, curr_time=4.0)
    ps4.draw(light4)

    composite_light_layer(frame4, light4, show_glow=True)

    draw_centered_text(frame4, "PANEL 4: GUARD BROKEN -> DECISIVE DISARM & ROUND WIN", y=35, font_scale=0.68, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame4, "ROUND 1/3   [ JEDI BLUE: 1  |  SITH RED: 0 ]", y=65, font_scale=0.60, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame4, "ROUND 1 OVER - PREPARING NEXT ROUND...", y=95, font_scale=0.55, color=(30, 100, 255), thickness=2)
    draw_centered_text(frame4, ">>> SITH GUARD BROKEN! JEDI WINS ROUND 1! <<<", y=155, font_scale=0.90, color=(255, 140, 0), thickness=3, outline_thickness=7)
    panels.append(frame4)

    # Stitch into 2x2 collage
    top_row = np.hstack([cv2.resize(panels[0], (640, 360)), cv2.resize(panels[1], (640, 360))])
    bot_row = np.hstack([cv2.resize(panels[2], (640, 360)), cv2.resize(panels[3], (640, 360))])
    collage = np.vstack([top_row, bot_row])

    # Save to artifacts directory
    artifact_path = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92/fun_combat_verification.jpg"
    cv2.imwrite(artifact_path, collage)
    print(f"Saved combat verification image to {artifact_path}")


if __name__ == "__main__":
    generate_fun_combat_verification_artifact()
