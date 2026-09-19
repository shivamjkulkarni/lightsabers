"""Visual verification script demonstrating the 3-Round Match System, Dual Ignition Chambers, and Persistent Winner Screen."""

import math
import os
import pathlib
import sys
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.config import AppConfig
from src.falling_saber import FallingSaber
from src.hand_tracker import HandObservation
from src.particles import ParticleSystem
from src.main import draw_centered_text
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
    render_match_winner_screen,
    render_side_chamber,
)
from src.saber import SaberInstance


def create_mock_hand(
    wrist: Tuple[float, float],
    pose_type: str = "peace",
    hand_size: float = 90.0,
) -> HandObservation:
    """Generate a mock HandObservation with landmarks."""
    wx, wy = wrist
    landmarks = [(wx, wy)] * 21
    landmarks_3d = [(wx, wy, 0.0)] * 21

    finger_defs = {
        "index": (5, 6, 7, 8, -25.0),
        "middle": (9, 10, 11, 12, -8.0),
        "ring": (13, 14, 15, 16, 10.0),
        "pinky": (17, 18, 19, 20, 26.0),
    }

    if pose_type == "peace":
        ext_map = {"index": True, "middle": True, "ring": False, "pinky": False}
    elif pose_type == "force_push":
        ext_map = {"index": True, "middle": True, "ring": True, "pinky": True}
    else:  # fist
        ext_map = {"index": False, "middle": False, "ring": False, "pinky": False}

    for name, (mcp, pip, dip, tip, x_off) in finger_defs.items():
        ext = ext_map[name]
        landmarks[mcp] = (wx + x_off, wy - 45)
        landmarks[pip] = (wx + x_off, wy - 70)
        landmarks[dip] = (wx + x_off, wy - 90)
        landmarks_3d[mcp] = (wx + x_off, wy - 45, -15.0)

        if ext:
            landmarks[tip] = (wx + x_off, wy - 120)
            landmarks_3d[tip] = (wx + x_off, wy - 120, 50.0)
        else:
            landmarks[tip] = (wx + x_off * 0.5, wy - 55)
            landmarks_3d[tip] = (wx + x_off * 0.5, wy - 55, -25.0)

    # Thumb
    landmarks[0] = (wx, wy)
    landmarks[1] = (wx - 15, wy - 20)
    landmarks[2] = (wx - 28, wy - 35)
    landmarks[3] = (wx - 38, wy - 50)
    landmarks[4] = (wx - 45, wy - 65)

    knuckles_center = (wx - 4.0, wy - 52.0)
    palm_center = (wx, wy - 25.0)

    return HandObservation(
        handedness="Right",
        landmarks=landmarks,
        landmarks_3d=landmarks_3d,
        wrist=wrist,
        index_mcp=landmarks[5],
        index_tip=landmarks[8],
        middle_mcp=landmarks[9],
        knuckles_center=knuckles_center,
        palm_center=palm_center,
        hand_size=hand_size,
    )


def render_scene_dual_chambers(w: int, h: int, config: AppConfig) -> np.ndarray:
    """Frame 1: Dual Independent Holographic Chambers (Left Blue, Right Red)."""
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    canvas = CanvasBuffer()
    light_canvas = canvas.reset_and_get(frame.shape)

    left_box_rect = (
        int(config.box.left_x_min * w),
        int(config.box.left_y_min * h),
        int(config.box.left_x_max * w),
        int(config.box.left_y_max * h),
    )
    right_box_rect = (
        int(config.box.right_x_min * w),
        int(config.box.right_y_min * h),
        int(config.box.right_x_max * w),
        int(config.box.right_y_max * h),
    )

    # Left Chamber: Hand inside performing Two-Finger Focus
    h_blue = create_mock_hand(wrist=(220, 270), pose_type="peace")
    # Draw hand dots
    for lm in h_blue.landmarks:
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 4, (0, 255, 180), -1)

    render_side_chamber(
        frame,
        light_canvas,
        left_box_rect,
        title="[ JEDI BLUE CHAMBER ]",
        prompt="TWO-FINGER FOCUS TO IGNITE",
        border_color=(255, 180, 50),
        glow_color=config.saber.jedi_blue,
        has_hand_inside=True,
        curr_time=1.0,
    )

    # Right Chamber: Hand entering performing Force Push
    h_red = create_mock_hand(wrist=(740, 270), pose_type="force_push")
    for lm in h_red.landmarks:
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 4, (100, 100, 255), -1)

    render_side_chamber(
        frame,
        light_canvas,
        right_box_rect,
        title="[ SITH RED CHAMBER ]",
        prompt="FORCE PUSH / PALM TO IGNITE",
        border_color=(50, 50, 255),
        glow_color=config.saber.sith_red,
        has_hand_inside=True,
        curr_time=1.0,
    )

    composite_light_layer(frame, light_canvas, show_glow=True)

    # Scoreboard HUD
    scoreboard_str = f"ROUND 1/{config.duel.max_rounds}   [ JEDI BLUE: 0  |  SITH RED: 0 ]"
    draw_centered_text(frame, scoreboard_str, y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame, "AWAITING JEDI & SITH IGNITION (0/2 ACTIVE)", y=65, font_scale=0.55, color=(160, 200, 255), thickness=2)

    return frame


def render_scene_duel_active(w: int, h: int, config: AppConfig) -> np.ndarray:
    """Frame 2: Duel Active - Chambers removed, Scoreboard tracking Round 1."""
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    canvas = CanvasBuffer()
    light_canvas = canvas.reset_and_get(frame.shape)

    s_blue = SaberInstance(handedness="Right", color=config.saber.jedi_blue)
    s_red = SaberInstance(handedness="Right", color=config.saber.sith_red)
    s_blue.ignite()
    s_red.ignite()

    h_blue = create_mock_hand(wrist=(340, 380), pose_type="peace")
    h_red = create_mock_hand(wrist=(620, 380), pose_type="force_push")

    s_blue.update(h_blue, curr_time=1.0, blade_length=420.0, scale_factor=1.0)
    s_red.update(h_red, curr_time=1.0, blade_length=420.0, scale_factor=1.0)

    # Draw hilts and blades
    render_hilt(frame, s_blue.current_hilt_start, s_blue.current_hilt_end, s_blue.current_direction, scale_factor=1.0)
    render_hilt(frame, s_red.current_hilt_start, s_red.current_hilt_end, s_red.current_direction, scale_factor=1.0)
    draw_blade_on_light_canvas(light_canvas, s_blue.current_emitter, s_blue.current_endpoint, s_blue.color, 1.0, scale_factor=1.0)
    draw_blade_on_light_canvas(light_canvas, s_red.current_emitter, s_red.current_endpoint, s_red.color, 1.0, scale_factor=1.0)

    composite_light_layer(frame, light_canvas, show_glow=True)

    scoreboard_str = f"ROUND 1/{config.duel.max_rounds}   [ JEDI BLUE: 0  |  SITH RED: 0 ]"
    draw_centered_text(frame, scoreboard_str, y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame, "[ DUEL ACTIVE: PARRY OR BE DISARMED ]", y=65, font_scale=0.55, color=(0, 255, 180), thickness=2)

    return frame


def render_scene_round_disarm(w: int, h: int, config: AppConfig) -> np.ndarray:
    """Frame 3: Round 1 Conclusion - Sith Red disarmed & tumbling, Jedi Blue steady, Score 1 - 0."""
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    canvas = CanvasBuffer()
    light_canvas = canvas.reset_and_get(frame.shape)
    particles = ParticleSystem()

    # Steady winning Blue Saber
    s_blue = SaberInstance(handedness="Right", color=config.saber.jedi_blue)
    s_blue.ignite()
    h_blue = create_mock_hand(wrist=(280, 360), pose_type="peace")
    s_blue.update(h_blue, curr_time=1.0, blade_length=420.0, scale_factor=1.0)
    render_hilt(frame, s_blue.current_hilt_start, s_blue.current_hilt_end, s_blue.current_direction, scale_factor=1.0)
    draw_blade_on_light_canvas(light_canvas, s_blue.current_emitter, s_blue.current_endpoint, s_blue.color, 1.0, scale_factor=1.0)

    # Disarmed falling Sith saber bouncing on floor with sparks
    falling = FallingSaber(
        hilt_pos=(680.0, 360.0),
        initial_velocity=(160.0, -120.0),
        initial_angle=1.2,
        angular_velocity=6.0,
        color=config.saber.sith_red,
        blade_length=380.0,
        floor_y=float(h - 20),
        floor_width=float(w),
    )
    # Advance falling physics
    for _ in range(12):
        falling.update(0.033, particles)
    particles.spawn_clash_sparks(680, h - 25, count=25)

    falling.draw(frame, light_canvas, curr_time=1.5)
    particles.draw(light_canvas)

    composite_light_layer(frame, light_canvas, show_glow=True)

    scoreboard_str = f"ROUND 1/{config.duel.max_rounds}   [ JEDI BLUE: 1  |  SITH RED: 0 ]"
    draw_centered_text(frame, scoreboard_str, y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(frame, "ROUND 1 OVER - PREPARING NEXT ROUND...", y=58, font_scale=0.55, color=(30, 100, 255), thickness=2)
    draw_centered_text(frame, "JEDI WINS ROUND 1!", y=130, font_scale=1.0, color=config.saber.jedi_blue, thickness=3, outline_thickness=7)

    return frame


def render_scene_match_winner(w: int, h: int, config: AppConfig) -> np.ndarray:
    """Frame 4: Match Over - Dramatic Persistent Victory Screen."""
    frame = np.full((h, w, 3), (25, 28, 35), dtype=np.uint8)
    canvas = CanvasBuffer()
    light_canvas = canvas.reset_and_get(frame.shape)

    render_match_winner_screen(
        frame=frame,
        light_canvas=light_canvas,
        winner_name="Jedi Blue",
        winner_color=config.saber.jedi_blue,
        score_blue=2,
        score_red=1,
        curr_time=1.0,
    )

    composite_light_layer(frame, light_canvas, show_glow=True)

    return frame


def main() -> None:
    config = AppConfig()
    pw, ph = 960, 540

    print("Rendering Panel 1: Dual Independent Chambers...")
    p1 = render_scene_dual_chambers(pw, ph, config)

    print("Rendering Panel 2: Duel Active...")
    p2 = render_scene_duel_active(pw, ph, config)

    print("Rendering Panel 3: Round Disarm...")
    p3 = render_scene_round_disarm(pw, ph, config)

    print("Rendering Panel 4: Match Winner Screen...")
    p4 = render_scene_match_winner(pw, ph, config)

    # Stitch into 2x2 grid (1920x1080 Full HD)
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bottom_row])

    # Save to artifacts directory
    artifact_dir = pathlib.Path("/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "match_system_verification.jpg"

    cv2.imwrite(str(out_path), grid)
    print(f"Successfully generated verification artifact: {out_path}")


if __name__ == "__main__":
    main()
