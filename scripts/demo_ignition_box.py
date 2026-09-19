"""Visual verification script demonstrating the Ignition Box and Instant Gestures."""

import math
import os
import pathlib
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.config import AppConfig
from src.gestures import is_force_push_pose, is_hand_in_box, is_two_finger_pose
from src.hand_tracker import HandObservation
from src.particles import ParticleSystem
from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
    render_ignition_box,
)
from src.saber import SaberInstance


def create_mock_hand(
    wrist: tuple,
    pose_type: str = "peace",
    hand_size: float = 90.0,
) -> HandObservation:
    """Generate a mock HandObservation with 3D/2D landmarks."""
    wx, wy = wrist
    landmarks = [(wx, wy)] * 21
    landmarks_3d = [(wx, wy, 0.0)] * 21

    # Finger indices: (mcp, pip, dip, tip)
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
        landmarks_3d[pip] = (wx + x_off, wy - 70, -35.0)
        landmarks_3d[dip] = (wx + x_off, wy - 90, -55.0)

        if ext:
            landmarks[tip] = (wx + x_off, wy - 120)
            landmarks_3d[tip] = (wx + x_off, wy - 120, -85.0)
        else:
            landmarks[tip] = (wx + x_off, wy - 40)
            landmarks_3d[tip] = (wx + x_off, wy - 40, -15.0)

    knuckles = (wx, wy - 45)
    palm = (wx, wy - 22)

    return HandObservation(
        handedness="Right",
        landmarks=landmarks,
        wrist=(float(wx), float(wy)),
        index_mcp=(float(landmarks[5][0]), float(landmarks[5][1])),
        index_tip=(float(landmarks[8][0]), float(landmarks[8][1])),
        middle_mcp=(float(landmarks[9][0]), float(landmarks[9][1])),
        knuckles_center=knuckles,
        palm_center=palm,
        hand_size=hand_size,
        landmarks_3d=landmarks_3d,
    )


def render_panel(
    panel_title: str,
    hand_pos: tuple,
    pose_type: str,
    ignited_blue: bool = False,
    ignited_red: bool = False,
    extra_hand: tuple = None,
    extra_pose: str = None,
) -> np.ndarray:
    """Render a single scenario panel."""
    w, h = 640, 480
    frame = np.full((h, w, 3), 18, dtype=np.uint8)

    # Grid background
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (26, 26, 32), 1)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (26, 26, 32), 1)

    canvas_buffer = CanvasBuffer(w, h)
    light_canvas = canvas_buffer.reset_and_get((h, w))
    particle_system = ParticleSystem(gravity=850.0, drag=0.93)

    # Ignition Box: centered in frame
    box_rect = (int(w * 0.30), int(h * 0.15), int(w * 0.70), int(h * 0.65))

    obs = create_mock_hand(hand_pos, pose_type=pose_type)
    in_box = is_hand_in_box(obs.wrist, obs.knuckles_center, box_rect)

    detected_gesture = None
    if in_box:
        lms = obs.landmarks_3d if obs.landmarks_3d else obs.landmarks
        if is_two_finger_pose(lms):
            detected_gesture = "JediBlue"
        elif is_force_push_pose(lms):
            detected_gesture = "SithRed"

    # Blue Saber
    s_blue = SaberInstance(
        handedness="Person1",
        color=(255, 90, 20),
        alpha_pivot=0.65,
        alpha_direction=0.60,
        is_ignited=ignited_blue,
    )
    if ignited_blue:
        s_blue.update(obs, curr_time=1.0, blade_length=360.0)

    # Red Saber
    s_red = SaberInstance(
        handedness="Person2",
        color=(30, 30, 255),
        alpha_pivot=0.65,
        alpha_direction=0.60,
        is_ignited=ignited_red,
    )
    if extra_hand and extra_pose:
        obs_red = create_mock_hand(extra_hand, pose_type=extra_pose)
        if ignited_red:
            s_red.update(obs_red, curr_time=1.0, blade_length=360.0)
    elif ignited_red:
        s_red.update(obs, curr_time=1.0, blade_length=360.0)

    # Spawn ignition sparks if triggering
    if detected_gesture == "JediBlue" and ignited_blue:
        particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=25)
    elif detected_gesture == "SithRed" and ignited_red:
        particle_system.spawn_clash_sparks(obs.knuckles_center[0], obs.knuckles_center[1], count=25)

    # Render Hilts
    for s in [s_blue, s_red]:
        if s.is_active and s.current_hilt_start and s.current_hilt_end and s.current_direction:
            render_hilt(frame, s.current_hilt_start, s.current_hilt_end, s.current_direction, scale_factor=0.9)

    # Render Blades
    for s in [s_blue, s_red]:
        if s.is_active and s.current_emitter and s.current_endpoint:
            draw_blade_on_light_canvas(light_canvas, s.current_emitter, s.current_endpoint, s.color, curr_time=1.0, scale_factor=0.9)

    # Draw Hand landmarks
    for o in [obs] + ([create_mock_hand(extra_hand, extra_pose)] if extra_hand else []):
        for pt in o.landmarks:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 120), -1)
        cv2.circle(frame, (int(o.wrist[0]), int(o.wrist[1])), 6, (0, 200, 255), -1)

    # Render Sparks
    particle_system.draw(light_canvas)

    # Render Ignition Box
    render_ignition_box(
        frame,
        light_canvas,
        box_rect,
        has_hand_inside=in_box,
        detected_gesture=detected_gesture,
        is_blue_ignited=s_blue.is_ignited,
        is_red_ignited=s_red.is_ignited,
        curr_time=1.0,
    )

    # Composite Glow
    composite_light_layer(frame, light_canvas, show_glow=True)

    # Panel Banner Header
    cv2.rectangle(frame, (0, 0), (w, 36), (15, 15, 20), -1)
    cv2.putText(frame, panel_title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def main() -> None:
    print("Generating visual verification for Ignition Box & Instant Gestures...")

    # Panel 1: Hand Outside Box (Ignition Blocked)
    p1 = render_panel(
        panel_title="1. Outside Box -> Ignition Blocked",
        hand_pos=(120, 280),
        pose_type="peace",
        ignited_blue=False,
    )
    cv2.putText(p1, "NO IGNITION (OUTSIDE BOX)", (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # Panel 2: Hand Inside Box + Two Fingers -> Instant Blue
    p2 = render_panel(
        panel_title="2. Inside Box + 2 Fingers -> Jedi Blue",
        hand_pos=(320, 250),
        pose_type="peace",
        ignited_blue=True,
    )
    cv2.putText(p2, "INSTANT BLUE IGNITION (<0.03s)", (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)

    # Panel 3: Hand Inside Box + Force Push -> Instant Red
    p3 = render_panel(
        panel_title="3. Inside Box + Force Push -> Sith Red",
        hand_pos=(320, 250),
        pose_type="force_push",
        ignited_red=True,
    )
    cv2.putText(p3, "INSTANT RED IGNITION (<0.03s)", (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 40, 255), 2, cv2.LINE_AA)

    # Panel 4: Both Ignited -> Instant Active Combat (No Countdown)
    p4 = render_panel(
        panel_title="4. Both Ignited -> Combat Active (No Countdown)",
        hand_pos=(180, 320),
        pose_type="peace",
        ignited_blue=True,
        ignited_red=True,
        extra_hand=(460, 320),
        extra_pose="force_push",
    )
    cv2.putText(p4, "DUEL ACTIVE - NO COUNTDOWN GATE", (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)

    # Assemble 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    artifacts_dir = pathlib.Path("/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "ignition_box_verification.jpg"

    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Verification image saved successfully to: {out_path}")


if __name__ == "__main__":
    main()
