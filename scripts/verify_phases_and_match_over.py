"""Verification script demonstrating phased wait times, chamber gating, and clean MATCH_OVER screen."""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.renderer import (
    CanvasBuffer,
    composite_light_layer,
    draw_blade_on_light_canvas,
    render_hilt,
    render_match_winner_screen,
    render_side_chamber,
)
from src.main import draw_centered_text


def generate_verification_panels():
    width, height = 1280, 720
    canvas_buffer = CanvasBuffer(width, height)
    panels = []

    left_box = (int(0.06 * width), int(0.20 * height), int(0.40 * width), int(0.70 * height))
    right_box = (int(0.60 * width), int(0.20 * height), int(0.94 * width), int(0.70 * height))

    jedi_blue = (255, 90, 20)
    sith_red = (30, 30, 255)

    def make_backdrop():
        f = np.full((height, width, 3), (15, 18, 24), dtype=np.uint8)
        for gy in range(0, height, 50):
            cv2.line(f, (0, gy), (width, gy), (22, 26, 32), 1)
        for gx in range(0, width, 50):
            cv2.line(f, (gx, 0), (gx, height), (22, 26, 32), 1)
        return f

    # Panel 1: ROUND_PREPARATION (chambers not open yet, player preparation)
    f1 = make_backdrop()
    lc1 = canvas_buffer.reset_and_get(f1.shape)
    draw_centered_text(f1, "ROUND 1/3   [ JEDI BLUE: 0  |  SITH RED: 0 ]", y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(f1, "STEP TO YOUR SIDES -- CHAMBERS OPEN IN 2s", y=88, font_scale=0.55, color=(0, 255, 255), thickness=2)
    draw_centered_text(f1, "ROUND 1 -- TAKE YOUR SIDES!", y=155, font_scale=0.95, color=(0, 255, 255), thickness=3)
    composite_light_layer(f1, lc1, show_glow=True)
    panels.append(("Phase 1: ROUND_PREPARATION (Wait Buffer Before Ignition)", f1))

    # Panel 2: AWAITING_IGNITION (both chambers open and active)
    f2 = make_backdrop()
    lc2 = canvas_buffer.reset_and_get(f2.shape)
    render_side_chamber(
        f2, lc2, left_box,
        title="[ JEDI BLUE CHAMBER ]",
        prompt="TWO-FINGER FOCUS TO IGNITE",
        border_color=(255, 180, 50),
        glow_color=jedi_blue,
        has_hand_inside=False,
        curr_time=1.0,
    )
    render_side_chamber(
        f2, lc2, right_box,
        title="[ SITH RED CHAMBER ]",
        prompt="FORCE PUSH / PALM TO IGNITE",
        border_color=(50, 50, 255),
        glow_color=sith_red,
        has_hand_inside=False,
        curr_time=1.0,
    )
    draw_centered_text(f2, "ROUND 1/3   [ JEDI BLUE: 0  |  SITH RED: 0 ]", y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(f2, "AWAITING JEDI & SITH IGNITION (0/2 ACTIVE)", y=88, font_scale=0.52, color=(160, 200, 255), thickness=2)
    composite_light_layer(f2, lc2, show_glow=True)
    panels.append(("Phase 2: AWAITING_IGNITION (Independent Spatial Chambers)", f2))

    # Panel 3: IGNITION_LOCKED (both ignited, chambers gone, 1.4s pause before countdown)
    f3 = make_backdrop()
    lc3 = canvas_buffer.reset_and_get(f3.shape)
    # Blue blade active on left
    render_hilt(f3, (280, 480), (320, 420), (0.7, -0.7))
    draw_blade_on_light_canvas(lc3, (320, 420), (520, 180), jedi_blue, curr_time=1.0)
    # Red blade active on right
    render_hilt(f3, (1000, 480), (960, 420), (-0.7, -0.7))
    draw_blade_on_light_canvas(lc3, (960, 420), (760, 180), sith_red, curr_time=1.0)
    draw_centered_text(f3, "ROUND 1/3   [ JEDI BLUE: 0  |  SITH RED: 0 ]", y=32, font_scale=0.65, color=(255, 255, 255), thickness=2)
    draw_centered_text(f3, "JEDI GUARD: [ + + ]         SITH GUARD: [ + + ]", y=60, font_scale=0.55, color=(0, 230, 255), thickness=2)
    draw_centered_text(f3, ">> BOTH SABERS IGNITED -- ASSUME BATTLE STANCE! <<", y=88, font_scale=0.56, color=(255, 220, 0), thickness=2)
    draw_centered_text(f3, ">> BOTH SABERS IGNITED -- PREPARE TO DUEL! <<", y=155, font_scale=0.90, color=(255, 215, 0), thickness=3)
    composite_light_layer(f3, lc3, show_glow=True)
    panels.append(("Phase 3: IGNITION_LOCKED (Lock-in Pause Before Countdown)", f3))

    # Panel 4: MATCH_OVER (Victory screen with ZERO swords or hilts)
    f4 = make_backdrop()
    lc4 = canvas_buffer.reset_and_get(f4.shape)
    # Swords/hilts are NOT drawn! Clean end screen
    render_match_winner_screen(
        f4,
        lc4,
        winner_name="Jedi Blue",
        winner_color=jedi_blue,
        score_blue=2,
        score_red=1,
        curr_time=1.0,
    )
    composite_light_layer(f4, lc4, show_glow=True)
    panels.append(("Phase 4: MATCH_OVER (Zero Swords / Hilts Bleeding Through)", f4))

    # Save 2x2 grid image
    scaled = [cv2.resize(img, (640, 360)) for _, img in panels]
    # Add title headers to each quadrant
    for idx, (title, _) in enumerate(panels):
        cv2.putText(scaled[idx], title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(scaled[idx], title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    top_row = np.hstack([scaled[0], scaled[1]])
    bottom_row = np.hstack([scaled[2], scaled[3]])
    grid = np.vstack([top_row, bottom_row])

    artifact_path = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92/phased_duel_and_clean_match_over.jpg"
    cv2.imwrite(artifact_path, grid)
    print(f"Saved verification grid to {artifact_path}")


if __name__ == "__main__":
    generate_verification_panels()
