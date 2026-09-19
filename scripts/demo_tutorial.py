"""Demo script generating visual verification artifact of all 3 tutorial briefing slides."""

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
    render_side_chamber,
    render_tutorial_card,
)


def generate_tutorial_verification_artifact():
    width, height = 1280, 720
    canvas_buffer = CanvasBuffer(width, height)
    slides_frames = []

    left_box = (int(0.06 * width), int(0.20 * height), int(0.40 * width), int(0.70 * height))
    right_box = (int(0.60 * width), int(0.20 * height), int(0.94 * width), int(0.70 * height))

    for slide_idx in (1, 2, 3):
        frame = np.full((height, width, 3), (16, 18, 24), dtype=np.uint8)
        light_canvas = canvas_buffer.reset_and_get(frame.shape)

        # Draw holographic chambers and demonstration sabers in background
        render_side_chamber(
            frame,
            light_canvas,
            left_box,
            title="[ JEDI BLUE CHAMBER ]",
            prompt="TWO-FINGER FOCUS TO IGNITE",
            border_color=(255, 180, 50),
            glow_color=(255, 90, 20),
            has_hand_inside=False,
            curr_time=1.0,
        )
        render_side_chamber(
            frame,
            light_canvas,
            right_box,
            title="[ SITH RED CHAMBER ]",
            prompt="FORCE PUSH / PALM TO IGNITE",
            border_color=(50, 50, 255),
            glow_color=(30, 30, 255),
            has_hand_inside=False,
            curr_time=1.0,
        )

        # Draw the tutorial card
        render_tutorial_card(frame, light_canvas, slide_index=slide_idx, curr_time=float(slide_idx))

        # Composite light layer
        composite_light_layer(frame, light_canvas, show_glow=True)

        slides_frames.append(frame)

    # Stitch into a vertical or horizontal triptych
    # Resize to 960x540 each and stack vertically
    scaled_slides = [cv2.resize(f, (854, 480)) for f in slides_frames]
    triptych = np.vstack(scaled_slides)

    artifact_path = "/Users/shivam/.gemini/antigravity/brain/0eb7146f-8505-4de8-80fa-cfb71f376f92/tutorial_cards_verification.jpg"
    cv2.imwrite(artifact_path, triptych)
    print(f"Saved tutorial cards verification image to {artifact_path}")


if __name__ == "__main__":
    generate_tutorial_verification_artifact()
