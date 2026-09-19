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
        # Add subtle sci-fi grid lines on frame
        for gy in range(0, height, 60):
            cv2.line(frame, (0, gy), (width, gy), (22, 26, 34), 1)
        for gx in range(0, width, 60):
            cv2.line(frame, (gx, 0), (gx, height), (22, 26, 34), 1)
        light_canvas = canvas_buffer.reset_and_get(frame.shape)

        # Draw the visual tutorial card
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

