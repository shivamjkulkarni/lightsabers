"""Procedural visual rendering for lightsaber blade, glow, hilt, and radiant motion trail."""

import math
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]
TrailPoint = Tuple[Point2D, Point2D, float]


def render_hilt(
    frame: np.ndarray,
    hilt_start: Point2D,
    hilt_end: Point2D,
    direction: Vector2D,
) -> None:
    """Render a clean procedural lightsaber hilt held inside the hand."""
    p_start = (int(hilt_start[0]), int(hilt_start[1]))
    p_end = (int(hilt_end[0]), int(hilt_end[1]))

    hilt_length = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
    if hilt_length < 8.0:
        return

    # Main dark cylindrical grip body
    cv2.line(frame, p_start, p_end, (35, 35, 35), 14, cv2.LINE_AA)

    # Textured grip segments
    num_ribs = 3
    for i in range(1, num_ribs + 1):
        frac = i / (num_ribs + 1)
        rx = int(p_start[0] + frac * (p_end[0] - p_start[0]))
        ry = int(p_start[1] + frac * (p_end[1] - p_start[1]))
        cv2.circle(frame, (rx, ry), 7, (18, 18, 18), 2, cv2.LINE_AA)

    # Metallic pommel cap at base
    pommel_len = min(10.0, hilt_length * 0.2)
    cv2.line(
        frame,
        p_start,
        (int(p_start[0] + direction[0] * pommel_len), int(p_start[1] + direction[1] * pommel_len)),
        (150, 150, 150),
        16,
        cv2.LINE_AA,
    )

    # Silver emitter collar where blade emerges
    collar_len = min(14.0, hilt_length * 0.25)
    collar_start = (
        int(p_end[0] - direction[0] * collar_len),
        int(p_end[1] - direction[1] * collar_len),
    )
    cv2.line(frame, collar_start, p_end, (190, 190, 190), 16, cv2.LINE_AA)

    # Amber activation switch stud
    switch_pt = (
        int(p_start[0] + 0.65 * (p_end[0] - p_start[0])),
        int(p_start[1] + 0.65 * (p_end[1] - p_start[1])),
    )
    cv2.circle(frame, switch_pt, 3, (0, 180, 255), -1, cv2.LINE_AA)


def draw_blade_on_light_canvas(
    light_canvas: np.ndarray,
    emitter: Point2D,
    endpoint: Point2D,
    color: Tuple[int, int, int],
    curr_time: float = 0.0,
) -> None:
    """Draw multi-tier lightsaber plasma core directly onto light canvas."""
    p_start = (int(emitter[0]), int(emitter[1]))
    p_end = (int(endpoint[0]), int(endpoint[1]))

    # Subtle plasma shimmer (micro-oscillations in core energy)
    shimmer = 1.0 + 0.06 * math.sin(curr_time * 28.0)

    # 1. Outer saturated blade
    outer_width = max(8, int(15 * shimmer))
    cv2.line(light_canvas, p_start, p_end, color, outer_width, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, outer_width // 2, color, -1, cv2.LINE_AA)

    # 2. Bright inner pastel core
    inner_color = (
        min(255, int(color[0] * 0.35 + 255 * 0.65)),
        min(255, int(color[1] * 0.35 + 255 * 0.65)),
        min(255, int(color[2] * 0.35 + 255 * 0.65)),
    )
    inner_width = max(4, int(7 * shimmer))
    cv2.line(light_canvas, p_start, p_end, inner_color, inner_width, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, max(2, inner_width // 2), inner_color, -1, cv2.LINE_AA)

    # 3. White-hot center
    cv2.line(light_canvas, p_start, p_end, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 4. Radiant Emitter Flare (corona where blade erupts from hand)
    cv2.circle(light_canvas, p_start, 9, inner_color, -1, cv2.LINE_AA)
    cv2.circle(light_canvas, p_start, 5, (255, 255, 255), -1, cv2.LINE_AA)


def draw_trail_on_light_canvas(
    light_canvas: np.ndarray,
    trail_history: Deque[TrailPoint],
    color: Tuple[int, int, int],
    trail_duration: float,
    curr_time: float,
) -> None:
    """Draw radiant swept ribbons and white tip arcs onto light canvas."""
    if len(trail_history) < 2:
        return

    history_list = list(trail_history)

    for i in range(len(history_list) - 1):
        p1_start, p1_end, t1 = history_list[i]
        p2_start, p2_end, _ = history_list[i + 1]

        age = curr_time - t1
        norm_age = max(0.0, min(1.0, age / trail_duration))
        alpha = (1.0 - norm_age) ** 0.70

        poly = np.array([
            [int(p1_start[0]), int(p1_start[1])],
            [int(p1_end[0]), int(p1_end[1])],
            [int(p2_end[0]), int(p2_end[1])],
            [int(p2_start[0]), int(p2_start[1])],
        ], dtype=np.int32)

        # Vivid trail color
        seg_color = (
            min(255, int(color[0] * alpha * 0.95)),
            min(255, int(color[1] * alpha * 0.95)),
            min(255, int(color[2] * alpha * 0.95)),
        )
        cv2.fillPoly(light_canvas, [poly], seg_color)

        # Historical blade stroke
        blade_alpha_color = (
            min(255, int(color[0] * alpha * 0.8)),
            min(255, int(color[1] * alpha * 0.8)),
            min(255, int(color[2] * alpha * 0.8)),
        )
        cv2.line(
            light_canvas,
            (int(p1_start[0]), int(p1_start[1])),
            (int(p1_end[0]), int(p1_end[1])),
            blade_alpha_color,
            max(1, int(4 * alpha)),
            cv2.LINE_AA,
        )

        # White-hot trailing arc along blade tip's path
        pt1 = (int(p1_end[0]), int(p1_end[1]))
        pt2 = (int(p2_end[0]), int(p2_end[1]))
        white_val = min(255, int(255 * alpha))
        cv2.line(light_canvas, pt1, pt2, (white_val, white_val, white_val), max(1, int(3 * alpha)), cv2.LINE_AA)


def composite_light_layer(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    show_glow: bool = True,
) -> None:
    """
    Generate bloom from light canvas and additively composite onto frame.
    Uses saturating cv2.add to eliminate color clipping/overflow.
    """
    h, w = frame.shape[:2]

    if show_glow:
        # Half-resolution bloom for fast 60+ FPS
        glow_w = w // 2
        glow_h = h // 2
        small_light = cv2.resize(light_canvas, (glow_w, glow_h), interpolation=cv2.INTER_AREA)

        # Wide soft optical haze
        blurred = cv2.GaussianBlur(small_light, (29, 29), 0)
        bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add bloom into light layer
        cv2.add(light_canvas, bloom, dst=light_canvas)

    # Saturating addition directly onto the frame
    cv2.add(frame, light_canvas, dst=frame)


def render_blade(
    frame: np.ndarray,
    emitter: Point2D,
    endpoint: Point2D,
    color: Tuple[int, int, int],
    show_glow: bool = True,
    curr_time: float = 0.0,
) -> None:
    """Convenience helper: render a single blade directly onto frame."""
    light = np.zeros_like(frame)
    draw_blade_on_light_canvas(light, emitter, endpoint, color, curr_time)
    composite_light_layer(frame, light, show_glow=show_glow)


def render_trail(
    frame: np.ndarray,
    trail_history: Deque[TrailPoint],
    color: Tuple[int, int, int],
    trail_duration: float,
    curr_time: float,
    show_glow: bool = True,
) -> None:
    """Convenience helper: render a trail directly onto frame."""
    light = np.zeros_like(frame)
    draw_trail_on_light_canvas(light, trail_history, color, trail_duration, curr_time)
    composite_light_layer(frame, light, show_glow=show_glow)
