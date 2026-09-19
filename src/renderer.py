"""Procedural visual rendering for lightsaber blade, glow, hilt, and motion trail."""

from typing import Deque, List, Tuple
import cv2
import numpy as np

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]
TrailPoint = Tuple[Point2D, Point2D, float]


def render_hilt(
    frame: np.ndarray,
    pivot: Point2D,
    direction: Vector2D,
    hilt_length: float = 50.0,
) -> None:
    """Render a clean procedural cylinder hilt around the wrist."""
    inv_dir = (-direction[0], -direction[1])
    p_start = (int(pivot[0]), int(pivot[1]))
    p_end = (
        int(pivot[0] + inv_dir[0] * hilt_length),
        int(pivot[1] + inv_dir[1] * hilt_length),
    )

    # Dark metallic main grip body
    cv2.line(frame, p_start, p_end, (40, 40, 40), 12, cv2.LINE_AA)

    # Metallic emitter collar at the blade base
    collar_end = (
        int(pivot[0] + inv_dir[0] * (hilt_length * 0.2)),
        int(pivot[1] + inv_dir[1] * (hilt_length * 0.2)),
    )
    cv2.line(frame, p_start, collar_end, (170, 170, 170), 14, cv2.LINE_AA)

    # Pommel cap at the base of the hilt
    pommel_start = (
        int(pivot[0] + inv_dir[0] * (hilt_length * 0.85)),
        int(pivot[1] + inv_dir[1] * (hilt_length * 0.85)),
    )
    cv2.line(frame, pommel_start, p_end, (140, 140, 140), 14, cv2.LINE_AA)

    # Small gold/amber activation stud
    stud_pt = (
        int(pivot[0] + inv_dir[0] * (hilt_length * 0.45)),
        int(pivot[1] + inv_dir[1] * (hilt_length * 0.45)),
    )
    cv2.circle(frame, stud_pt, 3, (0, 200, 255), -1, cv2.LINE_AA)


def render_blade(
    frame: np.ndarray,
    pivot: Point2D,
    endpoint: Point2D,
    color: Tuple[int, int, int],
    show_glow: bool = True,
) -> None:
    """
    Render a multi-layer luminous lightsaber blade:
    1. Half-resolution blurred glow pass composited additively
    2. Saturated outer colored blade
    3. Brighter pastel inner core
    4. White-hot center line and tip highlight
    """
    h, w = frame.shape[:2]
    p_start = (int(pivot[0]), int(pivot[1]))
    p_end = (int(endpoint[0]), int(endpoint[1]))

    # 1. Soft Luminous Glow (computed at half-res for fast 60fps performance)
    if show_glow:
        glow_w = w // 2
        glow_h = h // 2
        scale_x = glow_w / w
        scale_y = glow_h / h

        p_start_s = (int(pivot[0] * scale_x), int(pivot[1] * scale_y))
        p_end_s = (int(endpoint[0] * scale_x), int(endpoint[1] * scale_y))

        small_glow = np.zeros((glow_h, glow_w, 3), dtype=np.uint8)
        # Wide blurred outer glow
        cv2.line(small_glow, p_start_s, p_end_s, color, 20, cv2.LINE_AA)
        cv2.circle(small_glow, p_end_s, 10, color, -1, cv2.LINE_AA)

        blurred = cv2.GaussianBlur(small_glow, (21, 21), 0)
        glow_full = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        np.add(frame, glow_full, out=frame, casting="unsafe")

    # 2. Outer saturated blade
    cv2.line(frame, p_start, p_end, color, 12, cv2.LINE_AA)
    cv2.circle(frame, p_end, 6, color, -1, cv2.LINE_AA)

    # 3. Bright inner core (tinted pastel)
    inner_color = (
        min(255, int(color[0] * 0.35 + 255 * 0.65)),
        min(255, int(color[1] * 0.35 + 255 * 0.65)),
        min(255, int(color[2] * 0.35 + 255 * 0.65)),
    )
    cv2.line(frame, p_start, p_end, inner_color, 6, cv2.LINE_AA)
    cv2.circle(frame, p_end, 3, inner_color, -1, cv2.LINE_AA)

    # 4. White-hot center
    cv2.line(frame, p_start, p_end, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, p_end, 2, (255, 255, 255), -1, cv2.LINE_AA)


def render_trail(
    frame: np.ndarray,
    trail_history: Deque[TrailPoint],
    color: Tuple[int, int, int],
    trail_duration: float,
    curr_time: float,
) -> None:
    """Render fading motion trail ribbon between consecutive historical blade segments."""
    if len(trail_history) < 2:
        return

    trail_overlay = np.zeros_like(frame)
    history_list = list(trail_history)

    for i in range(len(history_list) - 1):
        p1_start, p1_end, t1 = history_list[i]
        p2_start, p2_end, _ = history_list[i + 1]

        age = curr_time - t1
        alpha = max(0.0, min(1.0, 1.0 - (age / trail_duration)))

        poly = np.array([
            [int(p1_start[0]), int(p1_start[1])],
            [int(p1_end[0]), int(p1_end[1])],
            [int(p2_end[0]), int(p2_end[1])],
            [int(p2_start[0]), int(p2_start[1])],
        ], dtype=np.int32)

        # Dimmer trail color with age
        seg_color = (
            int(color[0] * alpha * 0.5),
            int(color[1] * alpha * 0.5),
            int(color[2] * alpha * 0.5),
        )
        cv2.fillPoly(trail_overlay, [poly], seg_color)

    np.add(frame, trail_overlay, out=frame, casting="unsafe")
