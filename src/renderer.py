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
    scale_factor: float = 1.0,
) -> None:
    """Render a clean procedural lightsaber hilt held inside the hand."""
    p_start = (int(hilt_start[0]), int(hilt_start[1]))
    p_end = (int(hilt_end[0]), int(hilt_end[1]))

    hilt_length = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
    if hilt_length < 6.0:
        return

    # Scale hilt thickness with distance
    grip_width = max(6, int(14 * scale_factor))
    cap_width = max(7, int(16 * scale_factor))

    # Main dark cylindrical grip body
    cv2.line(frame, p_start, p_end, (35, 35, 35), grip_width, cv2.LINE_AA)

    # Textured grip segments
    num_ribs = 3
    for i in range(1, num_ribs + 1):
        frac = i / (num_ribs + 1)
        rx = int(p_start[0] + frac * (p_end[0] - p_start[0]))
        ry = int(p_start[1] + frac * (p_end[1] - p_start[1]))
        cv2.circle(frame, (rx, ry), max(3, int(7 * scale_factor)), (18, 18, 18), max(1, int(2 * scale_factor)), cv2.LINE_AA)

    # Metallic pommel cap at base
    pommel_len = min(10.0, hilt_length * 0.2)
    cv2.line(
        frame,
        p_start,
        (int(p_start[0] + direction[0] * pommel_len), int(p_start[1] + direction[1] * pommel_len)),
        (150, 150, 150),
        cap_width,
        cv2.LINE_AA,
    )

    # Silver emitter collar where blade emerges
    collar_len = min(14.0, hilt_length * 0.25)
    collar_start = (
        int(p_end[0] - direction[0] * collar_len),
        int(p_end[1] - direction[1] * collar_len),
    )
    cv2.line(frame, collar_start, p_end, (190, 190, 190), cap_width, cv2.LINE_AA)

    # Amber activation switch stud
    switch_pt = (
        int(p_start[0] + 0.65 * (p_end[0] - p_start[0])),
        int(p_start[1] + 0.65 * (p_end[1] - p_start[1])),
    )
    cv2.circle(frame, switch_pt, max(2, int(3 * scale_factor)), (0, 180, 255), -1, cv2.LINE_AA)


def draw_blade_on_light_canvas(
    light_canvas: np.ndarray,
    emitter: Point2D,
    endpoint: Point2D,
    color: Tuple[int, int, int],
    curr_time: float = 0.0,
    scale_factor: float = 1.0,
) -> None:
    """Draw multi-tier lightsaber plasma core directly onto light canvas."""
    p_start = (int(emitter[0]), int(emitter[1]))
    p_end = (int(endpoint[0]), int(endpoint[1]))

    # Subtle plasma shimmer (micro-oscillations in core energy)
    shimmer = 1.0 + 0.06 * math.sin(curr_time * 28.0)

    # 1. Outer saturated blade (scales with distance)
    outer_width = max(5, int(14 * scale_factor * shimmer))
    cv2.line(light_canvas, p_start, p_end, color, outer_width, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, outer_width // 2, color, -1, cv2.LINE_AA)

    # 2. Bright inner pastel core
    inner_color = (
        min(255, int(color[0] * 0.35 + 255 * 0.65)),
        min(255, int(color[1] * 0.35 + 255 * 0.65)),
        min(255, int(color[2] * 0.35 + 255 * 0.65)),
    )
    inner_width = max(3, int(6 * scale_factor * shimmer))
    cv2.line(light_canvas, p_start, p_end, inner_color, inner_width, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, max(2, inner_width // 2), inner_color, -1, cv2.LINE_AA)

    # 3. White-hot center
    center_width = max(1, int(2.5 * scale_factor))
    cv2.line(light_canvas, p_start, p_end, (255, 255, 255), center_width, cv2.LINE_AA)
    cv2.circle(light_canvas, p_end, center_width, (255, 255, 255), -1, cv2.LINE_AA)

    # 4. Radiant Emitter Flare (corona where blade erupts from hand)
    flare_radius = max(3, int(8 * scale_factor))
    cv2.circle(light_canvas, p_start, flare_radius, inner_color, -1, cv2.LINE_AA)
    cv2.circle(light_canvas, p_start, max(2, flare_radius // 2), (255, 255, 255), -1, cv2.LINE_AA)


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
        # Steeper exponent makes trail fade significantly faster
        alpha = (1.0 - norm_age) ** 1.35

        poly = np.array([
            [int(p1_start[0]), int(p1_start[1])],
            [int(p1_end[0]), int(p1_end[1])],
            [int(p2_end[0]), int(p2_end[1])],
            [int(p2_start[0]), int(p2_start[1])],
        ], dtype=np.int32)

        # 20% reduced max opacity (0.75 instead of 0.95)
        seg_color = (
            min(255, int(color[0] * alpha * 0.75)),
            min(255, int(color[1] * alpha * 0.75)),
            min(255, int(color[2] * alpha * 0.75)),
        )
        cv2.fillPoly(light_canvas, [poly], seg_color)

        # Historical blade stroke
        blade_alpha_color = (
            min(255, int(color[0] * alpha * 0.65)),
            min(255, int(color[1] * alpha * 0.65)),
            min(255, int(color[2] * alpha * 0.65)),
        )
        cv2.line(
            light_canvas,
            (int(p1_start[0]), int(p1_start[1])),
            (int(p1_end[0]), int(p1_end[1])),
            blade_alpha_color,
            max(1, int(3 * alpha)),
            cv2.LINE_AA,
        )

        # White-hot trailing arc along blade tip's path
        pt1 = (int(p1_end[0]), int(p1_end[1]))
        pt2 = (int(p2_end[0]), int(p2_end[1]))
        white_val = min(255, int(210 * alpha))
        cv2.line(light_canvas, pt1, pt2, (white_val, white_val, white_val), max(1, int(2 * alpha)), cv2.LINE_AA)


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

        # Fast 21x21 Gaussian blur for low latency soft haze
        blurred = cv2.GaussianBlur(small_light, (21, 21), 0)
        bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add bloom into light layer
        cv2.add(light_canvas, bloom, dst=light_canvas)

    # Saturating addition directly onto the frame
    cv2.add(frame, light_canvas, dst=frame)


class CanvasBuffer:
    """Reusable pre-allocated light canvas buffer to eliminate GC and allocation latency."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def reset_and_get(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Return zeroed canvas matching target frame dimensions."""
        h, w = shape[:2]
        if self.canvas.shape[0] != h or self.canvas.shape[1] != w:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            self.canvas.fill(0)
        return self.canvas


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


def render_side_chamber(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    box_rect: Tuple[int, int, int, int],
    title: str,
    prompt: str,
    border_color: Tuple[int, int, int],
    glow_color: Tuple[int, int, int],
    has_hand_inside: bool,
    curr_time: float,
) -> None:
    """
    Draw a stylized sci-fi holographic Ignition Chamber on either the Left or Right side.
    """
    bx1, by1, bx2, by2 = box_rect
    box_w = bx2 - bx1
    box_h = by2 - by1

    active_border = border_color
    active_glow = glow_color

    if has_hand_inside:
        pulse = 0.5 + 0.5 * math.sin(curr_time * 8.0)
        c_val = int(200 + 55 * pulse)
        active_border = (c_val, c_val, c_val)
        active_glow = (255, 255, 255)

    # Base border
    dim_color = (
        int(active_border[0] * 0.35),
        int(active_border[1] * 0.35),
        int(active_border[2] * 0.35),
    )
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), dim_color, 1, cv2.LINE_AA)

    # Corner brackets
    corner_len = min(36, box_w // 5, box_h // 5)
    thickness = 2 if not has_hand_inside else 3

    # Top-Left
    cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), active_border, thickness, cv2.LINE_AA)
    cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), active_border, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), active_border, thickness, cv2.LINE_AA)
    cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), active_border, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), active_border, thickness, cv2.LINE_AA)
    cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), active_border, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), active_border, thickness, cv2.LINE_AA)
    cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), border_color, thickness, cv2.LINE_AA)

    # Glowing bloom on light_canvas
    cv2.line(light_canvas, (bx1, by1), (bx1 + corner_len, by1), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx1, by1), (bx1, by1 + corner_len), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx2, by1), (bx2 - corner_len, by1), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx2, by1), (bx2, by1 + corner_len), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx1, by2), (bx1 + corner_len, by2), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx1, by2), (bx1, by2 - corner_len), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx2, by2), (bx2 - corner_len, by2), active_glow, thickness + 3, cv2.LINE_AA)
    cv2.line(light_canvas, (bx2, by2), (bx2, by2 - corner_len), active_glow, thickness + 3, cv2.LINE_AA)

    # Header title
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    tx = bx1 + (box_w - tw) // 2
    ty = max(22, by1 - 10)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, active_border, 2, cv2.LINE_AA)

    # Contextual prompt below
    sub = prompt if not has_hand_inside else "HAND DETECTED - FORM POSE!"
    sub_color = (0, 255, 255) if has_hand_inside else (200, 200, 200)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    sx = bx1 + (box_w - sw) // 2
    sy = by2 + 24
    cv2.putText(frame, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sub_color, 1, cv2.LINE_AA)


def render_match_winner_screen(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    winner_name: str,
    winner_color: Tuple[int, int, int],
    score_blue: int,
    score_red: int,
    curr_time: float,
) -> None:
    """
    Render a dramatic, persistent Victory Screen when a 3-round match concludes.
    Stays on screen until user presses 'R' to rematch or 'Q' to quit.
    """
    h, w = frame.shape[:2]

    # Dark translucent backdrop overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Frosted center card
    cw, ch = int(w * 0.80), int(h * 0.62)
    cx1 = (w - cw) // 2
    cy1 = (h - ch) // 2
    cx2 = cx1 + cw
    cy2 = cy1 + ch

    scale = max(0.48, min(1.1, w / 1280.0))

    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (25, 25, 35), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), winner_color, max(2, int(3 * scale)), cv2.LINE_AA)
    cv2.rectangle(light_canvas, (cx1, cy1), (cx2, cy2), winner_color, max(3, int(6 * scale)), cv2.LINE_AA)

    # 1. Victory Title (ASCII-safe for OpenCV)
    title = f"*** {winner_name.upper()} WINS THE MATCH! ***"
    t_scale = 0.85 * scale
    t_thick = max(2, int(3 * scale))
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_thick)
    tx = (w - tw) // 2
    ty = cy1 + int(ch * 0.28)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (0, 0, 0), t_thick + 3, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, winner_color, t_thick, cv2.LINE_AA)
    cv2.putText(light_canvas, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (255, 255, 255), t_thick, cv2.LINE_AA)

    # 2. Final Scoreboard
    score_text = f"FINAL SCORE:  JEDI BLUE  {score_blue}  -  {score_red}  SITH RED"
    s_scale = 0.68 * scale
    s_thick = max(1, int(2 * scale))
    (sw, sh), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, s_scale, s_thick)
    sx = (w - sw) // 2
    sy = cy1 + int(ch * 0.52)
    cv2.putText(frame, score_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (0, 0, 0), s_thick + 3, cv2.LINE_AA)
    cv2.putText(frame, score_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (0, 240, 255), s_thick, cv2.LINE_AA)

    # 3. Match format subtitle
    sub_text = "BEST OF 3 ROUNDS COMPLETED"
    m_scale = 0.45 * scale
    (mw, mh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, m_scale, 1)
    mx = (w - mw) // 2
    my = cy1 + int(ch * 0.68)
    cv2.putText(frame, sub_text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, m_scale, (180, 180, 180), 1, cv2.LINE_AA)

    # 4. Pulsing Rematch / Quit prompt
    pulse = 0.5 + 0.5 * math.sin(curr_time * 5.0)
    prompt_color = (int(160 + 95 * pulse), 255, int(160 + 95 * pulse))
    prompt = "PRESS  [R]  FOR REMATCH   |   PRESS  [Q]  OR  [ESC]  TO QUIT"
    p_scale = 0.52 * scale
    p_thick = max(1, int(2 * scale))
    (pw, ph), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, p_scale, p_thick)
    px = (w - pw) // 2
    py = cy2 - int(ch * 0.12)
    cv2.putText(frame, prompt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, p_scale, (0, 0, 0), p_thick + 3, cv2.LINE_AA)
    cv2.putText(frame, prompt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, p_scale, prompt_color, p_thick, cv2.LINE_AA)


def _draw_diagram_blade(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    hilt_len: int = 24,
    blade_thickness: int = 4,
) -> None:
    """Draw a stylized vector lightsaber with hilt, luminous blade, and white core."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 8.0:
        return
    ux, uy = dx / length, dy / length

    # Metallic hilt
    h_end = (int(start[0] + ux * hilt_len), int(start[1] + uy * hilt_len))
    cv2.line(frame, start, h_end, (35, 35, 40), 7, cv2.LINE_AA)
    cv2.line(frame, start, h_end, (150, 150, 155), 3, cv2.LINE_AA)

    # Glowing blade on light canvas (bloom)
    b_start = h_end
    cv2.line(light_canvas, b_start, end, color, blade_thickness * 3, cv2.LINE_AA)
    cv2.line(light_canvas, b_start, end, (255, 255, 255), blade_thickness, cv2.LINE_AA)

    # Core on frame
    cv2.line(frame, b_start, end, color, blade_thickness + 2, cv2.LINE_AA)
    cv2.line(frame, b_start, end, (255, 255, 255), max(1, blade_thickness // 2), cv2.LINE_AA)


def _draw_diagram_spark_burst(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    center: Tuple[int, int],
    color: Tuple[int, int, int],
    count: int = 16,
    radius: int = 22,
) -> None:
    """Draw a glowing clash spark burst at contact points."""
    cv2.circle(light_canvas, center, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
    for i in range(count):
        ang = (i / count) * 2 * math.pi
        r = radius * (0.6 + 0.4 * math.sin(i * 1.9))
        sx = int(center[0] + math.cos(ang) * r)
        sy = int(center[1] + math.sin(ang) * r)
        cv2.line(light_canvas, center, (sx, sy), color, 2, cv2.LINE_AA)
        cv2.circle(light_canvas, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_badge(
    frame: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    scale: float = 0.45,
    pad_x: int = 10,
    pad_y: int = 5,
    border_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw a compact pill badge with centered text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    bx1 = cx - tw // 2 - pad_x
    by1 = cy - th // 2 - pad_y
    bx2 = cx + tw // 2 + pad_x
    by2 = cy + th // 2 + pad_y
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1, cv2.LINE_AA)
    b_col = border_color if border_color else text_color
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), b_col, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def render_tutorial_card(
    frame: np.ndarray,
    light_canvas: np.ndarray,
    slide_index: int,
    curr_time: float,
) -> None:
    """
    Render a purely visual holographic training card over the live camera feed.
    
    Demonstrates successful combat moves and parries with zero unnecessary text walls:
      Card 1: MOVE 1 — SOLID FORTE BLOCK (Safe Forte vs Weak Tip)
      Card 2: MOVE 2 — ACTIVE PERFECT PARRY DEFLECTION (Shockwave + Counter)
      Card 3: MOVE 3 — POISE BREAK & DECISIVE DISARM (Round Victory)
    """
    h, w = frame.shape[:2]
    scale = max(0.48, min(1.1, w / 1280.0))

    card_w = int(w * 0.82)
    card_h = int(h * 0.78)
    cx1 = (w - card_w) // 2
    cy1 = (h - card_h) // 2
    cx2 = cx1 + card_w
    cy2 = cy1 + card_h

    # 1. Frosted dark sci-fi background overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # 2. Glowing sci-fi border & corner brackets
    border_color = (255, 190, 40)
    glow_color = (255, 140, 20)
    b_thick = max(1, int(2 * scale))
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), border_color, b_thick, cv2.LINE_AA)
    cv2.rectangle(light_canvas, (cx1, cy1), (cx2, cy2), glow_color, b_thick + 3, cv2.LINE_AA)

    c_len = min(40, int(card_w * 0.05))
    for bx, by, dx, dy in [
        (cx1, cy1, 1, 1),
        (cx2, cy1, -1, 1),
        (cx1, cy2, 1, -1),
        (cx2, cy2, -1, -1),
    ]:
        cv2.line(frame, (bx, by), (bx + dx * c_len, by), (0, 255, 255), b_thick + 2, cv2.LINE_AA)
        cv2.line(frame, (bx, by), (bx, by + dy * c_len), (0, 255, 255), b_thick + 2, cv2.LINE_AA)
        cv2.line(light_canvas, (bx, by), (bx + dx * c_len, by), (0, 200, 255), b_thick + 5, cv2.LINE_AA)
        cv2.line(light_canvas, (bx, by), (bx, by + dy * c_len), (0, 200, 255), b_thick + 5, cv2.LINE_AA)

    # Header Titles for pure visual moves
    headers = {
        1: ("MOVE 1: SOLID FORTE BLOCK", "BLOCK WITH LOWER 70% OF BLADE TO DEFEND GUARD POISE", (0, 255, 180)),
        2: ("MOVE 2: ACTIVE DEFLECTION PARRY", "SNAP BLADE INTO INCOMING STRIKE TO RESTORE POISE & COUNTER", (0, 230, 255)),
        3: ("MOVE 3: POISE BREAK & DISARM", "DEPLETE OPPONENT'S POISE TO KNOCK THEIR SABER FLYING & WIN", (255, 140, 0)),
    }
    title, subtitle, accent_color = headers.get(slide_index, headers[1])

    # Header Title
    t_scale = 0.66 * scale
    t_thick = max(1, int(2 * scale))
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_thick)
    tx = (w - tw) // 2
    ty = cy1 + int(card_h * 0.10)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (0, 0, 0), t_thick + 3, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, accent_color, t_thick, cv2.LINE_AA)
    cv2.putText(light_canvas, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, t_scale, accent_color, t_thick + 1, cv2.LINE_AA)

    # Subtitle
    sub_scale = 0.42 * scale
    (sub_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    cv2.putText(frame, subtitle, ((w - sub_w) // 2, ty + int(card_h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 210, 230), 1, cv2.LINE_AA)

    # Separator line
    div_y = ty + int(card_h * 0.095)
    cv2.line(frame, (cx1 + 30, div_y), (cx2 - 30, div_y), (70, 85, 110), 1, cv2.LINE_AA)

    jedi_blue = (255, 90, 20)
    sith_red = (30, 30, 255)

    # -----------------------------------------------------------------
    # VISUAL MOVE DIAGRAMS
    # -----------------------------------------------------------------
    if slide_index == 1:
        # -------------------------------------------------------------
        # CARD 1: FORTE BLOCK VS WEAK TIP (Side-by-side Visual Diagrams)
        # -------------------------------------------------------------
        mid_panel_x = (cx1 + cx2) // 2
        p1_cx = cx1 + int(card_w * 0.25)
        p2_cx = cx1 + int(card_w * 0.75)

        # Panel dividers
        cv2.line(frame, (mid_panel_x, div_y + 10), (mid_panel_x, cy2 - int(card_h * 0.16)), (50, 65, 85), 1, cv2.LINE_AA)

        # --- LEFT PANEL: SOLID FORTE BLOCK ---
        cv2.putText(frame, "[ SOLID FORTE BLOCK ]", (p1_cx - 90, div_y + int(card_h * 0.07)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (0, 255, 160), 2, cv2.LINE_AA)

        # Blue blade (defender angled)
        b_base = (p1_cx - 85, cy1 + int(card_h * 0.58))
        b_tip = (p1_cx + 70, cy1 + int(card_h * 0.30))
        _draw_diagram_blade(frame, light_canvas, b_base, b_tip, jedi_blue, hilt_len=24, blade_thickness=4)

        # Forte safe zone highlight (lower 70% in green)
        f_end = (int(b_base[0] + 0.70 * (b_tip[0] - b_base[0])), int(b_base[1] + 0.70 * (b_tip[1] - b_base[1])))
        cv2.line(frame, (b_base[0] - 8, b_base[1] + 8), (f_end[0] - 8, f_end[1] + 8), (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "FORTE (LOWER 70%)", (b_base[0] - 10, b_base[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (0, 255, 100), 1, cv2.LINE_AA)

        # Red blade striking down into forte
        r_base = (p1_cx - 20, cy1 + int(card_h * 0.22))
        r_tip = (p1_cx + 15, cy1 + int(card_h * 0.50))
        _draw_diagram_blade(frame, light_canvas, r_base, r_tip, sith_red, hilt_len=24, blade_thickness=4)

        # Spark clash burst at contact
        clash_p1 = (p1_cx + 2, cy1 + int(card_h * 0.43))
        _draw_diagram_spark_burst(frame, light_canvas, clash_p1, (100, 255, 255), count=18, radius=22)

        # Badges
        _draw_badge(frame, "SAFE BLOCK: NO DAMAGE", p1_cx, cy1 + int(card_h * 0.67), (20, 80, 30), (100, 255, 140), scale=0.44 * scale)
        cv2.putText(frame, "GUARD: [ + + ] (SECURE)", (p1_cx - 75, cy1 + int(card_h * 0.74)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (0, 255, 180), 1, cv2.LINE_AA)

        # --- RIGHT PANEL: WEAK TIP CONTACT ---
        cv2.putText(frame, "[ WEAK TIP CONTACT ]", (p2_cx - 85, div_y + int(card_h * 0.07)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (60, 80, 255), 2, cv2.LINE_AA)

        # Blue blade
        b2_base = (p2_cx - 80, cy1 + int(card_h * 0.58))
        b2_tip = (p2_cx + 50, cy1 + int(card_h * 0.32))
        _draw_diagram_blade(frame, light_canvas, b2_base, b2_tip, jedi_blue, hilt_len=24, blade_thickness=4)

        # Tip weak zone highlight (upper 30% in red)
        t_start = (int(b2_base[0] + 0.70 * (b2_tip[0] - b2_base[0])), int(b2_base[1] + 0.70 * (b2_tip[1] - b2_base[1])))
        cv2.line(frame, (t_start[0] + 8, t_start[1] - 8), (b2_tip[0] + 8, b2_tip[1] - 8), (40, 40, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "WEAK TIP (FOIBLE)", (b2_tip[0] - 40, b2_tip[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (60, 60, 255), 1, cv2.LINE_AA)

        # Red blade striking only the tip
        r2_base = (p2_cx + 110, cy1 + int(card_h * 0.24))
        r2_tip = (p2_cx + 40, cy1 + int(card_h * 0.35))
        _draw_diagram_blade(frame, light_canvas, r2_base, r2_tip, sith_red, hilt_len=24, blade_thickness=4)

        # Small red spark crackle
        clash_p2 = (p2_cx + 45, cy1 + int(card_h * 0.33))
        _draw_diagram_spark_burst(frame, light_canvas, clash_p2, (50, 80, 255), count=10, radius=14)

        # Badges
        _draw_badge(frame, "GUARD SHAKEN: -1 POISE", p2_cx, cy1 + int(card_h * 0.67), (20, 20, 90), (60, 100, 255), scale=0.44 * scale)
        cv2.putText(frame, "GUARD: [ + - ] (DAMAGED)", (p2_cx - 80, cy1 + int(card_h * 0.74)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (50, 120, 255), 1, cv2.LINE_AA)

    elif slide_index == 2:
        # -------------------------------------------------------------
        # CARD 2: ACTIVE PERFECT PARRY DEFLECTION
        # -------------------------------------------------------------
        pc_x = (cx1 + cx2) // 2

        # Attacker's incoming red blade
        r_base = (pc_x + 160, cy1 + int(card_h * 0.24))
        r_tip = (pc_x - 30, cy1 + int(card_h * 0.52))
        _draw_diagram_blade(frame, light_canvas, r_base, r_tip, sith_red, hilt_len=26, blade_thickness=5)
        cv2.putText(frame, "INCOMING STRIKE", (r_base[0] - 40, r_base[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (80, 80, 255), 1, cv2.LINE_AA)

        # Defender's blue blade snapping actively into strike
        b_base = (pc_x - 140, cy1 + int(card_h * 0.60))
        b_tip = (pc_x + 40, cy1 + int(card_h * 0.32))
        _draw_diagram_blade(frame, light_canvas, b_base, b_tip, jedi_blue, hilt_len=26, blade_thickness=5)

        # Active snap motion curve with arrow
        arc_pts = [
            (b_base[0] - 20, b_base[1] - 10),
            (pc_x - 70, cy1 + int(card_h * 0.52)),
            (pc_x - 10, cy1 + int(card_h * 0.42)),
        ]
        cv2.polylines(frame, [np.array(arc_pts, dtype=np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, arc_pts[-2], arc_pts[-1], (0, 255, 255), 3, tipLength=0.35)
        cv2.putText(frame, "ACTIVE SWEEP SNAP", (b_base[0] - 35, b_base[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (0, 230, 255), 1, cv2.LINE_AA)

        # Collision Point with Supernova Shockwave Rings
        clash_pt = (pc_x + 5, cy1 + int(card_h * 0.42))
        cv2.circle(light_canvas, clash_pt, 42, (60, 220, 255), 3, cv2.LINE_AA)
        cv2.circle(light_canvas, clash_pt, 24, (140, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, clash_pt, 30, (200, 255, 255), 2, cv2.LINE_AA)
        _draw_diagram_spark_burst(frame, light_canvas, clash_pt, (255, 255, 120), count=28, radius=36)

        # Badges & Counter-Strike Bonus
        _draw_badge(frame, "PERFECT PARRY: POISE RESTORED (+1) [ + + ]", pc_x, cy1 + int(card_h * 0.66), (10, 60, 40), (0, 255, 160), scale=0.46 * scale)
        _draw_badge(frame, ">> COUNTER-STRIKE ADVANTAGE: +35% BLADE SPEED READY <<", pc_x, cy1 + int(card_h * 0.73), (60, 50, 10), (0, 235, 255), scale=0.46 * scale, border_color=(0, 215, 255))

    elif slide_index == 3:
        # -------------------------------------------------------------
        # CARD 3: POISE BREAK & DECISIVE DISARM
        # -------------------------------------------------------------
        pc_x = (cx1 + cx2) // 2

        # Poise shattered display
        cv2.putText(frame, "GUARD POISE DEPLETED: [ - - ]", (pc_x - 130, div_y + int(card_h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * scale, (40, 40, 255), 2, cv2.LINE_AA)

        # Victorious red blade delivering finishing sweep
        r_base = (pc_x - 150, cy1 + int(card_h * 0.50))
        r_tip = (pc_x + 20, cy1 + int(card_h * 0.38))
        _draw_diagram_blade(frame, light_canvas, r_base, r_tip, sith_red, hilt_len=26, blade_thickness=5)
        cv2.putText(frame, "VICTORIOUS FINISHER", (r_base[0] - 20, r_base[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (50, 50, 255), 1, cv2.LINE_AA)

        # Tumbling flying saber hilt (disarmed)
        tumble_arc = [
            (pc_x + 10, cy1 + int(card_h * 0.42)),
            (pc_x + 80, cy1 + int(card_h * 0.28)),
            (pc_x + 150, cy1 + int(card_h * 0.44)),
            (pc_x + 190, cy1 + int(card_h * 0.60)),
        ]
        cv2.polylines(frame, [np.array(tumble_arc, dtype=np.int32)], False, (140, 170, 200), 1, cv2.LINE_AA)

        # Saber hilt spinning at peak
        peak_pt = tumble_arc[1]
        cv2.line(frame, (peak_pt[0] - 18, peak_pt[1] - 10), (peak_pt[0] + 18, peak_pt[1] + 10), (60, 60, 65), 6, cv2.LINE_AA)
        cv2.line(frame, (peak_pt[0] - 18, peak_pt[1] - 10), (peak_pt[0] + 18, peak_pt[1] + 10), (180, 180, 185), 2, cv2.LINE_AA)
        # Spin circle
        cv2.circle(frame, peak_pt, 22, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SABER FLIES LOOSE!", (peak_pt[0] - 55, peak_pt[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (100, 220, 255), 1, cv2.LINE_AA)

        # Landing bounce sparks at floor
        landing_pt = tumble_arc[-1]
        cv2.line(frame, (landing_pt[0] - 40, landing_pt[1]), (landing_pt[0] + 40, landing_pt[1]), (80, 100, 130), 2, cv2.LINE_AA)
        _draw_diagram_spark_burst(frame, light_canvas, landing_pt, (255, 180, 50), count=14, radius=18)
        cv2.putText(frame, "FLOOR BOUNCE", (landing_pt[0] - 40, landing_pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (160, 190, 210), 1, cv2.LINE_AA)

        # Victory Badges
        _draw_badge(frame, "DECISIVE DISARM: ROUND WON!", pc_x, cy1 + int(card_h * 0.67), (15, 45, 80), (255, 215, 0), scale=0.50 * scale, border_color=(255, 215, 0))
        _draw_badge(frame, "BEST OF 3 ROUNDS TO TAKE THE MATCH", pc_x, cy1 + int(card_h * 0.74), (20, 25, 35), (200, 220, 240), scale=0.42 * scale)

    # 4. Slide Pagination Indicator
    dots_y = cy2 - int(card_h * 0.10)
    dots_str = f"<<   SLIDE  {slide_index}  OF  3   >>"
    (dw, _), _ = cv2.getTextSize(dots_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, 1)
    cv2.putText(frame, dots_str, ((w - dw) // 2, dots_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (140, 170, 200), 1, cv2.LINE_AA)

    # 5. Pulsing Nav Prompt
    pulse = 0.5 + 0.5 * math.sin(curr_time * 5.0)
    prompt_col = (int(160 + 95 * pulse), 255, int(160 + 95 * pulse))

    if slide_index < 3:
        nav_text = "PRESS  [SPACE] / [ENTER]  FOR NEXT MOVE    |    PRESS  [S]  TO SKIP TUTORIAL"
    else:
        nav_text = "PRESS  [SPACE] / [ENTER]  TO COMMENCE DUEL    |    PRESS  [S]  TO SKIP TUTORIAL"

    p_scale = 0.48 * scale
    p_thick = max(1, int(2 * scale))
    (nw, _), _ = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, p_scale, p_thick)
    nx = (w - nw) // 2
    ny = cy2 - int(card_h * 0.04)
    cv2.putText(frame, nav_text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, p_scale, (0, 0, 0), p_thick + 3, cv2.LINE_AA)
    cv2.putText(frame, nav_text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, p_scale, prompt_col, p_thick, cv2.LINE_AA)
