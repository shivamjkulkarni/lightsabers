"""Geometry helpers for saber direction, endpoint, and vector mathematics."""

import math
from typing import Any, Tuple

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]


def distance(p1: Point2D, p2: Point2D) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def normalize(v: Vector2D) -> Vector2D:
    """Normalize a 2D vector to unit length. Returns (0.0, -1.0) upward if zero vector."""
    norm = math.hypot(v[0], v[1])
    if norm < 1e-6:
        return (0.0, -1.0)
    return (v[0] / norm, v[1] / norm)


def lerp(p1: Point2D, p2: Point2D, t: float) -> Point2D:
    """Linear interpolation between two 2D points."""
    return (
        p1[0] + (p2[0] - p1[0]) * t,
        p1[1] + (p2[1] - p1[1]) * t,
    )


def angle(v: Vector2D) -> float:
    """Angle of 2D vector in radians from positive x-axis."""
    return math.atan2(v[1], v[0])


def scale_length_for_resolution(
    base_length: float,
    current_height: float,
    reference_height: float = 720.0,
) -> float:
    """Scale pixel length proportionally to camera frame height."""
    if reference_height <= 0.0:
        return base_length
    return base_length * (current_height / reference_height)


def calculate_distance_scaled_blade_length(
    hand_size_px: float,
    hand_to_blade_ratio: float = 7.5,
    min_length: float = 180.0,
    max_length: float = 580.0,
) -> float:
    """
    Scale blade length proportionally with perceived hand distance.
    As a combatant steps back, hand_size_px shrinks, realistically
    scaling down the blade without making it look like an oversized pole.
    """
    raw_length = hand_size_px * hand_to_blade_ratio
    return max(min_length, min(max_length, raw_length))


def calculate_saber_direction(
    wrist: Point2D,
    index_mcp: Point2D,
    index_tip: Point2D,
    mcp_weight: float = 0.65,
    tip_weight: float = 0.35,
) -> Vector2D:
    """
    Estimate saber pointing direction outward from hand.
    
    Combines wrist->index_mcp (stable hand orientation) with wrist->index_tip
    (intentional finger pointing) to avoid wild flipping when fingers curl.
    """
    v_mcp = (index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])
    v_tip = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])

    combined_x = mcp_weight * v_mcp[0] + tip_weight * v_tip[0]
    combined_y = mcp_weight * v_mcp[1] + tip_weight * v_tip[1]

    return normalize((combined_x, combined_y))


def calculate_saber_endpoint(
    pivot: Point2D,
    direction: Vector2D,
    blade_length: float,
) -> Point2D:
    """Calculate the tip endpoint of the saber given pivot, unit direction, and length."""
    return (
        pivot[0] + direction[0] * blade_length,
        pivot[1] + direction[1] * blade_length,
    )


def calculate_arm_extended_geometry(
    wrist: Point2D,
    knuckles_center: Point2D,
    index_tip: Point2D,
    blade_length: float = 650.0,
) -> Tuple[Point2D, Point2D, Point2D, Point2D, Vector2D]:
    """
    Calculate lightsaber geometry extending outward from the arm/hand into space.
    
    Returns:
        emitter: Point where the luminous blade starts (just past knuckles/hand)
        endpoint: Tip of the lightsaber blade
        hilt_start: Base of hilt (near wrist)
        hilt_end: Emitter collar of hilt (at knuckles)
        direction: Unit vector pointing along the arm outward into space
    """
    # Vector along forearm through hand
    arm_v = (knuckles_center[0] - wrist[0], knuckles_center[1] - wrist[1])
    hand_len = distance(wrist, knuckles_center)
    if hand_len < 5.0:
        hand_len = 50.0

    # Combine arm axis (70%) and finger direction (30%)
    tip_v = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
    combined_dir = normalize((
        0.70 * arm_v[0] + 0.30 * tip_v[0],
        0.70 * arm_v[1] + 0.30 * tip_v[1],
    ))

    # Hilt rests in the hand between wrist and knuckles
    hilt_start = wrist
    hilt_end = (
        knuckles_center[0] + combined_dir[0] * (hand_len * 0.20),
        knuckles_center[1] + combined_dir[1] * (hand_len * 0.20),
    )

    # Blade emitter begins at the front of the hand/hilt, extending outward
    emitter = hilt_end

    # Endpoint extends into open space
    endpoint = (
        emitter[0] + combined_dir[0] * blade_length,
        emitter[1] + combined_dir[1] * blade_length,
    )

    return emitter, endpoint, hilt_start, hilt_end, combined_dir


def is_saber_active(hand_observation: Any) -> bool:
    """Gesture activation hook for lightsaber."""
    return hand_observation is not None
