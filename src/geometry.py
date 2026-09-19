"""Geometry helpers for saber direction, endpoint, and vector mathematics."""

import math
from typing import Tuple

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
