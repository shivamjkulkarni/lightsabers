"""Temporal smoothing utilities using Exponential Moving Average (EMA)."""

from typing import Optional, Tuple
from src.geometry import normalize, Point2D, Vector2D


class PointSmoother:
    """Exponential Moving Average filter for 2D points (e.g., wrist pivot)."""

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._current: Optional[Point2D] = None

    def update(self, point: Point2D) -> Point2D:
        """Update with a new raw point and return the smoothed point."""
        if self._current is None:
            self._current = point
            return point

        sx = self.alpha * point[0] + (1.0 - self.alpha) * self._current[0]
        sy = self.alpha * point[1] + (1.0 - self.alpha) * self._current[1]
        self._current = (sx, sy)
        return self._current

    def reset(self) -> None:
        """Reset smoother state."""
        self._current = None


class DirectionSmoother:
    """EMA filter for 2D direction vectors with automatic unit normalization."""

    def __init__(self, alpha: float = 0.60) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._current: Optional[Vector2D] = None

    def update(self, direction: Vector2D) -> Vector2D:
        """Update with a new raw direction and return the normalized smoothed vector."""
        norm_dir = normalize(direction)
        if self._current is None:
            self._current = norm_dir
            return norm_dir

        sx = self.alpha * norm_dir[0] + (1.0 - self.alpha) * self._current[0]
        sy = self.alpha * norm_dir[1] + (1.0 - self.alpha) * self._current[1]
        self._current = normalize((sx, sy))
        return self._current

    def reset(self) -> None:
        """Reset smoother state."""
        self._current = None
