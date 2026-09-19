"""Saber state management and tracking for individual hands."""

import collections
from typing import Deque, Optional, Tuple

from src.geometry import calculate_saber_direction, calculate_saber_endpoint, Point2D, Vector2D
from src.hand_tracker import HandObservation
from src.smoothing import DirectionSmoother, PointSmoother

TrailPoint = Tuple[Point2D, Point2D, float]


class SaberInstance:
    """Manages tracking, smoothing, and trail history for a single hand's lightsaber."""

    def __init__(
        self,
        handedness: str,
        color: Tuple[int, int, int] = (255, 120, 30),
        alpha_pivot: float = 0.65,
        alpha_direction: float = 0.60,
        grace_period: float = 0.25,
    ) -> None:
        self.handedness = handedness
        self.color = color
        self.grace_period = grace_period

        self.pivot_smoother = PointSmoother(alpha=alpha_pivot)
        self.direction_smoother = DirectionSmoother(alpha=alpha_direction)
        self.trail_history: Deque[TrailPoint] = collections.deque()

        self.current_pivot: Optional[Point2D] = None
        self.current_endpoint: Optional[Point2D] = None
        self.current_direction: Optional[Vector2D] = None
        self.last_seen_time: float = 0.0
        self.is_active: bool = False

    def update(
        self,
        observation: Optional[HandObservation],
        curr_time: float,
        blade_length: float = 320.0,
        trail_duration: float = 0.35,
    ) -> None:
        """Update saber state from an optional hand observation and expire old trail points."""
        if observation is not None:
            smooth_pivot = self.pivot_smoother.update(observation.wrist)
            raw_dir = calculate_saber_direction(
                observation.wrist, observation.index_mcp, observation.index_tip
            )
            smooth_dir = self.direction_smoother.update(raw_dir)
            endpoint = calculate_saber_endpoint(smooth_pivot, smooth_dir, blade_length)

            self.current_pivot = smooth_pivot
            self.current_endpoint = endpoint
            self.current_direction = smooth_dir
            self.last_seen_time = curr_time
            self.is_active = True

            # Append to trail history
            self.trail_history.append((smooth_pivot, endpoint, curr_time))
        else:
            # Grace period: keep active briefly so saber doesn't snap off upon single frame drop
            if curr_time - self.last_seen_time > self.grace_period:
                self.is_active = False
                self.pivot_smoother.reset()
                self.direction_smoother.reset()

        # Expire old trail points
        while self.trail_history and (curr_time - self.trail_history[0][2] > trail_duration):
            self.trail_history.popleft()

    def reset(self) -> None:
        """Clear tracking history and reset state."""
        self.pivot_smoother.reset()
        self.direction_smoother.reset()
        self.trail_history.clear()
        self.current_pivot = None
        self.current_endpoint = None
        self.current_direction = None
        self.is_active = False
        self.last_seen_time = 0.0
