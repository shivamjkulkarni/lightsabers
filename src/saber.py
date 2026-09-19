"""Saber state management and tracking for individual hands."""

import collections
from typing import Deque, Optional, Tuple

from src.geometry import (
    calculate_arm_extended_geometry,
    calculate_saber_direction,
    calculate_saber_endpoint,
    Point2D,
    Vector2D,
)
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

        self.emitter_smoother = PointSmoother(alpha=alpha_pivot)
        self.hilt_start_smoother = PointSmoother(alpha=alpha_pivot)
        self.direction_smoother = DirectionSmoother(alpha=alpha_direction)
        self.trail_history: Deque[TrailPoint] = collections.deque()

        self.current_emitter: Optional[Point2D] = None
        self.current_endpoint: Optional[Point2D] = None
        self.current_hilt_start: Optional[Point2D] = None
        self.current_hilt_end: Optional[Point2D] = None
        self.current_direction: Optional[Vector2D] = None
        self.last_seen_time: float = 0.0
        self.is_active: bool = False

    def update(
        self,
        observation: Optional[HandObservation],
        curr_time: float,
        blade_length: float = 650.0,
        trail_duration: float = 0.38,
    ) -> None:
        """Update saber state from hand observation and expire old trail points."""
        if observation is not None:
            # Use arm-extended geometry if knuckles landmark is available
            if hasattr(observation, "knuckles_center"):
                raw_emitter, _, raw_hstart, raw_hend, raw_dir = calculate_arm_extended_geometry(
                    observation.wrist,
                    observation.knuckles_center,
                    observation.index_tip,
                    blade_length=blade_length,
                )
            else:
                raw_hstart = observation.wrist
                raw_dir = calculate_saber_direction(
                    observation.wrist, observation.index_mcp, observation.index_tip
                )
                raw_emitter = observation.index_mcp
                raw_hend = raw_emitter

            # Temporal smoothing
            smooth_emitter = self.emitter_smoother.update(raw_emitter)
            smooth_hstart = self.hilt_start_smoother.update(raw_hstart)
            smooth_dir = self.direction_smoother.update(raw_dir)

            smooth_endpoint = (
                smooth_emitter[0] + smooth_dir[0] * blade_length,
                smooth_emitter[1] + smooth_dir[1] * blade_length,
            )

            self.current_emitter = smooth_emitter
            self.current_endpoint = smooth_endpoint
            self.current_hilt_start = smooth_hstart
            self.current_hilt_end = smooth_emitter
            self.current_direction = smooth_dir
            self.last_seen_time = curr_time
            self.is_active = True

            # Append to trail history: (emitter, endpoint, timestamp)
            self.trail_history.append((smooth_emitter, smooth_endpoint, curr_time))
        else:
            # Grace period: keep active briefly so saber doesn't snap off upon single frame drop
            if curr_time - self.last_seen_time > self.grace_period:
                self.is_active = False
                self.emitter_smoother.reset()
                self.hilt_start_smoother.reset()
                self.direction_smoother.reset()

        # Expire old trail points
        while self.trail_history and (curr_time - self.trail_history[0][2] > trail_duration):
            self.trail_history.popleft()

    def reset(self) -> None:
        """Clear tracking history and reset state."""
        self.emitter_smoother.reset()
        self.hilt_start_smoother.reset()
        self.direction_smoother.reset()
        self.trail_history.clear()
        self.current_emitter = None
        self.current_endpoint = None
        self.current_hilt_start = None
        self.current_hilt_end = None
        self.current_direction = None
        self.is_active = False
        self.last_seen_time = 0.0
