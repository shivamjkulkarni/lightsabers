"""Saber state management, velocity tracking, recoil, and disarm cooldown."""

import collections
import math
from typing import Deque, Optional, Tuple

from src.geometry import (
    calculate_arm_extended_geometry,
    calculate_saber_direction,
    Point2D,
    Vector2D,
)
from src.hand_tracker import HandObservation
from src.smoothing import DirectionSmoother, PointSmoother

TrailPoint = Tuple[Point2D, Point2D, float]


class SaberInstance:
    """Manages tracking, velocity, elastic recoil, and disarm states for a single saber."""

    def __init__(
        self,
        handedness: str,
        color: Tuple[int, int, int] = (255, 90, 20),
        alpha_pivot: float = 0.65,
        alpha_direction: float = 0.60,
        grace_period: float = 0.20,
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

        # Velocity tracking
        self.tip_speed: float = 0.0
        self._prev_endpoint: Optional[Point2D] = None
        self._prev_time: float = 0.0

        # Disarm state
        self.is_disarmed: bool = False
        self.disarm_until_time: float = 0.0

    def disarm(self, curr_time: float, duration: float = 3.5) -> None:
        """Knock saber out of hand and initiate re-arm cooldown."""
        self.is_disarmed = True
        self.is_active = False
        self.disarm_until_time = curr_time + duration
        self.trail_history.clear()
        self.emitter_smoother.reset()
        self.hilt_start_smoother.reset()
        self.direction_smoother.reset()

    def apply_recoil(self, recoil_angle: float, blade_length: float) -> None:
        """Apply elastic angular recoil deflection to the blade direction and endpoint."""
        if not self.is_active or abs(recoil_angle) < 1e-4:
            return
        if self.current_direction is None or self.current_emitter is None:
            return

        cos_a, sin_a = math.cos(recoil_angle), math.sin(recoil_angle)
        dx, dy = self.current_direction

        # Rotated unit direction vector
        rot_dx = cos_a * dx - sin_a * dy
        rot_dy = sin_a * dx + cos_a * dy
        self.current_direction = (rot_dx, rot_dy)

        # Updated endpoint
        self.current_endpoint = (
            self.current_emitter[0] + rot_dx * blade_length,
            self.current_emitter[1] + rot_dy * blade_length,
        )

    def update(
        self,
        observation: Optional[HandObservation],
        curr_time: float,
        blade_length: float = 550.0,
        trail_duration: float = 0.22,
    ) -> None:
        """Update saber state from hand observation, tracking velocity and disarm status."""
        # Check disarm cooldown recovery
        if self.is_disarmed:
            if curr_time >= self.disarm_until_time:
                self.is_disarmed = False
            else:
                self.is_active = False
                return

        if observation is not None:
            # Calculate arm-extended geometry
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

            # Compute blade tip speed (pixels/second)
            if self._prev_endpoint is not None and curr_time > self._prev_time:
                dt = max(1e-4, curr_time - self._prev_time)
                dist = math.hypot(
                    smooth_endpoint[0] - self._prev_endpoint[0],
                    smooth_endpoint[1] - self._prev_endpoint[1],
                )
                self.tip_speed = 0.7 * self.tip_speed + 0.3 * (dist / dt)
            self._prev_endpoint = smooth_endpoint
            self._prev_time = curr_time

            self.current_emitter = smooth_emitter
            self.current_endpoint = smooth_endpoint
            self.current_hilt_start = smooth_hstart
            self.current_hilt_end = smooth_emitter
            self.current_direction = smooth_dir
            self.last_seen_time = curr_time
            self.is_active = True

            # Append to trail history
            self.trail_history.append((smooth_emitter, smooth_endpoint, curr_time))
        else:
            # Grace period
            if curr_time - self.last_seen_time > self.grace_period:
                self.is_active = False
                self.tip_speed = 0.0
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
        self.is_disarmed = False
        self.disarm_until_time = 0.0
        self.tip_speed = 0.0
        self.last_seen_time = 0.0
