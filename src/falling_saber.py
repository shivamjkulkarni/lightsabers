"""Physics simulation and rendering for disarmed lightsabers falling and bouncing on the floor."""

import math
from typing import Tuple
import cv2
import numpy as np

from src.particles import ParticleSystem
from src.renderer import draw_blade_on_light_canvas, render_hilt

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]


class FallingSaber:
    """Simulates a lightsaber knocked out of hand tumbling through air and bouncing on floor."""

    def __init__(
        self,
        hilt_pos: Point2D,
        initial_velocity: Vector2D,
        initial_angle: float,
        angular_velocity: float,
        color: Tuple[int, int, int],
        blade_length: float = 550.0,
        floor_y: float = 690.0,
        floor_width: float = 1280.0,
    ) -> None:
        self.x, self.y = hilt_pos
        self.vx, self.vy = initial_velocity
        self.angle = initial_angle
        self.angular_vel = angular_velocity
        self.color = color
        self.blade_length = blade_length
        self.initial_blade_length = blade_length
        self.floor_y = floor_y
        self.floor_width = floor_width

        self.gravity = 950.0
        self.bounce_count = 0
        self.is_retracting = False
        self.is_dead = False
        self.time_on_ground = 0.0

    def update(self, dt: float, particles: ParticleSystem) -> None:
        """Advance physical tumble, gravity, floor bounces, and blade retraction."""
        if self.is_dead or dt <= 0.0:
            return

        dt = min(dt, 0.05)

        # 1. Gravity acceleration
        self.vy += self.gravity * dt

        # 2. Position and tumbling integration
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_vel * dt

        # Current blade direction vector
        dir_x = math.cos(self.angle)
        dir_y = math.sin(self.angle)

        tip_x = self.x + dir_x * self.blade_length
        tip_y = self.y + dir_y * self.blade_length

        # 3. Floor collisions (hilt or blade tip hitting the ground)
        hit_floor = False

        if self.y >= self.floor_y:
            self.y = self.floor_y
            if abs(self.vy) > 60.0:
                self.vy = -self.vy * 0.42  # Bouncy restitution
                particles.spawn_ground_sparks(self.x, self.floor_y, count=12)
            else:
                self.vy = 0.0
            self.vx *= 0.78  # Floor friction
            self.angular_vel *= 0.65
            self.bounce_count += 1
            hit_floor = True

        if tip_y >= self.floor_y and self.blade_length > 10.0:
            tip_clamped_x = min(max(tip_x, 0.0), max(0.0, self.floor_width - 1.0))
            if abs(self.vy) > 40.0:
                particles.spawn_ground_sparks(tip_clamped_x, self.floor_y, count=8)
                self.angular_vel = -self.angular_vel * 0.50
                self.vy = -abs(self.vy) * 0.35
            self.vx *= 0.85
            hit_floor = True

        # 4. Blade retraction sequence after floor impact
        if self.bounce_count >= 2 or hit_floor:
            self.is_retracting = True

        if self.is_retracting:
            self.time_on_ground += dt
            # Retract blade inward towards hilt
            self.blade_length = max(0.0, self.blade_length - 850.0 * dt)
            if self.time_on_ground > 4.5:
                self.is_dead = True

    def draw(self, frame: np.ndarray, light_canvas: np.ndarray, curr_time: float) -> None:
        """Render tumbling physical hilt and glowing/retracting blade."""
        if self.is_dead:
            return

        dir_x = math.cos(self.angle)
        dir_y = math.sin(self.angle)
        direction: Vector2D = (dir_x, dir_y)

        # Hilt points
        hilt_len = 50.0
        h_end: Point2D = (self.x, self.y)
        h_start: Point2D = (self.x - dir_x * hilt_len, self.y - dir_y * hilt_len)

        # Render physical hilt
        render_hilt(frame, h_start, h_end, direction)

        # Render glowing blade if still extended
        if self.blade_length > 5.0:
            emitter: Point2D = h_end
            tip: Point2D = (self.x + dir_x * self.blade_length, self.y + dir_y * self.blade_length)
            draw_blade_on_light_canvas(light_canvas, emitter, tip, self.color, curr_time=curr_time)
