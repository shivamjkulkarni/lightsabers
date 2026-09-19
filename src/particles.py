"""Newtonian spark particle physics system for lightsaber clashes and floor impacts."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class Spark:
    """A single Newtonian spark particle."""
    x: float
    y: float
    vx: float
    vy: float
    lifetime: float
    max_lifetime: float
    size: float = 2.0


class ParticleSystem:
    """Simulates and renders high-velocity sparks with gravity, drag, and thermal color decay."""

    def __init__(self, gravity: float = 850.0, drag: float = 0.93) -> None:
        self.gravity = gravity
        self.drag = drag
        self.sparks: List[Spark] = []

    def spawn_clash_sparks(
        self,
        x: float,
        y: float,
        normal_x: float = 0.0,
        normal_y: float = -1.0,
        count: int = 30,
        speed_min: float = 180.0,
        speed_max: float = 450.0,
    ) -> None:
        """Spawn radiant sparks spraying outward along the clash plane."""
        # Tangent vector to collision normal
        tan_x, tan_y = -normal_y, normal_x

        for _ in range(count):
            # Spray primarily along tangent with random radial spread
            dir_sign = 1.0 if random.random() > 0.5 else -1.0
            spread_angle = random.uniform(-0.8, 0.8)
            cos_a, sin_a = math.cos(spread_angle), math.sin(spread_angle)

            # Rotate tangent by spread angle
            bx = tan_x * dir_sign
            by = tan_y * dir_sign
            rx = bx * cos_a - by * sin_a
            ry = bx * sin_a + by * cos_a

            speed = random.uniform(speed_min, speed_max)
            lifetime = random.uniform(0.18, 0.36)

            self.sparks.append(
                Spark(
                    x=x,
                    y=y,
                    vx=rx * speed,
                    vy=ry * speed,
                    lifetime=lifetime,
                    max_lifetime=lifetime,
                    size=random.uniform(1.5, 3.0),
                )
            )

    def spawn_ground_sparks(
        self,
        x: float,
        y: float,
        count: int = 15,
        speed_min: float = 120.0,
        speed_max: float = 320.0,
    ) -> None:
        """Spawn upward bouncing sparks when a falling lightsaber strikes the floor."""
        for _ in range(count):
            angle = random.uniform(-math.pi * 0.85, -math.pi * 0.15)  # Upward arc
            speed = random.uniform(speed_min, speed_max)
            lifetime = random.uniform(0.15, 0.30)

            self.sparks.append(
                Spark(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    lifetime=lifetime,
                    max_lifetime=lifetime,
                    size=random.uniform(1.5, 2.5),
                )
            )

    def update(self, dt: float) -> None:
        """Advance physical simulation of all sparks."""
        if dt <= 0.0 or not self.sparks:
            return

        dt = min(dt, 0.05)  # Safeguard against huge delta steps
        alive_sparks: List[Spark] = []

        # Exponential drag scaled by dt
        frame_drag = self.drag ** (dt / 0.016)

        for spark in self.sparks:
            spark.lifetime -= dt
            if spark.lifetime <= 0.0:
                continue

            # Gravity downward
            spark.vy += self.gravity * dt

            # Drag deceleration
            spark.vx *= frame_drag
            spark.vy *= frame_drag

            # Position integration
            spark.x += spark.vx * dt
            spark.y += spark.vy * dt

            alive_sparks.append(spark)

        self.sparks = alive_sparks

    def draw(self, light_canvas: np.ndarray) -> None:
        """Render sparks as motion-blurred streaks onto the light canvas."""
        h, w = light_canvas.shape[:2]

        for spark in self.sparks:
            px, py = spark.x, spark.y
            if not (0 <= px < w and 0 <= py < h):
                continue

            # Motion streak tail (previous position over ~15ms)
            tail_x = int(px - spark.vx * 0.015)
            tail_y = int(py - spark.vy * 0.015)
            head_x = int(px)
            head_y = int(py)

            # Thermal decay: white-hot -> yellow -> amber -> red ember
            progress = max(0.0, min(1.0, spark.lifetime / spark.max_lifetime))
            if progress > 0.65:
                # White-hot core with slight cyan/yellow fringe
                color = (255, 255, 255)
            elif progress > 0.35:
                # Radiant electric yellow
                color = (0, int(210 * (progress / 0.65)), 255)
            else:
                # Cooling orange/red ember
                color = (0, int(90 * (progress / 0.35)), int(220 * (progress / 0.35)))

            cv2.line(light_canvas, (tail_x, tail_y), (head_x, head_y), color, int(spark.size), cv2.LINE_AA)
            cv2.circle(light_canvas, (head_x, head_y), 1, (255, 255, 255), -1, cv2.LINE_AA)

    def clear(self) -> None:
        """Clear all active sparks."""
        self.sparks.clear()
