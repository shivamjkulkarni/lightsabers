"""Collision detection, elastic blade recoil physics, and disarm trigger mechanics."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]


@dataclass
class CollisionInfo:
    """Detailed information regarding a blade-on-blade intersection."""
    clash_point: Point2D
    normal: Vector2D
    ratio1: float  # Distance along blade 1 (0.0 at emitter, 1.0 at tip)
    ratio2: float  # Distance along blade 2 (0.0 at emitter, 1.0 at tip)


def check_blade_collision(
    s1_emitter: Point2D,
    s1_tip: Point2D,
    s2_emitter: Point2D,
    s2_tip: Point2D,
) -> Optional[CollisionInfo]:
    """
    Test 2D line segment intersection between two lightsaber blades.
    Returns CollisionInfo if segments intersect, otherwise None.
    """
    p1x, p1y = s1_emitter
    e1x, e1y = s1_tip
    p2x, p2y = s2_emitter
    e2x, e2y = s2_tip

    v1x, v1y = e1x - p1x, e1y - p1y
    v2x, v2y = e2x - p2x, e2y - p2y

    denom = v1x * v2y - v1y * v2x
    if abs(denom) < 1e-6:
        # Parallel or collinear
        return None

    dx = p2x - p1x
    dy = p2y - p1y

    t = (dx * v2y - dy * v2x) / denom
    u = (dx * v1y - dy * v1x) / denom

    # Check if intersection lies within both line segments
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        clash_x = p1x + t * v1x
        clash_y = p1y + t * v1y

        # Compute collision normal (perpendicular to blade 1)
        len_v1 = math.hypot(v1x, v1y)
        if len_v1 > 1e-5:
            nx = -v1y / len_v1
            ny = v1x / len_v1
        else:
            nx, ny = 0.0, -1.0

        return CollisionInfo(
            clash_point=(clash_x, clash_y),
            normal=(nx, ny),
            ratio1=t,
            ratio2=u,
        )

    return None


class BladeRecoilController:
    """
    Simulates elastic blade recoil using a damped harmonic oscillator.
    
    When two blades clash, an angular impulse pushes each blade away from
    the collision boundary. The spring-damper returns the blade smoothly
    to the hand's natural orientation with a subtle tactile recoil bounce.
    """

    def __init__(self, omega: float = 38.0, zeta: float = 0.75) -> None:
        self.omega = omega  # Natural frequency of blade rigidity
        self.zeta = zeta    # Damping ratio (under-damped for responsive snap)
        self.angles: Dict[str, float] = {}
        self.velocities: Dict[str, float] = {}

    def apply_clash_impulse(
        self,
        slot1: str,
        slot2: str,
        normal: Vector2D,
        s1_dir: Vector2D,
        s2_dir: Vector2D,
        impulse_magnitude: float = 0.45,
    ) -> None:
        """Apply repulsive angular recoil impulses to both blades."""
        self.angles[slot1] = self.angles.get(slot1, 0.0)
        self.angles[slot2] = self.angles.get(slot2, 0.0)
        # Deflect blade 1 away from normal
        self.velocities[slot1] = self.velocities.get(slot1, 0.0) + impulse_magnitude
        # Deflect blade 2 in opposite direction
        self.velocities[slot2] = self.velocities.get(slot2, 0.0) - impulse_magnitude

    def update(self, dt: float) -> None:
        """Advance spring-damper physics for all recoiling blades."""
        if dt <= 0.0:
            return

        dt = min(dt, 0.05)
        for slot in list(self.angles.keys()):
            theta = self.angles[slot]
            v = self.velocities.get(slot, 0.0)

            # d^2(theta)/dt^2 = -omega^2 * theta - 2 * zeta * omega * v
            accel = - (self.omega ** 2) * theta - 2.0 * self.zeta * self.omega * v
            v += accel * dt
            theta += v * dt

            # Damping cutoff
            if abs(theta) < 1e-4 and abs(v) < 1e-3:
                theta = 0.0
                v = 0.0

            self.angles[slot] = theta
            self.velocities[slot] = v

    def get_recoil_angle(self, slot: str) -> float:
        """Get current angular deflection in radians."""
        return self.angles.get(slot, 0.0)

    def reset(self) -> None:
        """Reset all recoil state."""
        self.angles.clear()
        self.velocities.clear()


def check_disarm_trigger(
    collision: CollisionInfo,
    v_rel: float,
    s1_speed: float,
    s2_speed: float,
    disarm_threshold: float = 520.0,
) -> Optional[str]:
    """
    Determine whether the clash was forceful enough to disarm one of the blades.
    
    If relative strike speed is high and one combatant struck with significantly
    greater speed, the defending/slower blade gets disarmed and knocked away!
    Returns slot name of the disarmed blade ('Person1' or 'Person2') or None.
    """
    if v_rel < disarm_threshold:
        return None

    # Determine who struck and who was struck
    speed_diff = abs(s1_speed - s2_speed)
    if speed_diff > 160.0:
        if s1_speed > s2_speed:
            # Person 1 struck Person 2 -> Person 2 is disarmed
            return "Person2"
        else:
            # Person 2 struck Person 1 -> Person 1 is disarmed
            return "Person1"

    return None
