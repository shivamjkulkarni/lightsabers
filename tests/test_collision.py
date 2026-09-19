"""Unit tests for collision detection, recoil physics, spark particles, and falling saber mechanics."""

import math
import unittest

from src.collision import (
    BladeRecoilController,
    check_blade_collision,
    check_disarm_trigger,
    CollisionInfo,
)
from src.falling_saber import FallingSaber
from src.particles import ParticleSystem


class TestCollision(unittest.TestCase):

    def test_blade_intersection_crossing(self):
        # Blade 1: vertical line from (500, 200) to (500, 600)
        # Blade 2: horizontal line from (300, 400) to (700, 400)
        col = check_blade_collision(
            (500.0, 200.0), (500.0, 600.0),
            (300.0, 400.0), (700.0, 400.0)
        )
        self.assertIsNotNone(col)
        self.assertAlmostEqual(col.clash_point[0], 500.0)
        self.assertAlmostEqual(col.clash_point[1], 400.0)
        self.assertAlmostEqual(col.ratio1, 0.5)
        self.assertAlmostEqual(col.ratio2, 0.5)

    def test_blade_intersection_parallel(self):
        # Parallel vertical blades
        col = check_blade_collision(
            (400.0, 200.0), (400.0, 600.0),
            (500.0, 200.0), (500.0, 600.0)
        )
        self.assertIsNone(col)

    def test_blade_intersection_non_overlapping(self):
        # Blades pointing away from each other
        col = check_blade_collision(
            (100.0, 100.0), (200.0, 100.0),
            (300.0, 100.0), (400.0, 100.0)
        )
        self.assertIsNone(col)

    def test_blade_recoil_spring_damper(self):
        recoil = BladeRecoilController(omega=30.0, zeta=0.7)
        # Apply clash impulse
        recoil.apply_clash_impulse(
            "Person1", "Person2", (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), impulse_magnitude=0.5
        )

        # Initially velocity is set
        self.assertAlmostEqual(recoil.velocities["Person1"], 0.5)
        self.assertAlmostEqual(recoil.velocities["Person2"], -0.5)

        # Advance physics
        recoil.update(0.016)
        angle1 = recoil.get_recoil_angle("Person1")
        angle2 = recoil.get_recoil_angle("Person2")

        # Blades deflect in opposite directions
        self.assertGreater(angle1, 0.0)
        self.assertLess(angle2, 0.0)

        # After several seconds of simulation, spring-damper settles to 0.0
        for _ in range(120):
            recoil.update(0.02)
        self.assertAlmostEqual(recoil.get_recoil_angle("Person1"), 0.0, places=3)
        self.assertAlmostEqual(recoil.get_recoil_angle("Person2"), 0.0, places=3)

    def test_disarm_trigger(self):
        col = CollisionInfo(clash_point=(500.0, 400.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.7)

        # Low velocity: no disarm
        self.assertIsNone(check_disarm_trigger(col, v_rel=200.0, s1_speed=150.0, s2_speed=50.0))

        # High velocity where Person 1 strikes fast and Person 2 is slow: Person 2 disarmed
        disarmed = check_disarm_trigger(col, v_rel=700.0, s1_speed=650.0, s2_speed=100.0)
        self.assertEqual(disarmed, "Person2")

        # High velocity where Person 2 strikes fast and Person 1 is slow: Person 1 disarmed
        disarmed = check_disarm_trigger(col, v_rel=700.0, s1_speed=100.0, s2_speed=650.0)
        self.assertEqual(disarmed, "Person1")

    def test_spark_particle_physics(self):
        ps = ParticleSystem(gravity=800.0, drag=0.90)
        ps.spawn_clash_sparks(500.0, 400.0, normal_x=0.0, normal_y=-1.0, count=20)
        self.assertEqual(len(ps.sparks), 20)

        # Update 1 step: position moves, vy increases under gravity
        initial_vy = ps.sparks[0].vy
        ps.update(0.02)
        self.assertGreater(ps.sparks[0].vy, initial_vy * (0.90 ** (0.02 / 0.016)) - 1.0)

        # Run past lifetime: all sparks expire
        for _ in range(50):
            ps.update(0.02)
        self.assertEqual(len(ps.sparks), 0)

    def test_falling_saber_floor_bounce_and_retract(self):
        ps = ParticleSystem()
        # Saber dropping from y=600 with floor at y=690
        falling = FallingSaber(
            hilt_pos=(500.0, 600.0),
            initial_velocity=(50.0, 100.0),
            initial_angle=0.0,
            angular_velocity=5.0,
            color=(255, 90, 20),
            blade_length=550.0,
            floor_y=690.0,
        )

        # Simulate falling to floor
        for _ in range(30):
            falling.update(0.02, ps)

        # Should have hit floor and bounced
        self.assertGreaterEqual(falling.bounce_count, 1)
        self.assertTrue(falling.is_retracting)

        # Simulate retraction on ground
        for _ in range(60):
            falling.update(0.02, ps)
        self.assertLess(falling.blade_length, 550.0)


if __name__ == "__main__":
    unittest.main()

