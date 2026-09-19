"""Unit tests for geometry functions."""

import math
import unittest

from src.geometry import (
    angle,
    calculate_arm_extended_geometry,
    calculate_saber_direction,
    calculate_saber_endpoint,
    distance,
    is_saber_active,
    lerp,
    normalize,
    scale_length_for_resolution,
)


class TestGeometry(unittest.TestCase):

    def test_normalize_standard_vector(self):
        v = (3.0, 4.0)
        norm = normalize(v)
        self.assertAlmostEqual(norm[0], 0.6)
        self.assertAlmostEqual(norm[1], 0.8)
        self.assertAlmostEqual(math.hypot(norm[0], norm[1]), 1.0)

    def test_normalize_zero_vector(self):
        # Should gracefully return upward unit vector without ZeroDivisionError
        v = (0.0, 0.0)
        norm = normalize(v)
        self.assertEqual(norm, (0.0, -1.0))

    def test_distance(self):
        p1 = (1.0, 2.0)
        p2 = (4.0, 6.0)
        self.assertAlmostEqual(distance(p1, p2), 5.0)

    def test_lerp(self):
        p1 = (0.0, 10.0)
        p2 = (10.0, 20.0)
        self.assertEqual(lerp(p1, p2, 0.0), (0.0, 10.0))
        self.assertEqual(lerp(p1, p2, 1.0), (10.0, 20.0))
        self.assertEqual(lerp(p1, p2, 0.5), (5.0, 15.0))

    def test_angle(self):
        self.assertAlmostEqual(angle((1.0, 0.0)), 0.0)
        self.assertAlmostEqual(angle((0.0, 1.0)), math.pi / 2)
        self.assertAlmostEqual(angle((-1.0, 0.0)), math.pi)

    def test_calculate_saber_direction(self):
        wrist = (500.0, 500.0)
        mcp = (500.0, 450.0)
        tip = (500.0, 400.0)
        dir_vec = calculate_saber_direction(wrist, mcp, tip)
        self.assertAlmostEqual(dir_vec[0], 0.0)
        self.assertAlmostEqual(dir_vec[1], -1.0)

    def test_calculate_saber_endpoint(self):
        pivot = (200.0, 300.0)
        direction = (1.0, 0.0)
        blade_length = 250.0
        endpoint = calculate_saber_endpoint(pivot, direction, blade_length)
        self.assertAlmostEqual(endpoint[0], 450.0)
        self.assertAlmostEqual(endpoint[1], 300.0)

    def test_calculate_arm_extended_geometry(self):
        wrist = (500.0, 500.0)
        knuckles = (500.0, 400.0)  # Arm pointing up (-y)
        tip = (500.0, 350.0)
        emitter, endpoint, hstart, hend, direction = calculate_arm_extended_geometry(
            wrist, knuckles, tip, blade_length=600.0
        )
        # Direction should be pointing straight up
        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], -1.0)
        # Emitter should be beyond knuckles in direction of arm (y < 400.0)
        self.assertLess(emitter[1], knuckles[1])
        # Endpoint should be 600px beyond emitter
        self.assertAlmostEqual(emitter[1] - endpoint[1], 600.0)
        # Hilt starts at wrist
        self.assertEqual(hstart, wrist)

    def test_scale_length_for_resolution(self):
        # Default 720p reference
        self.assertAlmostEqual(scale_length_for_resolution(320.0, 720.0, 720.0), 320.0)
        # 1080p scaled
        self.assertAlmostEqual(scale_length_for_resolution(320.0, 1080.0, 720.0), 480.0)
        # 360p scaled
        self.assertAlmostEqual(scale_length_for_resolution(320.0, 360.0, 720.0), 160.0)

    def test_is_saber_active(self):
        self.assertTrue(is_saber_active("mock_observation"))
        self.assertFalse(is_saber_active(None))


if __name__ == "__main__":
    unittest.main()
