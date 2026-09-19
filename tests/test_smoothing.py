"""Unit tests for temporal smoothing and saber state management."""

import math
import unittest

from src.hand_tracker import HandObservation
from src.saber import SaberInstance
from src.smoothing import DirectionSmoother, PointSmoother


def make_test_observation(handedness: str, wrist=(500.0, 500.0)) -> HandObservation:
    knuckles = (wrist[0], wrist[1] - 80.0)
    return HandObservation(
        handedness=handedness,
        landmarks=[(0.0, 0.0)] * 21,
        wrist=wrist,
        index_mcp=(wrist[0] - 20.0, wrist[1] - 80.0),
        index_tip=(wrist[0] - 20.0, wrist[1] - 140.0),
        middle_mcp=(wrist[0], wrist[1] - 85.0),
        knuckles_center=knuckles,
        palm_center=(wrist[0], wrist[1] - 40.0),
    )


class TestSmoothing(unittest.TestCase):

    def test_point_smoother(self):
        smoother = PointSmoother(alpha=0.5)

        # First point sets initial value
        p1 = smoother.update((100.0, 100.0))
        self.assertEqual(p1, (100.0, 100.0))

        # Second point smoothed with 0.5 weight
        p2 = smoother.update((200.0, 200.0))
        self.assertEqual(p2, (150.0, 150.0))

        # Reset clears state
        smoother.reset()
        p3 = smoother.update((50.0, 50.0))
        self.assertEqual(p3, (50.0, 50.0))

    def test_direction_smoother_preserves_unit_length(self):
        smoother = DirectionSmoother(alpha=0.5)

        d1 = smoother.update((1.0, 0.0))
        self.assertAlmostEqual(math.hypot(d1[0], d1[1]), 1.0)

        d2 = smoother.update((0.0, 1.0))
        self.assertAlmostEqual(math.hypot(d2[0], d2[1]), 1.0)
        self.assertAlmostEqual(d2[0], math.sqrt(0.5))
        self.assertAlmostEqual(d2[1], math.sqrt(0.5))

    def test_saber_instance_trail_and_expiration(self):
        saber = SaberInstance(handedness="Right", grace_period=0.20)
        obs = make_test_observation("Right", wrist=(500.0, 500.0))

        # Update at t = 0.0
        saber.update(obs, curr_time=0.0, blade_length=600.0, trail_duration=0.30)
        self.assertTrue(saber.is_active)
        self.assertEqual(len(saber.trail_history), 1)

        # Update at t = 0.1
        saber.update(obs, curr_time=0.1, blade_length=600.0, trail_duration=0.30)
        self.assertEqual(len(saber.trail_history), 2)

        # Update at t = 0.45 (point at t=0.0 and t=0.1 should expire if trail_duration is 0.30)
        saber.update(obs, curr_time=0.45, blade_length=600.0, trail_duration=0.30)
        # Only the point from t = 0.45 should remain
        self.assertEqual(len(saber.trail_history), 1)

    def test_saber_instance_grace_period(self):
        saber = SaberInstance(handedness="Left", grace_period=0.20)
        obs = make_test_observation("Left", wrist=(200.0, 300.0))

        saber.update(obs, curr_time=0.0)
        self.assertTrue(saber.is_active)

        # Frame dropped at t = 0.1 (within grace period of 0.2s)
        saber.update(None, curr_time=0.1)
        self.assertTrue(saber.is_active)

        # Frame dropped at t = 0.3 (exceeds grace period)
        saber.update(None, curr_time=0.3)
        self.assertFalse(saber.is_active)


if __name__ == "__main__":
    unittest.main()
