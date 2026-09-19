"""Unit tests for temporal smoothing, saber state management, and hand filtering."""

import math
import unittest

import numpy as np

from src.config import AppConfig
from src.hand_tracker import HandObservation, filter_one_hand_per_person
from src.renderer import CanvasBuffer
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
        saber.update(obs, curr_time=0.0, blade_length=550.0, trail_duration=0.22)
        self.assertTrue(saber.is_active)
        self.assertEqual(len(saber.trail_history), 1)

        # Update at t = 0.1
        saber.update(obs, curr_time=0.1, blade_length=550.0, trail_duration=0.22)
        self.assertEqual(len(saber.trail_history), 2)

        # Update at t = 0.35 (points at t=0.0 and t=0.1 should expire if trail_duration is 0.22)
        saber.update(obs, curr_time=0.35, blade_length=550.0, trail_duration=0.22)
        # Only the point from t = 0.35 should remain
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

    def test_filter_one_hand_per_person_same_person(self):
        # Two hands of the same person: wrists 150px apart, hand size ~80px
        # threshold is 3.5 * 80 = 280px > 150px -> same person
        h_left = make_test_observation("Left", wrist=(450.0, 420.0))
        h_right = make_test_observation("Right", wrist=(550.0, 380.0))  # higher up (y=380 vs 420)

        filtered = filter_one_hand_per_person([h_left, h_right], same_person_max_distance_ratio=3.5)
        # Should collapse to 1 hand, picking the higher one (h_right)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].handedness, "Right")

    def test_filter_one_hand_per_person_two_people(self):
        # Two distinct people: wrists 700px apart across the screen
        # threshold is 3.5 * 80 = 280px < 700px -> two people
        p1 = make_test_observation("Left", wrist=(200.0, 400.0))
        p2 = make_test_observation("Right", wrist=(950.0, 400.0))

        filtered = filter_one_hand_per_person([p1, p2], same_person_max_distance_ratio=3.5)
        # Both hands retained
        self.assertEqual(len(filtered), 2)

    def test_canvas_buffer_zero_allocation(self):
        buf = CanvasBuffer(1280, 720)
        arr1 = buf.reset_and_get((720, 1280, 3))
        self.assertEqual(arr1.shape, (720, 1280, 3))
        # Modify arr1
        arr1[0, 0, 0] = 255
        # Second call returns the same memory buffer cleared
        arr2 = buf.reset_and_get((720, 1280, 3))
        self.assertIs(arr1, arr2)
        self.assertEqual(arr2[0, 0, 0], 0)

    def test_config_values(self):
        cfg = AppConfig()
        # 15% reduction from 650px = 550px
        self.assertEqual(cfg.saber.base_blade_length, 550.0)
        self.assertEqual(cfg.saber.trail_duration, 0.22)
        self.assertEqual(cfg.saber.trail_leading_opacity, 0.75)
        self.assertEqual(cfg.saber.jedi_blue, (255, 90, 20))
        self.assertEqual(cfg.saber.sith_red, (30, 30, 255))


if __name__ == "__main__":
    unittest.main()
