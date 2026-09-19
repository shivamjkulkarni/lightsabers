"""Unit tests for duel countdown, distance-based blade scaling, two-person tracker, and parry mechanics."""

import unittest
from src.collision import CollisionInfo, evaluate_duel_clash
from src.geometry import calculate_distance_scaled_blade_length
from src.hand_tracker import HandObservation, TwoPersonTracker


def make_mock_observation(
    handedness: str,
    wrist: tuple,
    knuckles: tuple,
    hand_size: float = 60.0,
) -> HandObservation:
    """Helper to construct a mock HandObservation."""
    return HandObservation(
        handedness=handedness,
        landmarks=[wrist] * 21,
        wrist=wrist,
        index_mcp=knuckles,
        index_tip=(knuckles[0] + 10, knuckles[1]),
        middle_mcp=knuckles,
        knuckles_center=knuckles,
        palm_center=((wrist[0] + knuckles[0]) * 0.5, (wrist[1] + knuckles[1]) * 0.5),
        hand_size=hand_size,
    )


class TestDuelMechanics(unittest.TestCase):
    """Test suite for duel countdown, scaling, tracking, and combat evaluation."""

    def test_distance_scaled_blade_length(self) -> None:
        """Verify blade scales proportionally with hand distance and respects clamps."""
        # Reference desk distance (hand_size = 95px -> 550.0px)
        length_normal = calculate_distance_scaled_blade_length(95.0)
        self.assertAlmostEqual(length_normal, 550.0, delta=1.0)

        # Mid distance (hand_size = 65px -> 376.3px)
        length_mid = calculate_distance_scaled_blade_length(65.0)
        self.assertAlmostEqual(length_mid, 376.3, delta=1.0)

        # Far distance (hand_size = 15px -> 550 * (15/95) = 86.8px, clamped to min 120px)
        length_far = calculate_distance_scaled_blade_length(15.0, min_length=120.0)
        self.assertEqual(length_far, 120.0)

        # Close-up distance (hand_size = 130px -> clamped to max 580px)
        length_close = calculate_distance_scaled_blade_length(130.0, max_length=580.0)
        self.assertEqual(length_close, 580.0)

    def test_two_person_tracker_single_person_filter(self) -> None:
        """Verify a single person raising 2 hands is filtered to strictly 1 dominant hand."""
        tracker = TwoPersonTracker(same_person_max_distance_ratio=5.5)
        h1 = make_mock_observation("Right", (400.0, 300.0), (400.0, 240.0), hand_size=60.0)
        h2 = make_mock_observation("Left", (550.0, 350.0), (550.0, 290.0), hand_size=60.0)

        filtered = tracker.update([h1, h2], curr_time=1.0)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].handedness, "Right")

    def test_two_person_tracker_distinct_people(self) -> None:
        """Verify two people standing side-by-side are recognized as 2 distinct combatants."""
        tracker = TwoPersonTracker(same_person_max_distance_ratio=5.5, confirmation_distance_ratio=4.0)
        # Person 1 on left, Person 2 on right (distance = 550px > 4.0 * 60 = 240px)
        p1 = make_mock_observation("Right", (250.0, 350.0), (250.0, 290.0), hand_size=60.0)
        p2 = make_mock_observation("Right", (800.0, 350.0), (800.0, 290.0), hand_size=60.0)

        filtered = tracker.update([p1, p2], curr_time=1.0)
        self.assertEqual(len(filtered), 2)
        self.assertTrue(tracker.two_people_confirmed)

    def test_two_person_tracker_clash_proximity_hysteresis(self) -> None:
        """Verify two distinct people do NOT merge when crossing hands in close proximity during a clash."""
        tracker = TwoPersonTracker(same_person_max_distance_ratio=5.5, confirmation_distance_ratio=4.0)

        # Step 1: Establish 2 distinct people standing apart
        p1_initial = make_mock_observation("Right", (300.0, 400.0), (300.0, 340.0), hand_size=60.0)
        p2_initial = make_mock_observation("Right", (850.0, 400.0), (850.0, 340.0), hand_size=60.0)
        filtered1 = tracker.update([p1_initial, p2_initial], curr_time=1.0)
        self.assertEqual(len(filtered1), 2)
        self.assertTrue(tracker.two_people_confirmed)

        # Step 2: Combatants clash! Wrists move close to screen center (150px apart)
        p1_clash = make_mock_observation("Right", (550.0, 420.0), (550.0, 360.0), hand_size=60.0)
        p2_clash = make_mock_observation("Right", (700.0, 420.0), (700.0, 360.0), hand_size=60.0)
        filtered2 = tracker.update([p1_clash, p2_clash], curr_time=1.05)

        # Should still maintain 2 distinct players despite close proximity!
        self.assertEqual(len(filtered2), 2)
        self.assertTrue(tracker.two_people_confirmed)

    def test_parry_evaluation_countdown_suppression(self) -> None:
        """Verify disarms are suppressed when duel countdown is active."""
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.5, ratio2=0.85)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),
            s1_speed=450.0,
            s2_speed=50.0,
            is_new_clash=True,
            is_duel_active=False,  # Countdown active
        )
        self.assertIsNone(res.disarmed_slot)
        self.assertEqual(res.reason, "countdown_active")

    def test_parry_evaluation_mutual_clash(self) -> None:
        """Verify mutual high-velocity attacks result in mutual clash with no disarm."""
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.5, ratio2=0.5)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(0.7, 0.7),
            s2_dir=(-0.7, 0.7),
            s1_speed=420.0,
            s2_speed=390.0,
            is_new_clash=True,
            is_duel_active=True,
        )
        self.assertTrue(res.is_mutual)
        self.assertIsNone(res.disarmed_slot)
        self.assertEqual(res.banner_text, "MUTUAL CLASH!")

    def test_parry_evaluation_forte_success(self) -> None:
        """Verify a defender blocking with the forte (lower blade, ratio <= 0.72) executes a successful parry."""
        # Person 1 attacks fast (440 px/s), Person 2 blocks across at 90 deg with forte (ratio2 = 0.45)
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.45)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),  # Perpendicular cross (90 deg)
            s1_speed=440.0,
            s2_speed=80.0,
            is_new_clash=True,
            is_duel_active=True,
        )
        self.assertTrue(res.is_parry)
        self.assertIsNone(res.disarmed_slot)
        self.assertIn("PARRIED", res.banner_text or "")

    def test_parry_evaluation_foible_miss(self) -> None:
        """Verify a strike hitting the defender's weak tip (foible, ratio > 0.72) results in missed parry disarm."""
        # Person 1 attacks fast (450 px/s), Person 2 struck at weak tip (ratio2 = 0.88)
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.88)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),
            s1_speed=450.0,
            s2_speed=80.0,
            is_new_clash=True,
            is_duel_active=True,
        )
        self.assertFalse(res.is_parry)
        self.assertEqual(res.disarmed_slot, "Person2")
        self.assertIn("WEAK TIP", res.banner_text or "")

    def test_parry_evaluation_parallel_slip_miss(self) -> None:
        """Verify a defender with poor crossing angle (< 30 deg) causes missed parry disarm."""
        # Blades nearly parallel (angle < 20 deg)
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.5, ratio2=0.45)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(0.98, 0.20),
            s2_dir=(0.95, 0.31),  # Angle approx 7 degrees
            s1_speed=440.0,
            s2_speed=70.0,
            is_new_clash=True,
            is_duel_active=True,
        )
        self.assertFalse(res.is_parry)
        self.assertEqual(res.disarmed_slot, "Person2")
        self.assertIn("SLIPPED", res.banner_text or "")


    def test_dual_ignition_chambers_boundary_isolation(self) -> None:
        """Verify left and right ignition chambers are spatially isolated."""
        from src.gestures import is_hand_in_box
        from src.config import DualIgnitionBoxConfig

        box_cfg = DualIgnitionBoxConfig()
        frame_w, frame_h = 1000, 1000
        left_box = (
            int(box_cfg.left_x_min * frame_w),
            int(box_cfg.left_y_min * frame_h),
            int(box_cfg.left_x_max * frame_w),
            int(box_cfg.left_y_max * frame_h),
        )
        right_box = (
            int(box_cfg.right_x_min * frame_w),
            int(box_cfg.right_y_min * frame_h),
            int(box_cfg.right_x_max * frame_w),
            int(box_cfg.right_y_max * frame_h),
        )

        # Hand on the left side (x = 200, y = 450)
        left_hand_wrist = (200.0, 450.0)
        left_hand_knuckles = (200.0, 400.0)
        self.assertTrue(is_hand_in_box(left_hand_wrist, left_hand_knuckles, left_box))
        self.assertFalse(is_hand_in_box(left_hand_wrist, left_hand_knuckles, right_box))

        # Hand on the right side (x = 800, y = 450)
        right_hand_wrist = (800.0, 450.0)
        right_hand_knuckles = (800.0, 400.0)
        self.assertFalse(is_hand_in_box(right_hand_wrist, right_hand_knuckles, left_box))
        self.assertTrue(is_hand_in_box(right_hand_wrist, right_hand_knuckles, right_box))

        # Hand in the center neutral zone (x = 500, y = 450)
        center_wrist = (500.0, 450.0)
        center_knuckles = (500.0, 400.0)
        self.assertFalse(is_hand_in_box(center_wrist, center_knuckles, left_box))
        self.assertFalse(is_hand_in_box(center_wrist, center_knuckles, right_box))

    def test_saber_reset_between_rounds(self) -> None:
        """Verify saber reset extinguishes blades and clears ignition state for next round."""
        from src.saber import SaberInstance
        saber = SaberInstance(handedness="Right", color=(255, 140, 0))
        saber.ignite()
        saber.assigned_wrist_pos = (200.0, 400.0)
        self.assertTrue(saber.is_ignited)
        self.assertIsNotNone(saber.assigned_wrist_pos)

        # Extinguish for next round
        saber.reset()
        self.assertFalse(saber.is_ignited)
        self.assertIsNone(saber.assigned_wrist_pos)
        self.assertFalse(saber.is_active)
        self.assertFalse(saber.is_disarmed)

    def test_match_winner_screen_rendering(self) -> None:
        """Smoke test verifying render_match_winner_screen renders onto frame and light canvas."""
        import numpy as np
        from src.renderer import render_match_winner_screen

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        light_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        render_match_winner_screen(
            frame=frame,
            light_canvas=light_canvas,
            winner_name="Jedi Blue",
            winner_color=(255, 140, 0),
            score_blue=2,
            score_red=1,
            curr_time=1.0,
        )

        # Verify pixels were drawn
        self.assertGreater(int(frame.sum()), 0)
        self.assertGreater(int(light_canvas.sum()), 0)

    def test_guard_poise_reduction_on_weak_block(self) -> None:
        """Verify that when defender has >1 poise, a weak tip hit shakes guard without disarming."""
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.88)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),
            s1_speed=450.0,
            s2_speed=80.0,
            is_new_clash=True,
            is_duel_active=True,
            s1_poise=2,
            s2_poise=2,
        )
        self.assertFalse(res.is_parry)
        self.assertIsNone(res.disarmed_slot)  # NOT disarmed!
        self.assertEqual(res.poise_damaged_slot, "Person2")
        self.assertIn("GUARD SHAKEN", res.banner_text or "")

    def test_guard_break_disarm_on_last_poise(self) -> None:
        """Verify that when defender has <= 1 poise, a weak tip hit breaks guard and disarms."""
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.88)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),
            s1_speed=450.0,
            s2_speed=80.0,
            is_new_clash=True,
            is_duel_active=True,
            s1_poise=1,
            s2_poise=1,
        )
        self.assertFalse(res.is_parry)
        self.assertEqual(res.disarmed_slot, "Person2")  # Disarmed!
        self.assertIn("DISARMED", res.banner_text or "")

    def test_perfect_parry_active_deflection(self) -> None:
        """Verify active swing into incoming strike at sharp angle executes a Perfect Parry."""
        # Person 1 attacks fast (440 px/s), Person 2 actively deflects with forte at 90 deg (speed = 130 px/s, ratio2 = 0.40)
        col = CollisionInfo(clash_point=(600.0, 350.0), normal=(0.0, 1.0), ratio1=0.7, ratio2=0.40)
        res = evaluate_duel_clash(
            collision=col,
            s1_dir=(1.0, 0.0),
            s2_dir=(0.0, 1.0),
            s1_speed=440.0,
            s2_speed=130.0,  # Actively swung blade into strike!
            is_new_clash=True,
            is_duel_active=True,
            s1_poise=2,
            s2_poise=2,
        )
        self.assertTrue(res.is_parry)
        self.assertTrue(res.is_perfect_parry)
        self.assertEqual(res.shockwave_type, "perfect")
        self.assertIn("PERFECT PARRY", res.banner_text or "")

    def test_shockwave_particle_lifecycle(self) -> None:
        """Verify Shockwave particles expand outward and decay cleanly."""
        import numpy as np
        from src.particles import ParticleSystem

        ps = ParticleSystem()
        ps.spawn_shockwave(500.0, 400.0, max_radius=150.0, speed=400.0, lifetime=0.30)
        self.assertEqual(len(ps.shockwaves), 1)

        sw = ps.shockwaves[0]
        initial_r = sw.radius

        # Advance 50ms
        ps.update(0.05)
        self.assertGreater(sw.radius, initial_r)

        # Draw onto canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        ps.draw(canvas)
        self.assertGreater(int(canvas.sum()), 0)

        # Advance beyond lifetime -> should expire cleanly
        for _ in range(10):
            ps.update(0.04)
        self.assertEqual(len(ps.shockwaves), 0)

    def test_spatial_hand_reacquisition_logic(self) -> None:
        """Verify spatial assignment correctly routes off-screen hands back to their respective sabers."""
        frame_width = 1280
        # Two hands: one on left (x=320), one on right (x=960)
        h_left = make_mock_observation("Right", (320.0, 400.0), (320.0, 340.0))
        h_right = make_mock_observation("Right", (960.0, 400.0), (960.0, 340.0))
        obs_list = [h_right, h_left]  # Order reversed

        sorted_obs = sorted(obs_list, key=lambda o: o.wrist[0])
        self.assertEqual(sorted_obs[0].wrist[0], 320.0)  # Left hand to Blue
        self.assertEqual(sorted_obs[-1].wrist[0], 960.0)  # Right hand to Red

    def test_render_tutorial_cards_all_slides(self) -> None:
        """Verify render_tutorial_card renders cleanly without exceptions across all slides."""
        import numpy as np
        from src.renderer import render_tutorial_card

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        light_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        for slide_idx in (1, 2, 3):
            frame.fill(20)
            light_canvas.fill(0)
            render_tutorial_card(frame, light_canvas, slide_index=slide_idx, curr_time=1.0)
            self.assertGreater(int(frame.sum()), 0)
            self.assertGreater(int(light_canvas.sum()), 0)


if __name__ == "__main__":
    unittest.main()




