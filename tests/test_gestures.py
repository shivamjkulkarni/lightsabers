"""Unit tests for hand sign and gesture recognition, 3D camera pointing, and ignition box check."""

import unittest
from src.gestures import (
    GestureIgnitionTracker,
    is_force_push_pose,
    is_hand_in_box,
    is_two_finger_pose,
)


def create_hand_landmarks(finger_states: dict) -> list:
    """
    Construct a synthetic 21-landmark skeleton (2D).
    finger_states: dict with keys 'index', 'middle', 'ring', 'pinky' -> bool (True=extended, False=curled)
    """
    wrist = (200.0, 400.0)
    landmarks = [wrist] * 21

    fingers = {
        "thumb": (1, 2, 3, 4, 0.4),
        "index": (5, 6, 7, 8, -0.2),
        "middle": (9, 10, 11, 12, 0.0),
        "ring": (13, 14, 15, 16, 0.2),
        "pinky": (17, 18, 19, 20, 0.4),
    }

    for name, (mcp, pip, dip, tip, angle) in fingers.items():
        is_ext = finger_states.get(name, True)
        landmarks[mcp] = (wrist[0] + angle * 40, wrist[1] - 50)
        landmarks[pip] = (landmarks[mcp][0] + angle * 20, landmarks[mcp][1] - 40)
        landmarks[dip] = (landmarks[pip][0] + angle * 10, landmarks[pip][1] - 30)

        if is_ext:
            landmarks[tip] = (landmarks[dip][0] + angle * 10, landmarks[dip][1] - 40)
        else:
            landmarks[tip] = (landmarks[mcp][0], landmarks[mcp][1] - 10)

    return landmarks


def create_3d_camera_pointing_landmarks(finger_states: dict) -> list:
    """
    Construct a synthetic 3D skeleton where fingers point towards the camera (-Z direction).
    Tests foreshortening robustness where 2D distances shrink.
    """
    wrist = (300.0, 400.0, 0.0)
    landmarks = [wrist] * 21

    fingers = {
        "thumb": (1, 2, 3, 4, -20.0),
        "index": (5, 6, 7, 8, -30.0),
        "middle": (9, 10, 11, 12, -10.0),
        "ring": (13, 14, 15, 16, 10.0),
        "pinky": (17, 18, 19, 20, 30.0),
    }

    for name, (mcp, pip, dip, tip, x_offset) in fingers.items():
        is_ext = finger_states.get(name, True)
        landmarks[mcp] = (wrist[0] + x_offset, wrist[1] - 40, -15.0)
        landmarks[pip] = (wrist[0] + x_offset, wrist[1] - 50, -45.0)
        landmarks[dip] = (wrist[0] + x_offset, wrist[1] - 55, -75.0)

        if is_ext:
            # Extended forward toward camera in -Z axis
            landmarks[tip] = (wrist[0] + x_offset, wrist[1] - 60, -115.0)
        else:
            # Curled back onto palm
            landmarks[tip] = (wrist[0] + x_offset, wrist[1] - 35, -20.0)

    return landmarks


class TestGestures(unittest.TestCase):
    """Test suite for gesture recognition."""

    def test_two_finger_pose_recognition_2d(self) -> None:
        """Verify Two-Finger pose (Jedi Blue) triggers when Index+Middle extended and Ring+Pinky curled."""
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertTrue(is_two_finger_pose(landmarks))
        self.assertFalse(is_force_push_pose(landmarks))

    def test_two_finger_pose_3d_camera_pointing(self) -> None:
        """Verify Two-Finger pose works when fingers point directly towards the camera in 3D."""
        landmarks_3d = create_3d_camera_pointing_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertTrue(is_two_finger_pose(landmarks_3d))
        self.assertFalse(is_force_push_pose(landmarks_3d))

    def test_force_push_pose_recognition_2d(self) -> None:
        """Verify Force Push pose (Sith Red) triggers when all 4 main fingers are extended."""
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": True,
            "thumb": True,
        })
        self.assertTrue(is_force_push_pose(landmarks))
        self.assertFalse(is_two_finger_pose(landmarks))

    def test_force_push_pose_3d_camera_pointing(self) -> None:
        """Verify Force Push pose triggers when pushing towards the camera in 3D."""
        landmarks_3d = create_3d_camera_pointing_landmarks({
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": True,
            "thumb": True,
        })
        self.assertTrue(is_force_push_pose(landmarks_3d))
        self.assertFalse(is_two_finger_pose(landmarks_3d))

    def test_fist_no_gesture(self) -> None:
        """Verify closed fist does not trigger either ignition gesture in 2D or 3D."""
        landmarks = create_hand_landmarks({
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertFalse(is_two_finger_pose(landmarks))
        self.assertFalse(is_force_push_pose(landmarks))

        landmarks_3d = create_3d_camera_pointing_landmarks({
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertFalse(is_two_finger_pose(landmarks_3d))
        self.assertFalse(is_force_push_pose(landmarks_3d))

    def test_gesture_ignition_tracker_debounce(self) -> None:
        """Verify tracker requires holding pose for required frame count before ignition."""
        tracker = GestureIgnitionTracker(required_frames=2)
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
        })

        # Frame 1: not ready
        self.assertIsNone(tracker.check_ignition(landmarks))
        # Frame 2: confirmed!
        self.assertEqual(tracker.check_ignition(landmarks), "JediBlue")

    def test_is_hand_in_box(self) -> None:
        """Verify spatial containment for the designated Ignition Box."""
        box_rect = (400, 120, 880, 480)

        # Center inside box
        self.assertTrue(is_hand_in_box(wrist=(640, 300), knuckles=(640, 240), box_rect=box_rect))

        # Knuckles inside, wrist outside bottom
        self.assertTrue(is_hand_in_box(wrist=(640, 520), knuckles=(640, 450), box_rect=box_rect))

        # Completely to the left
        self.assertFalse(is_hand_in_box(wrist=(200, 300), knuckles=(200, 240), box_rect=box_rect))

        # Completely to the right
        self.assertFalse(is_hand_in_box(wrist=(1000, 300), knuckles=(1000, 240), box_rect=box_rect))

        # Above the box
        self.assertFalse(is_hand_in_box(wrist=(640, 40), knuckles=(640, 20), box_rect=box_rect))

    def test_is_hand_in_box_with_palm_center(self) -> None:
        """Verify palm_center allows detection when wrist and knuckles are on the boundary."""
        box_rect = (400, 120, 880, 480)
        # Palm center is inside, while knuckles are near top and wrist is near bottom
        self.assertTrue(
            is_hand_in_box(
                wrist=(640, 500),
                knuckles=(640, 100),
                box_rect=box_rect,
                palm_center=(640, 300),
            )
        )

    def test_jedi_two_finger_relaxed_curling(self) -> None:
        """Verify Jedi pose triggers even if ring finger is naturally relaxed rather than clamped."""
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
        })
        # Simulate ring finger slightly relaxed (tip not all the way in MCP)
        landmarks[16] = (landmarks[13][0], landmarks[13][1] - 25)
        self.assertTrue(is_two_finger_pose(landmarks))
        self.assertFalse(is_force_push_pose(landmarks))

    def test_jedi_rejects_single_point_and_horns(self) -> None:
        """Verify single pointing finger and rock-on/horns gestures are rejected."""
        # Pointing (only index extended)
        pointing = create_hand_landmarks({
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": False,
        })
        self.assertFalse(is_two_finger_pose(pointing))

        # Horns / Rock-on (index + pinky extended, middle curled)
        horns = create_hand_landmarks({
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": True,
        })
        self.assertFalse(is_two_finger_pose(horns))

        # Three fingers (index + middle + ring extended)
        three_fingers = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": False,
        })
        self.assertFalse(is_two_finger_pose(three_fingers))


if __name__ == "__main__":
    unittest.main()

