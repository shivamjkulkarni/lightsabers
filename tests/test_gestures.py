"""Unit tests for hand sign and gesture recognition."""

import unittest
from src.gestures import (
    GestureIgnitionTracker,
    is_force_push_pose,
    is_two_finger_pose,
)


def create_hand_landmarks(finger_states: dict) -> list:
    """
    Construct a synthetic 21-landmark skeleton.
    finger_states: dict with keys 'index', 'middle', 'ring', 'pinky' -> bool (True=extended, False=curled)
    """
    wrist = (200.0, 400.0)
    landmarks = [wrist] * 21

    # Finger definitions: (mcp, pip, dip, tip, angle_rad)
    fingers = {
        "thumb": (1, 2, 3, 4, 0.4),
        "index": (5, 6, 7, 8, -0.2),
        "middle": (9, 10, 11, 12, 0.0),
        "ring": (13, 14, 15, 16, 0.2),
        "pinky": (17, 18, 19, 20, 0.4),
    }

    for name, (mcp, pip, dip, tip, angle) in fingers.items():
        is_ext = finger_states.get(name, True)
        # Base MCP
        landmarks[mcp] = (wrist[0] + angle * 40, wrist[1] - 50)
        # PIP
        landmarks[pip] = (landmarks[mcp][0] + angle * 20, landmarks[mcp][1] - 40)
        # DIP
        landmarks[dip] = (landmarks[pip][0] + angle * 10, landmarks[pip][1] - 30)

        if is_ext:
            # Tip extended away from wrist (dist > 1.25 * pip dist)
            landmarks[tip] = (landmarks[dip][0] + angle * 10, landmarks[dip][1] - 40)
        else:
            # Tip curled downward back toward wrist/palm
            landmarks[tip] = (landmarks[mcp][0], landmarks[mcp][1] - 10)

    return landmarks


class TestGestures(unittest.TestCase):
    """Test suite for gesture recognition."""

    def test_two_finger_pose_recognition(self) -> None:
        """Verify Two-Finger pose (Jedi Blue) triggers when Index+Middle extended and Ring+Pinky curled."""
        # Index & Middle extended, Ring & Pinky curled
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertTrue(is_two_finger_pose(landmarks))
        self.assertFalse(is_force_push_pose(landmarks))

    def test_force_push_pose_recognition(self) -> None:
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

    def test_fist_no_gesture(self) -> None:
        """Verify closed fist does not trigger either ignition gesture."""
        landmarks = create_hand_landmarks({
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
            "thumb": False,
        })
        self.assertFalse(is_two_finger_pose(landmarks))
        self.assertFalse(is_force_push_pose(landmarks))

    def test_gesture_ignition_tracker_debounce(self) -> None:
        """Verify tracker requires holding pose for required frame count before ignition."""
        tracker = GestureIgnitionTracker(required_frames=3)
        landmarks = create_hand_landmarks({
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False,
        })

        # Frame 1: not ready
        self.assertIsNone(tracker.check_ignition(landmarks))
        # Frame 2: not ready
        self.assertIsNone(tracker.check_ignition(landmarks))
        # Frame 3: confirmed!
        self.assertEqual(tracker.check_ignition(landmarks), "JediBlue")


if __name__ == "__main__":
    unittest.main()
