"""Hand sign and gesture recognition for lightsaber ignition and hand assignment."""

import math
from typing import Dict, List, Optional, Tuple

Point2D = Tuple[float, float]

WRIST_IDX = 0
THUMB_TIP_IDX = 4
THUMB_IP_IDX = 3
INDEX_TIP_IDX = 8
INDEX_PIP_IDX = 6
INDEX_MCP_IDX = 5
MIDDLE_TIP_IDX = 12
MIDDLE_PIP_IDX = 10
MIDDLE_MCP_IDX = 9
RING_TIP_IDX = 16
RING_PIP_IDX = 14
RING_MCP_IDX = 13
PINKY_TIP_IDX = 20
PINKY_PIP_IDX = 18
PINKY_MCP_IDX = 17


def is_finger_extended(
    landmarks: List[Point2D],
    tip_idx: int,
    pip_idx: int,
    wrist_idx: int = WRIST_IDX,
    ratio_threshold: float = 1.20,
) -> bool:
    """Determine if a finger is extended based on tip vs PIP distance from wrist."""
    wrist = landmarks[wrist_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    d_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
    d_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])

    return d_tip > ratio_threshold * max(10.0, d_pip)


def is_finger_curled(
    landmarks: List[Point2D],
    tip_idx: int,
    pip_idx: int,
    wrist_idx: int = WRIST_IDX,
    ratio_threshold: float = 1.10,
) -> bool:
    """Determine if a finger is curled into the palm."""
    wrist = landmarks[wrist_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    d_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
    d_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])

    return d_tip < ratio_threshold * max(10.0, d_pip)


def is_two_finger_pose(landmarks: List[Point2D]) -> bool:
    """
    Check for Obi-Wan's iconic Two-Finger Focus / Peace pose (Jedi Blue).
    Index and Middle fingers are extended; Ring and Pinky are curled into palm.
    Done directly in front of camera with the hand (no full body pose required).
    """
    if len(landmarks) < 21:
        return False

    index_ext = is_finger_extended(landmarks, INDEX_TIP_IDX, INDEX_PIP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX)
    ring_curled = is_finger_curled(landmarks, RING_TIP_IDX, RING_PIP_IDX)
    pinky_curled = is_finger_curled(landmarks, PINKY_TIP_IDX, PINKY_PIP_IDX)

    return index_ext and middle_ext and ring_curled and pinky_curled


def is_force_push_pose(landmarks: List[Point2D]) -> bool:
    """
    Check for the Force Push / Open Palm pose (Sith Red).
    All 4 fingers (Index, Middle, Ring, Pinky) are extended and spread outward.
    """
    if len(landmarks) < 21:
        return False

    index_ext = is_finger_extended(landmarks, INDEX_TIP_IDX, INDEX_PIP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX)
    ring_ext = is_finger_extended(landmarks, RING_TIP_IDX, RING_PIP_IDX)
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP_IDX, PINKY_PIP_IDX)

    return index_ext and middle_ext and ring_ext and pinky_ext


class GestureIgnitionTracker:
    """
    Tracks hand gestures across frames to trigger saber ignition on the specific hand.
    
    Prevents instantaneous false triggers by requiring the gesture to be held
    for a brief confirmation window (~0.18s / 3-4 consecutive frames).
    """

    def __init__(self, required_frames: int = 4) -> None:
        self.required_frames = required_frames
        self.blue_held_frames: int = 0
        self.red_held_frames: int = 0

    def check_ignition(
        self,
        landmarks: List[Point2D],
    ) -> Optional[str]:
        """
        Check if the hand is holding an ignition sign.
        Returns:
            "JediBlue" if Two-Finger pose held,
            "SithRed" if Force Push pose held,
            or None.
        """
        if is_two_finger_pose(landmarks):
            self.blue_held_frames += 1
            self.red_held_frames = 0
            if self.blue_held_frames >= self.required_frames:
                return "JediBlue"
        elif is_force_push_pose(landmarks):
            self.red_held_frames += 1
            self.blue_held_frames = 0
            if self.red_held_frames >= self.required_frames:
                return "SithRed"
        else:
            self.blue_held_frames = max(0, self.blue_held_frames - 1)
            self.red_held_frames = max(0, self.red_held_frames - 1)

        return None

    def reset(self) -> None:
        self.blue_held_frames = 0
        self.red_held_frames = 0
