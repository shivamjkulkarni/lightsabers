"""Hand sign and gesture recognition for lightsaber ignition and hand assignment."""

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, ...]

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


def _distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between 2D or 3D points."""
    if len(p1) >= 3 and len(p2) >= 3:
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_finger_extended(
    landmarks: Sequence[Point],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
    wrist_idx: int = WRIST_IDX,
    threshold: float = 1.25,
) -> bool:
    """
    Determine if a finger is extended.
    Uses anatomical MCP-relative ratio (tip to MCP vs PIP to MCP).
    Also supports wrist-to-tip vs wrist-to-PIP for planar poses.
    Works seamlessly with 3D or 2D landmarks.
    """
    if len(landmarks) <= max(tip_idx, pip_idx, mcp_idx, wrist_idx):
        return False

    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    d_mcp_tip = _distance(tip, mcp)
    d_mcp_pip = _distance(pip, mcp)
    d_wrist_tip = _distance(tip, wrist)
    d_wrist_pip = _distance(pip, wrist)

    mcp_extended = d_mcp_tip > threshold * max(1.0, d_mcp_pip)
    wrist_extended = d_wrist_tip > 1.12 * max(1.0, d_wrist_pip)

    return mcp_extended or wrist_extended


def is_finger_curled(
    landmarks: Sequence[Point],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
    wrist_idx: int = WRIST_IDX,
    threshold: float = 1.15,
) -> bool:
    """
    Determine if a finger is curled into the palm.
    When curled, the tip folds back towards the palm/MCP joint.
    """
    if len(landmarks) <= max(tip_idx, pip_idx, mcp_idx, wrist_idx):
        return False

    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    d_mcp_tip = _distance(tip, mcp)
    d_mcp_pip = _distance(pip, mcp)
    d_wrist_tip = _distance(tip, wrist)
    d_wrist_pip = _distance(pip, wrist)

    mcp_curled = d_mcp_tip < threshold * max(1.0, d_mcp_pip)
    wrist_curled = d_wrist_tip < 1.05 * max(1.0, d_wrist_pip)

    return mcp_curled or wrist_curled


def is_two_finger_pose(landmarks: Sequence[Point]) -> bool:
    """
    Check for Obi-Wan's iconic Two-Finger Focus / Peace pose (Jedi Blue).
    Index and Middle fingers are extended; Ring and Pinky are curled into palm.
    Invariant to rotation and works directly pointing at camera in 3D.
    """
    if len(landmarks) < 21:
        return False

    index_ext = is_finger_extended(landmarks, INDEX_TIP_IDX, INDEX_PIP_IDX, INDEX_MCP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX, MIDDLE_MCP_IDX)
    ring_curled = is_finger_curled(landmarks, RING_TIP_IDX, RING_PIP_IDX, RING_MCP_IDX)
    pinky_curled = is_finger_curled(landmarks, PINKY_TIP_IDX, PINKY_PIP_IDX, PINKY_MCP_IDX)

    return index_ext and middle_ext and ring_curled and pinky_curled


def is_force_push_pose(landmarks: Sequence[Point]) -> bool:
    """
    Check for the Force Push / Open Palm pose (Sith Red).
    All 4 fingers (Index, Middle, Ring, Pinky) are extended and spread outward.
    Fast and robust under perspective foreshortening.
    """
    if len(landmarks) < 21:
        return False

    index_ext = is_finger_extended(landmarks, INDEX_TIP_IDX, INDEX_PIP_IDX, INDEX_MCP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX, MIDDLE_MCP_IDX)
    ring_ext = is_finger_extended(landmarks, RING_TIP_IDX, RING_PIP_IDX, RING_MCP_IDX)
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP_IDX, PINKY_PIP_IDX, PINKY_MCP_IDX)

    return index_ext and middle_ext and ring_ext and pinky_ext


def is_hand_in_box(
    wrist: Tuple[float, float],
    knuckles: Tuple[float, float],
    box_rect: Tuple[int, int, int, int],
) -> bool:
    """
    Check whether a hand's physical center (knuckles or wrist) is inside the on-screen box.
    box_rect is (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    bx1, by1, bx2, by2 = box_rect
    kx, ky = knuckles
    wx, wy = wrist

    in_knuckles = (bx1 <= kx <= bx2) and (by1 <= ky <= by2)
    in_wrist = (bx1 <= wx <= bx2) and (by1 <= wy <= by2)
    return in_knuckles or in_wrist


class GestureIgnitionTracker:
    """
    Tracks hand gestures across frames to trigger saber ignition on the specific hand.
    Inside the ignition box, requires only 2 consecutive frames for ultra-low latency (< 0.035s).
    """

    def __init__(self, required_frames: int = 2) -> None:
        self.required_frames = required_frames
        self.blue_held_frames: int = 0
        self.red_held_frames: int = 0

    def check_ignition(
        self,
        landmarks: Sequence[Point],
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
