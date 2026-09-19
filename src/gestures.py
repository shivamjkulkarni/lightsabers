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
    threshold: float = 1.15,
) -> bool:
    """
    Determine if a finger is extended.
    Supports planar 2D extension, 3D Euclidean extension, and forward camera (-Z) pointing.
    Works seamlessly with 3D or 2D landmarks.
    """
    if len(landmarks) <= max(tip_idx, pip_idx, mcp_idx, wrist_idx):
        return False

    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    # 2D planar distances
    d2_wt = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
    d2_wp = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
    d2_mt = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])
    d2_mp = math.hypot(pip[0] - mcp[0], pip[1] - mcp[1])

    # Hand scale proxy (wrist to MCP)
    hand_scale = math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1])
    if hand_scale < 10.0:
        hand_scale = 60.0

    has_3d = len(tip) >= 3 and len(pip) >= 3 and len(mcp) >= 3 and len(wrist) >= 3
    forward_ext = False
    if has_3d:
        d3_wt = math.sqrt((tip[0] - wrist[0]) ** 2 + (tip[1] - wrist[1]) ** 2 + (tip[2] - wrist[2]) ** 2)
        d3_wp = math.sqrt((pip[0] - wrist[0]) ** 2 + (pip[1] - wrist[1]) ** 2 + (pip[2] - wrist[2]) ** 2)
        d3_mt = math.sqrt((tip[0] - mcp[0]) ** 2 + (tip[1] - mcp[1]) ** 2 + (tip[2] - mcp[2]) ** 2)
        d3_mp = math.sqrt((pip[0] - mcp[0]) ** 2 + (pip[1] - mcp[1]) ** 2 + (pip[2] - mcp[2]) ** 2)

        # Forward extension towards camera: tip is forward in -Z relative to MCP and PIP
        forward_ext = (mcp[2] - tip[2] > 0.18 * hand_scale) and (pip[2] - tip[2] > 0.10 * hand_scale)
    else:
        d3_wt, d3_wp, d3_mt, d3_mp = d2_wt, d2_wp, d2_mt, d2_mp

    planar_ext = (
        (d2_wt > 1.08 * d2_wp)
        or (d2_mt > threshold * d2_mp)
        or (d3_wt > 1.08 * d3_wp)
        or (d3_mt > threshold * d3_mp)
    )

    return planar_ext or forward_ext


def is_finger_curled(
    landmarks: Sequence[Point],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
    wrist_idx: int = WRIST_IDX,
    threshold: float = 1.30,
) -> bool:
    """
    Determine if a finger is curled into the palm.
    When curled, the tip folds back towards the palm/MCP joint and does not point forward.
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
    wrist_curled = d_wrist_tip < 1.10 * max(1.0, d_wrist_pip)

    # If 3D, verify tip is not pointing forward towards camera
    if len(tip) >= 3 and len(mcp) >= 3:
        if mcp[2] - tip[2] > 25.0:
            return False

    return mcp_curled or wrist_curled


def is_two_finger_pose(landmarks: Sequence[Point]) -> bool:
    """
    Check for Obi-Wan's iconic Two-Finger Focus / Peace pose (Jedi Blue).
    - Index and Middle fingers are extended (either in 2D plane or forward in 3D).
    - Index and Middle fingers are significantly more extended than Ring and Pinky fingers.
    - Ring and Pinky fingers are NOT both extended (rejects open palm / force push).
    Invariant to camera distance, tilt angle, and supports natural relaxed finger curling.
    """
    if len(landmarks) < 21:
        return False

    wrist = landmarks[WRIST_IDX]
    index_tip = landmarks[INDEX_TIP_IDX]
    index_pip = landmarks[INDEX_PIP_IDX]
    index_mcp = landmarks[INDEX_MCP_IDX]

    middle_tip = landmarks[MIDDLE_TIP_IDX]
    middle_pip = landmarks[MIDDLE_PIP_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    ring_tip = landmarks[RING_TIP_IDX]
    ring_pip = landmarks[RING_PIP_IDX]
    ring_mcp = landmarks[RING_MCP_IDX]

    pinky_tip = landmarks[PINKY_TIP_IDX]
    pinky_pip = landmarks[PINKY_PIP_IDX]
    pinky_mcp = landmarks[PINKY_MCP_IDX]

    # 1. Primary extension checks for Index and Middle
    index_ext = is_finger_extended(landmarks, INDEX_TIP_IDX, INDEX_PIP_IDX, INDEX_MCP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP_IDX, MIDDLE_PIP_IDX, MIDDLE_MCP_IDX)

    if not (index_ext and middle_ext):
        return False

    # 2. Ring and Pinky extension checks (neither should be fully extended)
    ring_ext = is_finger_extended(landmarks, RING_TIP_IDX, RING_PIP_IDX, RING_MCP_IDX)
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP_IDX, PINKY_PIP_IDX, PINKY_MCP_IDX)

    # If both ring and pinky are extended, it's an open palm / force push!
    if ring_ext and pinky_ext:
        return False

    # 3. Relative extension contrast (Index/Middle must clearly dominate Ring/Pinky)
    d_w_idx = _distance(wrist, index_tip)
    d_w_mid = _distance(wrist, middle_tip)
    d_w_rng = _distance(wrist, ring_tip)
    d_w_pky = _distance(wrist, pinky_tip)

    hand_scale = _distance(wrist, middle_mcp)
    if hand_scale < 10.0:
        hand_scale = 60.0

    idx_over_rng = d_w_idx > 1.10 * d_w_rng
    mid_over_pky = d_w_mid > 1.10 * d_w_pky

    has_3d = len(wrist) >= 3 and len(index_tip) >= 3 and len(ring_tip) >= 3
    if has_3d:
        z_idx_over_rng = (ring_tip[2] - index_tip[2]) > 0.18 * hand_scale
        z_mid_over_pky = (pinky_tip[2] - middle_tip[2]) > 0.18 * hand_scale
        idx_over_rng = idx_over_rng or z_idx_over_rng
        mid_over_pky = mid_over_pky or z_mid_over_pky

    return idx_over_rng and mid_over_pky


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
    palm_center: Optional[Tuple[float, float]] = None,
    padding: float = 35.0,
) -> bool:
    """
    Check whether a hand's physical center (knuckles, palm, or wrist) is inside the on-screen box.
    Includes a comfort padding margin so hands near chamber edges are smoothly accepted.
    box_rect is (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    bx1, by1, bx2, by2 = box_rect
    bx1 -= padding
    by1 -= padding
    bx2 += padding
    by2 += padding

    kx, ky = knuckles
    wx, wy = wrist

    in_knuckles = (bx1 <= kx <= bx2) and (by1 <= ky <= by2)
    in_wrist = (bx1 <= wx <= bx2) and (by1 <= wy <= by2)
    in_palm = False
    if palm_center is not None:
        px, py = palm_center
        in_palm = (bx1 <= px <= bx2) and (by1 <= py <= by2)

    return in_knuckles or in_wrist or in_palm


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
