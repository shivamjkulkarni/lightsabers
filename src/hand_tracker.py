"""Hand tracking using MediaPipe Tasks Python Hand Landmarker API."""

import math
import os
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Landmark indices in MediaPipe Hands
WRIST_IDX = 0
INDEX_MCP_IDX = 5
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9
MIDDLE_TIP_IDX = 12
RING_MCP_IDX = 13
PINKY_MCP_IDX = 17


@dataclass
class HandObservation:
    """Clean internal representation of a detected hand in pixel coordinates."""
    handedness: str  # "Left" or "Right"
    landmarks: List[Tuple[float, float]]  # All 21 landmarks in (x, y) pixel coordinates
    wrist: Tuple[float, float]
    index_mcp: Tuple[float, float]
    index_tip: Tuple[float, float]
    middle_mcp: Tuple[float, float]
    knuckles_center: Tuple[float, float]
    palm_center: Tuple[float, float]
    hand_size: float = 65.0  # 3D tilt-invariant physical hand span in pixels
    landmarks_3d: Optional[List[Tuple[float, float, float]]] = None


class HandTracker:
    """Wrapper around MediaPipe Tasks HandLandmarker in VIDEO running mode."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if model_path is None:
            project_root = pathlib.Path(__file__).resolve().parent.parent
            model_path = str(project_root / "assets" / "hand_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{model_path}'.\n"
                "Please download 'hand_landmarker.task' into the 'assets/' directory."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> List[HandObservation]:
        """Detect hands in the given BGR frame and return HandObservation objects."""
        if self._landmarker is None:
            return []

        h, w = frame.shape[:2]

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Enforce monotonically increasing timestamps for VIDEO mode
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        observations: List[HandObservation] = []
        if not detection_result.hand_landmarks:
            return observations

        for i, landmarks_proto in enumerate(detection_result.hand_landmarks):
            handedness_str = "Unknown"
            if detection_result.handedness and i < len(detection_result.handedness):
                handedness_str = detection_result.handedness[i][0].category_name

            pixel_landmarks: List[Tuple[float, float]] = [
                (lm.x * w, lm.y * h) for lm in landmarks_proto
            ]
            landmarks_3d: List[Tuple[float, float, float]] = [
                (lm.x * w, lm.y * h, getattr(lm, "z", 0.0) * w) for lm in landmarks_proto
            ]

            wrist = pixel_landmarks[WRIST_IDX]
            index_mcp = pixel_landmarks[INDEX_MCP_IDX]
            index_tip = pixel_landmarks[INDEX_TIP_IDX]
            middle_mcp = pixel_landmarks[MIDDLE_MCP_IDX]
            ring_mcp = pixel_landmarks[RING_MCP_IDX]
            pinky_mcp = pixel_landmarks[PINKY_MCP_IDX]

            # Center of the 4 knuckles (MCP joints)
            knuckles_center = (
                (index_mcp[0] + middle_mcp[0] + ring_mcp[0] + pinky_mcp[0]) * 0.25,
                (index_mcp[1] + middle_mcp[1] + ring_mcp[1] + pinky_mcp[1]) * 0.25,
            )

            # Center of the palm (midpoint between wrist and knuckles)
            palm_center = (
                (wrist[0] + knuckles_center[0]) * 0.5,
                (wrist[1] + knuckles_center[1]) * 0.5,
            )

            # Compute 3D tilt-invariant hand physical scale
            lm_wrist = landmarks_proto[WRIST_IDX]
            lm_idx = landmarks_proto[INDEX_MCP_IDX]
            lm_mid = landmarks_proto[MIDDLE_MCP_IDX]
            lm_rng = landmarks_proto[RING_MCP_IDX]
            lm_pky = landmarks_proto[PINKY_MCP_IDX]
            km_x = (lm_idx.x + lm_mid.x + lm_rng.x + lm_pky.x) * 0.25
            km_y = (lm_idx.y + lm_mid.y + lm_rng.y + lm_pky.y) * 0.25
            km_z = getattr(lm_idx, "z", 0.0) + getattr(lm_mid, "z", 0.0) + getattr(lm_rng, "z", 0.0) + getattr(lm_pky, "z", 0.0)
            km_z *= 0.25

            dx = (km_x - lm_wrist.x) * w
            dy = (km_y - lm_wrist.y) * h
            dz = (km_z - getattr(lm_wrist, "z", 0.0)) * w
            hand_size = max(15.0, math.sqrt(dx * dx + dy * dy + dz * dz))

            observations.append(
                HandObservation(
                    handedness=handedness_str,
                    landmarks=pixel_landmarks,
                    wrist=wrist,
                    index_mcp=index_mcp,
                    index_tip=index_tip,
                    middle_mcp=middle_mcp,
                    knuckles_center=knuckles_center,
                    palm_center=palm_center,
                    hand_size=hand_size,
                    landmarks_3d=landmarks_3d,
                )
            )

        return observations

    def close(self) -> None:
        """Close the landmarker and release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_hand_landmarks(frame: np.ndarray, observations: List[HandObservation]) -> None:
    """Simple debug visualization: draw landmark dots, skeleton lines, and handedness."""
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    for obs in observations:
        pts = [(int(x), int(y)) for x, y in obs.landmarks]

        # Draw bones
        for start_idx, end_idx in CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 128), 2, cv2.LINE_AA)

        # Draw landmark points
        for i, pt in enumerate(pts):
            if i in (WRIST_IDX, INDEX_MCP_IDX, INDEX_TIP_IDX):
                cv2.circle(frame, pt, 5, (0, 200, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)

        # Draw knuckles center and palm center in debug mode
        k_pt = (int(obs.knuckles_center[0]), int(obs.knuckles_center[1]))
        cv2.circle(frame, k_pt, 6, (255, 0, 255), -1, cv2.LINE_AA)

        # Label handedness above wrist
        wrist_pt = (int(obs.wrist[0]), int(obs.wrist[1]))
        label_pos = (wrist_pt[0] - 30, wrist_pt[1] - 15)
        cv2.putText(
            frame,
            f"{obs.handedness} Hand",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def filter_one_hand_per_person(
    observations: List[HandObservation],
    same_person_max_distance_ratio: float = 5.5,
) -> List[HandObservation]:
    """
    Ensure only one hand per person is tracked.
    
    If two detected hands belong to the same person (wrists within typical arm span),
    suppress the secondary hand and keep only the primary dominant hand.
    If hands are far apart (distinct people in frame), returns both.
    """
    if len(observations) <= 1:
        return observations

    h1, h2 = observations[0], observations[1]
    wrist_dist = math.hypot(h1.wrist[0] - h2.wrist[0], h1.wrist[1] - h2.wrist[1])

    size1 = getattr(h1, "hand_size", math.hypot(h1.knuckles_center[0] - h1.wrist[0], h1.knuckles_center[1] - h1.wrist[1]))
    size2 = getattr(h2, "hand_size", math.hypot(h2.knuckles_center[0] - h2.wrist[0], h2.knuckles_center[1] - h2.wrist[1]))
    avg_size = max(10.0, (size1 + size2) * 0.5)

    # Distance threshold representing arm-span of a single person in frame
    threshold = same_person_max_distance_ratio * avg_size

    if wrist_dist < threshold:
        # Same person: pick the dominant hand (prefer hand held higher in frame)
        y_diff = h1.wrist[1] - h2.wrist[1]
        if abs(y_diff) > 25.0:
            dominant = h1 if y_diff < 0 else h2
        else:
            # Roughly level: prefer Right hand, else first
            dominant = h1 if h1.handedness == "Right" else h2
        return [dominant]

    # Distant hands = two distinct people in frame
    return observations[:2]


class TwoPersonTracker:
    """
    Stateful tracker that maintains distinct player identities across frames.
    
    Prevents the 'clash proximity person-merging' bug: when two distinct players
    cross blades, their hands can temporarily be close together. This tracker
    uses spatial hysteresis and trajectory history to keep both players active
    during clashes, while ensuring a single person in the room is strictly
    limited to 1 lightsaber.
    """

    def __init__(
        self,
        same_person_max_distance_ratio: float = 5.5,
        confirmation_distance_ratio: float = 4.2,
        loss_timeout: float = 0.6,
    ) -> None:
        self.same_person_max_distance_ratio = same_person_max_distance_ratio
        self.confirmation_distance_ratio = confirmation_distance_ratio
        self.loss_timeout = loss_timeout

        self.two_people_confirmed: bool = False
        self.last_two_people_time: float = 0.0
        self.p1_x: Optional[float] = None
        self.p2_x: Optional[float] = None

    def update(
        self,
        raw_observations: List[HandObservation],
        curr_time: float,
    ) -> List[HandObservation]:
        """Filter and map observations to at most 2 distinct people."""
        if not raw_observations:
            if curr_time - self.last_two_people_time > self.loss_timeout:
                self.two_people_confirmed = False
                self.p1_x = None
                self.p2_x = None
            return []

        if len(raw_observations) == 1:
            if curr_time - self.last_two_people_time > self.loss_timeout:
                self.two_people_confirmed = False
            obs = raw_observations[0]
            self.p1_x = obs.wrist[0]
            return [obs]

        # Two or more hands detected
        h1, h2 = raw_observations[0], raw_observations[1]
        wrist_dist = math.hypot(h1.wrist[0] - h2.wrist[0], h1.wrist[1] - h2.wrist[1])
        avg_size = max(10.0, (getattr(h1, "hand_size", 65.0) + getattr(h2, "hand_size", 65.0)) * 0.5)

        arm_span_threshold = self.same_person_max_distance_ratio * avg_size
        confirm_threshold = self.confirmation_distance_ratio * avg_size

        # If wrists are far apart, confirm 2 distinct people
        if wrist_dist >= confirm_threshold:
            self.two_people_confirmed = True
            self.last_two_people_time = curr_time

        # If 2 people were previously confirmed and hasn't timed out, maintain both during clash
        if self.two_people_confirmed:
            if curr_time - self.last_two_people_time <= self.loss_timeout:
                self.last_two_people_time = curr_time
                obs_sorted = sorted([h1, h2], key=lambda o: o.wrist[0])
                self.p1_x = obs_sorted[0].wrist[0]
                self.p2_x = obs_sorted[1].wrist[0]
                return obs_sorted
            else:
                self.two_people_confirmed = False

        # In 1-person mode: if wrists are within single-person arm span, pick dominant hand
        if wrist_dist < arm_span_threshold:
            y_diff = h1.wrist[1] - h2.wrist[1]
            if abs(y_diff) > 25.0:
                dominant = h1 if y_diff < 0 else h2
            else:
                dominant = h1 if h1.handedness == "Right" else h2
            self.p1_x = dominant.wrist[0]
            return [dominant]

        # Hands beyond arm span -> 2 people
        self.two_people_confirmed = True
        self.last_two_people_time = curr_time
        obs_sorted = sorted([h1, h2], key=lambda o: o.wrist[0])
        self.p1_x = obs_sorted[0].wrist[0]
        self.p2_x = obs_sorted[1].wrist[0]
        return obs_sorted

    def reset(self) -> None:
        self.two_people_confirmed = False
        self.last_two_people_time = 0.0
        self.p1_x = None
        self.p2_x = None
