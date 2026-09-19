"""Hand tracking using MediaPipe Tasks Python Hand Landmarker API."""

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


@dataclass
class HandObservation:
    """Clean internal representation of a detected hand in pixel coordinates."""
    handedness: str  # "Left" or "Right"
    landmarks: List[Tuple[float, float]]  # All 21 landmarks in (x, y) pixel coordinates
    wrist: Tuple[float, float]
    index_mcp: Tuple[float, float]
    index_tip: Tuple[float, float]


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

            wrist = pixel_landmarks[WRIST_IDX]
            index_mcp = pixel_landmarks[INDEX_MCP_IDX]
            index_tip = pixel_landmarks[INDEX_TIP_IDX]

            observations.append(
                HandObservation(
                    handedness=handedness_str,
                    landmarks=pixel_landmarks,
                    wrist=wrist,
                    index_mcp=index_mcp,
                    index_tip=index_tip,
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
    # Hand skeleton connections
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
