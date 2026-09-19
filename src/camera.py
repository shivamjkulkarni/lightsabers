"""Camera abstraction for video capture and frame pre-processing."""

from typing import Optional, Tuple
import cv2
import numpy as np


class Camera:
    """Wrapper around cv2.VideoCapture for reliable webcam input."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        flip_horizontal: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.flip_horizontal = flip_horizontal

        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Open the camera and set desired resolution."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera with index {self.camera_index}. "
                "Check camera permissions and verify that the webcam is connected."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Minimize buffer size to eliminate camera latency/queue lag
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        """Read and return a frame from the camera, or None if read fails."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self) -> None:
        """Release camera resources cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

