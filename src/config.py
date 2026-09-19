"""Configuration dataclasses and visual defaults for the Lightsabers application."""

import pathlib
from dataclasses import dataclass, field
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    flip_horizontal: bool = True


@dataclass
class TrackerConfig:
    model_path: str = str(PROJECT_ROOT / "assets" / "hand_landmarker.task")
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class SaberVisualConfig:
    base_blade_length: float = 320.0
    reference_height: float = 720.0
    trail_duration: float = 0.35  # seconds
    hilt_length: float = 50.0
    alpha_pivot: float = 0.65
    alpha_direction: float = 0.60
    grace_period: float = 0.25  # seconds before lost hand disappears

    # BGR colors
    left_color: Tuple[int, int, int] = (255, 120, 30)   # Jedi Guardian Blue/Cyan
    right_color: Tuple[int, int, int] = (30, 255, 120)  # Jedi Consular Emerald Green


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    saber: SaberVisualConfig = field(default_factory=SaberVisualConfig)

    # Runtime toggles
    show_glow: bool = True
    show_trail: bool = True
    debug_mode: bool = False
