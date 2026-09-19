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
    same_person_max_distance_ratio: float = 3.5


@dataclass
class SaberVisualConfig:
    base_blade_length: float = 550.0  # Reduced by 15% from 650px
    reference_height: float = 720.0
    trail_duration: float = 0.22  # Fast fade (down from 0.38s)
    trail_leading_opacity: float = 0.75  # Reduced by 20% from 0.95
    trail_decay_exponent: float = 1.35  # Steeper exponential dropoff
    hilt_length: float = 55.0
    alpha_pivot: float = 0.65
    alpha_direction: float = 0.60
    grace_period: float = 0.20  # seconds before lost hand disappears

    # Traditional Star Wars BGR colors
    jedi_blue: Tuple[int, int, int] = (255, 90, 20)   # Iconic Jedi Blue
    sith_red: Tuple[int, int, int] = (30, 30, 255)    # Iconic Sith Crimson Red

    # Aliases for dual slots
    left_color: Tuple[int, int, int] = (255, 90, 20)   # Slot 1: Jedi Blue
    right_color: Tuple[int, int, int] = (30, 30, 255)  # Slot 2: Sith Red


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    saber: SaberVisualConfig = field(default_factory=SaberVisualConfig)

    # Runtime toggles
    show_glow: bool = True
    show_trail: bool = True
    debug_mode: bool = False
