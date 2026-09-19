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
    same_person_max_distance_ratio: float = 5.5


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

    # Distance-based blade scaling
    hand_to_blade_ratio: float = 5.8
    min_blade_length: float = 120.0
    max_blade_length: float = 580.0
    reference_hand_size: float = 95.0

    # Traditional Star Wars BGR colors
    jedi_blue: Tuple[int, int, int] = (255, 90, 20)   # Iconic Jedi Blue
    sith_red: Tuple[int, int, int] = (30, 30, 255)    # Iconic Sith Crimson Red

    # Aliases for dual slots
    left_color: Tuple[int, int, int] = (255, 90, 20)   # Slot 1: Jedi Blue
    right_color: Tuple[int, int, int] = (30, 30, 255)  # Slot 2: Sith Red


@dataclass
class DualIgnitionBoxConfig:
    """Configuration for separate Left and Right on-screen Ignition Chambers."""
    enabled: bool = True
    # Left box (Jedi Blue) in normalized coordinates (0.0 to 1.0)
    left_x_min: float = 0.06
    left_y_min: float = 0.20
    left_x_max: float = 0.40
    left_y_max: float = 0.70

    # Right box (Sith Red) in normalized coordinates (0.0 to 1.0)
    right_x_min: float = 0.60
    right_y_min: float = 0.20
    right_x_max: float = 0.94
    right_y_max: float = 0.70

    # Styling colors (BGR)
    blue_idle: Tuple[int, int, int] = (255, 140, 0)
    blue_glow: Tuple[int, int, int] = (255, 180, 50)
    red_idle: Tuple[int, int, int] = (30, 40, 255)
    red_glow: Tuple[int, int, int] = (80, 80, 255)


@dataclass
class DuelConfig:
    enable_tutorial: bool = True
    countdown_seconds: float = 3.0
    post_disarm_cooldown: float = 3.0
    max_rounds: int = 3
    wins_to_win: int = 2
    strike_min_speed: float = 340.0
    strike_speed_ratio: float = 1.7
    parry_min_angle_deg: float = 30.0
    parry_max_foible_ratio: float = 0.72
    mutual_clash_min_speed: float = 280.0
    mutual_clash_ratio: float = 1.6


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    saber: SaberVisualConfig = field(default_factory=SaberVisualConfig)
    box: DualIgnitionBoxConfig = field(default_factory=DualIgnitionBoxConfig)
    duel: DuelConfig = field(default_factory=DuelConfig)

    # Runtime toggles
    show_glow: bool = True
    show_trail: bool = True
    debug_mode: bool = False
