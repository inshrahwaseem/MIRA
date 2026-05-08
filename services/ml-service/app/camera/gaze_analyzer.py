"""
MIRA Gaze Analyzer.

Uses Dlib 68-point facial landmarks for precise eye tracking:
  - Eye Aspect Ratio (EAR) → blink detection
  - Blink rate per minute (stress: >30/min)
  - Gaze direction (DIRECT | LEFT | RIGHT | UP | DOWN | AVERTED)
  - Eye contact percentage
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class GazeDirection(str, Enum):
    """Gaze direction classification."""

    DIRECT = "DIRECT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    AVERTED = "AVERTED"


@dataclass
class EyeData:
    """Complete gaze analysis output."""

    gaze_direction: GazeDirection
    blink_rate_per_min: float
    eye_contact_pct: float
    pupil_ratio_estimate: float  # left / right pupil size ratio (1.0 = symmetric)
    ear_left: float
    ear_right: float


# ---------------------------------------------------------------------------
# Constants — Dlib 68-point landmark indices
# ---------------------------------------------------------------------------
# Left eye: 36-41, Right eye: 42-47
_LEFT_EYE_INDICES = list(range(36, 42))
_RIGHT_EYE_INDICES = list(range(42, 48))


# ---------------------------------------------------------------------------
# GazeAnalyzer
# ---------------------------------------------------------------------------
class GazeAnalyzer:
    """
    Eye tracking and blink detection using Dlib 68-point landmarks.

    Can also work with MediaPipe 468 landmarks by mapping to equivalent
    indices (see _MEDIAPIPE_LEFT_EYE / _MEDIAPIPE_RIGHT_EYE below).
    """

    # MediaPipe FaceMesh equivalent eye indices
    _MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
    _MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Blink threshold
    EAR_THRESHOLD = 0.20
    CONSECUTIVE_FRAMES_FOR_BLINK = 3

    def __init__(self) -> None:
        self._blink_timestamps: deque[float] = deque(maxlen=300)  # last 5 min
        self._blink_counter = 0
        self._consecutive_below = 0
        self._gaze_history: deque[GazeDirection] = deque(maxlen=600)  # 60s at 10fps

    def calculate_ear(self, eye_landmarks: list[tuple[float, float]]) -> float:
        """
        Eye Aspect Ratio: (A + B) / (2 * C).

        A, B = vertical distances between eyelid landmarks
        C = horizontal distance between eye corners

        EAR < 0.20 for 3+ frames → blink detected.
        """
        if len(eye_landmarks) < 6:
            return 0.3  # open-eye default

        # Vertical distances
        dist_a = np.linalg.norm(
            np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])
        )
        dist_b = np.linalg.norm(
            np.array(eye_landmarks[2]) - np.array(eye_landmarks[4])
        )
        # Horizontal distance
        dist_c = np.linalg.norm(
            np.array(eye_landmarks[0]) - np.array(eye_landmarks[3])
        )

        if dist_c < 1e-6:
            return 0.3

        ear = float((dist_a + dist_b) / (2.0 * dist_c))
        return ear

    def update_blink(self, ear: float) -> bool:
        """
        Track blink state. Returns True if a new blink was just completed.
        """
        if ear < self.EAR_THRESHOLD:
            self._consecutive_below += 1
        else:
            if self._consecutive_below >= self.CONSECUTIVE_FRAMES_FOR_BLINK:
                self._blink_counter += 1
                self._blink_timestamps.append(time.time())
                self._consecutive_below = 0
                return True
            self._consecutive_below = 0
        return False

    @property
    def blinks_per_minute(self) -> float:
        """Blink rate over the last 60 seconds."""
        now = time.time()
        cutoff = now - 60.0
        recent = [ts for ts in self._blink_timestamps if ts > cutoff]
        return float(len(recent))

    def track_gaze_direction(
        self,
        left_eye_lm: list[tuple[float, float]],
        right_eye_lm: list[tuple[float, float]],
        iris_left: Optional[tuple[float, float]] = None,
        iris_right: Optional[tuple[float, float]] = None,
    ) -> GazeDirection:
        """
        Classify gaze direction based on iris position relative to eye corners.

        If iris landmarks aren't available (no refined mesh), falls back to
        using the midpoint of upper/lower eyelid as a proxy.
        """
        if len(left_eye_lm) < 6 or len(right_eye_lm) < 6:
            return GazeDirection.DIRECT

        # Use provided iris or estimate from eye center
        if iris_left is None:
            iris_left = (
                float(np.mean([pt[0] for pt in left_eye_lm])),
                float(np.mean([pt[1] for pt in left_eye_lm])),
            )
        if iris_right is None:
            iris_right = (
                float(np.mean([pt[0] for pt in right_eye_lm])),
                float(np.mean([pt[1] for pt in right_eye_lm])),
            )

        # Eye bounding box
        left_inner = left_eye_lm[3]  # inner corner
        left_outer = left_eye_lm[0]  # outer corner
        eye_width = abs(left_outer[0] - left_inner[0])
        eye_height = abs(left_eye_lm[1][1] - left_eye_lm[5][1])

        if eye_width < 1e-6:
            return GazeDirection.DIRECT

        # Horizontal displacement (normalised)
        eye_center_x = (left_inner[0] + left_outer[0]) / 2
        horizontal_offset = (iris_left[0] - eye_center_x) / eye_width

        # Vertical displacement
        eye_center_y = (left_eye_lm[1][1] + left_eye_lm[5][1]) / 2
        vertical_offset = (iris_left[1] - eye_center_y) / (eye_height + 1e-6)

        # Thresholds
        averted_threshold = 0.30

        if abs(horizontal_offset) > averted_threshold and abs(vertical_offset) > averted_threshold:
            direction = GazeDirection.AVERTED
        elif horizontal_offset < -0.20:
            direction = GazeDirection.LEFT
        elif horizontal_offset > 0.20:
            direction = GazeDirection.RIGHT
        elif vertical_offset < -0.25:
            direction = GazeDirection.UP
        elif vertical_offset > 0.25:
            direction = GazeDirection.DOWN
        else:
            direction = GazeDirection.DIRECT

        self._gaze_history.append(direction)
        return direction

    def eye_contact_percentage(self) -> float:
        """
        Percentage of recent frames with DIRECT gaze.

        Low eye contact (<30%) → shame, avoidance, social anxiety.
        """
        if not self._gaze_history:
            return 0.0
        direct_count = sum(1 for gaze in self._gaze_history if gaze == GazeDirection.DIRECT)
        return float(direct_count / len(self._gaze_history))

    def analyze_from_mediapipe(self, landmarks: list[Any]) -> EyeData:
        """
        Full gaze analysis using MediaPipe FaceMesh 468 landmarks.
        """
        # Extract eye landmark coordinates
        left_eye = [(landmarks[i].x, landmarks[i].y) for i in self._MP_LEFT_EYE]
        right_eye = [(landmarks[i].x, landmarks[i].y) for i in self._MP_RIGHT_EYE]

        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        avg_ear = (ear_left + ear_right) / 2

        self.update_blink(avg_ear)

        # Iris: MediaPipe refine_landmarks provides iris (468-472 left, 473-477 right)
        iris_left = None
        iris_right = None
        if len(landmarks) > 473:
            iris_left = (landmarks[468].x, landmarks[468].y)
            iris_right = (landmarks[473].x, landmarks[473].y)

        gaze = self.track_gaze_direction(left_eye, right_eye, iris_left, iris_right)

        # Pupil ratio estimate (symmetry from iris landmark sizes)
        pupil_ratio = 1.0
        if iris_left and iris_right:
            left_size = _lm_iris_size(landmarks, [468, 469, 470, 471])
            right_size = _lm_iris_size(landmarks, [473, 474, 475, 476])
            if right_size > 0:
                pupil_ratio = left_size / right_size

        return EyeData(
            gaze_direction=gaze,
            blink_rate_per_min=self.blinks_per_minute,
            eye_contact_pct=self.eye_contact_percentage(),
            pupil_ratio_estimate=pupil_ratio,
            ear_left=ear_left,
            ear_right=ear_right,
        )


def _lm_iris_size(landmarks: list[Any], indices: list[int]) -> float:
    """Approximate iris diameter from 4 boundary landmarks."""
    if len(landmarks) <= max(indices):
        return 1.0
    coords = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    diameter_h = np.linalg.norm(coords[0] - coords[2])
    diameter_v = np.linalg.norm(coords[1] - coords[3])
    return float((diameter_h + diameter_v) / 2)
