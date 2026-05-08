"""
MIRA Posture Analyzer.

Uses MediaPipe Pose (33 keypoints) to measure:
  - Shoulder slope (tension / pain indicator)
  - Head tilt (engagement / rejection)
  - Slouching severity (depression / fatigue)
  - Body openness (approach / avoidance)

Posture → emotion mapping follows clinical body-language research.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SlouchResult:
    """Slouching detection output."""

    is_slouching: bool
    severity: int  # 0 = none, 1 = mild, 2 = moderate, 3 = severe
    spine_angle: float  # degrees — normal: 160-180


@dataclass
class PostureAnalysisResult:
    """Complete posture analysis output."""

    shoulder_slope: float       # degrees — 0 = level, >0 = left higher
    head_tilt: float            # degrees — lateral tilt
    slouch: SlouchResult
    body_openness: float        # 0.0 (closed/crossed) … 1.0 (open)
    emotion_signals: list[str]  # clinical interpretations


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmark Indices (33 keypoints)
# ---------------------------------------------------------------------------
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24


# ---------------------------------------------------------------------------
# PostureAnalyzer
# ---------------------------------------------------------------------------
class PostureAnalyzer:
    """Posture analysis from MediaPipe Pose 33 keypoints."""

    def calculate_shoulder_slope(self, pose_landmarks: list[Any]) -> float:
        """
        Angle between left (11) and right (12) shoulder landmarks.

        0° = perfectly level. Positive = left shoulder higher.
        Nonzero slope → tension, asymmetry, guarding pain.
        """
        left = pose_landmarks[_LEFT_SHOULDER]
        right = pose_landmarks[_RIGHT_SHOULDER]

        dy = left.y - right.y
        dx = right.x - left.x  # positive = right is further right
        if abs(dx) < 1e-6:
            return 0.0

        angle_rad = math.atan2(dy, dx)
        return float(math.degrees(angle_rad))

    def calculate_head_tilt(self, pose_landmarks: list[Any]) -> float:
        """
        Angle of nose (0) relative to the midpoint of shoulders.

        Lateral tilt → curiosity or confusion.
        Forward lean → engagement.
        Backward lean → rejection / disgust.
        """
        nose = pose_landmarks[_NOSE]
        left_sh = pose_landmarks[_LEFT_SHOULDER]
        right_sh = pose_landmarks[_RIGHT_SHOULDER]

        mid_shoulder_x = (left_sh.x + right_sh.x) / 2
        mid_shoulder_y = (left_sh.y + right_sh.y) / 2

        dx = nose.x - mid_shoulder_x
        dy = mid_shoulder_y - nose.y  # invert y (screen coords: y increases downward)

        angle_rad = math.atan2(dx, dy)
        return float(math.degrees(angle_rad))

    def detect_slouching(self, pose_landmarks: list[Any]) -> SlouchResult:
        """
        Spine angle: midpoint of shoulders → midpoint of hips.

        Normal: 160–180°. Slouching: < 150°.
        """
        left_sh = pose_landmarks[_LEFT_SHOULDER]
        right_sh = pose_landmarks[_RIGHT_SHOULDER]
        left_hip = pose_landmarks[_LEFT_HIP]
        right_hip = pose_landmarks[_RIGHT_HIP]

        mid_shoulder = np.array([
            (left_sh.x + right_sh.x) / 2,
            (left_sh.y + right_sh.y) / 2,
            (left_sh.z + right_sh.z) / 2,
        ])
        mid_hip = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2,
        ])

        # Spine vector
        spine_vec = mid_shoulder - mid_hip
        vertical = np.array([0.0, -1.0, 0.0])  # up in screen space

        cos_angle = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) * np.linalg.norm(vertical) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        spine_angle = float(math.degrees(math.acos(cos_angle)))

        # Map to severity
        if spine_angle >= 160:
            severity = 0
        elif spine_angle >= 150:
            severity = 1
        elif spine_angle >= 135:
            severity = 2
        else:
            severity = 3

        return SlouchResult(
            is_slouching=severity > 0,
            severity=severity,
            spine_angle=spine_angle,
        )

    def calculate_body_openness(self, pose_landmarks: list[Any]) -> float:
        """
        Body openness score [0.0, 1.0]:
          - Arms crossed (wrists cross midline) → 0.0
          - Open (wrists outside shoulder width) → 1.0
          - Facing forward (balanced z-coordinates) → bonus
        """
        left_sh = pose_landmarks[_LEFT_SHOULDER]
        right_sh = pose_landmarks[_RIGHT_SHOULDER]
        left_wrist = pose_landmarks[_LEFT_WRIST]
        right_wrist = pose_landmarks[_RIGHT_WRIST]

        # Shoulder width
        shoulder_width = abs(left_sh.x - right_sh.x)
        if shoulder_width < 1e-6:
            return 0.5

        # Wrist spread relative to shoulders
        wrist_spread = abs(left_wrist.x - right_wrist.x)
        spread_ratio = wrist_spread / shoulder_width

        # Crossed arms: both wrists near/past midline
        midline_x = (left_sh.x + right_sh.x) / 2
        left_crossed = left_wrist.x > midline_x  # left wrist on right side
        right_crossed = right_wrist.x < midline_x  # right wrist on left side

        if left_crossed and right_crossed:
            return 0.0

        # Openness from spread ratio
        openness = float(np.clip(spread_ratio / 1.5, 0.0, 1.0))

        # Z-balance bonus (facing forward)
        z_diff = abs(left_sh.z - right_sh.z)
        if z_diff < 0.05:
            openness = min(openness + 0.1, 1.0)

        return openness

    def analyze(self, pose_landmarks: list[Any]) -> PostureAnalysisResult:
        """Full posture analysis pipeline."""
        shoulder_slope = self.calculate_shoulder_slope(pose_landmarks)
        head_tilt = self.calculate_head_tilt(pose_landmarks)
        slouch = self.detect_slouching(pose_landmarks)
        body_openness = self.calculate_body_openness(pose_landmarks)

        # Clinical emotion signals
        emotion_signals: list[str] = []

        if slouch.severity >= 2 and body_openness < 0.3:
            emotion_signals.append("DEPRESSION_POSTURE: hunched + closed body")

        if abs(shoulder_slope) > 10:
            emotion_signals.append("TENSION: asymmetric shoulders — possible guarding/stress")

        if slouch.severity == 0 and body_openness > 0.7:
            emotion_signals.append("CONFIDENCE: upright + open posture")

        if head_tilt > 15:
            emotion_signals.append("CURIOSITY: forward head tilt — engagement signal")
        elif head_tilt < -10:
            emotion_signals.append("WITHDRAWAL: backward head lean — rejection/avoidance")

        if slouch.severity >= 2 and body_openness < 0.2:
            emotion_signals.append("EMOTIONAL_BLUNTING: collapsed posture + closed body")

        return PostureAnalysisResult(
            shoulder_slope=shoulder_slope,
            head_tilt=head_tilt,
            slouch=slouch,
            body_openness=body_openness,
            emotion_signals=emotion_signals,
        )
