"""
MIRA Hand Gesture Analyzer.

Uses MediaPipe Hands (21 landmarks per hand) to detect:
  - Self-touching frequency (anxiety biomarker)
  - Fidgeting score (agitation level)
  - Hand tension / clenched fists (anger indicator)
  - Gesture velocity (urgency / restlessness)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SelfTouchResult:
    """Self-touching detection output."""

    is_touching_face: bool
    touch_frequency_per_min: float
    touch_zone: str  # "forehead" | "cheek" | "mouth" | "chin" | "none"


@dataclass
class HandAnalysisResult:
    """Complete hand/gesture analysis output."""

    self_touch: SelfTouchResult
    fidget_score: float     # 0.0 (still) … 1.0 (constant movement)
    is_clenched: bool
    gesture_velocity: float  # mean displacement per frame
    emotion_signals: list[str]


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmark Indices (21 per hand)
# ---------------------------------------------------------------------------
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_RING_MCP = 13
_PINKY_MCP = 17
_PALM_CENTER = 9  # middle MCP approximates palm center


# ---------------------------------------------------------------------------
# HandGestureAnalyzer
# ---------------------------------------------------------------------------
class HandGestureAnalyzer:
    """Hand gesture and fidgeting analysis from MediaPipe Hands."""

    def __init__(self) -> None:
        self._palm_history: deque[tuple[float, float]] = deque(maxlen=300)  # 30s at 10fps
        self._touch_timestamps: deque[float] = deque(maxlen=200)
        self._current_time_sec: float = 0.0

    def detect_self_touching(
        self,
        hand_landmarks: list[Any],
        face_bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> SelfTouchResult:
        """
        Detect hand proximity to face bounding box.

        Self-touching is a clinically validated anxiety/stress biomarker.

        Args:
            hand_landmarks: MediaPipe 21 landmarks for one hand
            face_bbox: (x_min, y_min, x_max, y_max) normalised coordinates
        """
        if face_bbox is None:
            return SelfTouchResult(
                is_touching_face=False,
                touch_frequency_per_min=self._touch_freq(),
                touch_zone="none",
            )

        fx_min, fy_min, fx_max, fy_max = face_bbox
        face_width = fx_max - fx_min
        face_height = fy_max - fy_min

        # Check fingertips proximity to face
        fingertip_indices = [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]
        closest_zone = "none"
        is_touching = False

        for idx in fingertip_indices:
            tip = hand_landmarks[idx]
            # Touching = within 10% of face bounding box
            margin_x = face_width * 0.10
            margin_y = face_height * 0.10

            if (
                fx_min - margin_x <= tip.x <= fx_max + margin_x
                and fy_min - margin_y <= tip.y <= fy_max + margin_y
            ):
                is_touching = True
                # Determine zone
                face_center_y = (fy_min + fy_max) / 2
                if tip.y < face_center_y - face_height * 0.2:
                    closest_zone = "forehead"
                elif tip.y < face_center_y:
                    closest_zone = "cheek"
                elif tip.y < face_center_y + face_height * 0.2:
                    closest_zone = "mouth"
                else:
                    closest_zone = "chin"
                break

        if is_touching:
            self._touch_timestamps.append(self._current_time_sec)

        return SelfTouchResult(
            is_touching_face=is_touching,
            touch_frequency_per_min=self._touch_freq(),
            touch_zone=closest_zone,
        )

    def calculate_fidget_score(
        self,
        hand_landmarks_sequence: list[list[Any]],
        window_sec: float = 30.0,
    ) -> float:
        """
        Fidget score [0.0, 1.0] from hand landmark displacement variance.

        High fidgeting = anxiety / restlessness.
        """
        if len(hand_landmarks_sequence) < 3:
            return 0.0

        palm_positions: list[np.ndarray] = []
        for frame_lms in hand_landmarks_sequence:
            if len(frame_lms) > _PALM_CENTER:
                palm = frame_lms[_PALM_CENTER]
                palm_positions.append(np.array([palm.x, palm.y]))

        if len(palm_positions) < 3:
            return 0.0

        # Frame-to-frame displacement
        displacements: list[float] = []
        for frame_idx in range(1, len(palm_positions)):
            displacement = float(np.linalg.norm(
                palm_positions[frame_idx] - palm_positions[frame_idx - 1]
            ))
            displacements.append(displacement)

        # Standard deviation of displacement = fidgeting
        std_displacement = float(np.std(displacements))

        # Normalise to 0-1 (empirical: std > 0.02 = very fidgety)
        fidget_score = float(np.clip(std_displacement / 0.02, 0.0, 1.0))
        return fidget_score

    def detect_hand_tension(self, hand_landmarks: list[Any]) -> bool:
        """
        Detect clenched fist via finger curl ratio.

        Curl = fingertip closer to wrist than MCP joint → clenched.
        """
        if len(hand_landmarks) < 21:
            return False

        wrist = np.array([hand_landmarks[_WRIST].x, hand_landmarks[_WRIST].y])
        tips = [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]
        mcps = [_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]

        curled_count = 0
        for tip_idx, mcp_idx in zip(tips, mcps):
            tip_pos = np.array([hand_landmarks[tip_idx].x, hand_landmarks[tip_idx].y])
            mcp_pos = np.array([hand_landmarks[mcp_idx].x, hand_landmarks[mcp_idx].y])

            tip_dist = float(np.linalg.norm(tip_pos - wrist))
            mcp_dist = float(np.linalg.norm(mcp_pos - wrist))

            if tip_dist < mcp_dist:
                curled_count += 1

        # Clenched = 3+ fingers curled
        return curled_count >= 3

    def gesture_velocity(self, hand_landmarks_sequence: list[list[Any]]) -> float:
        """
        Mean Euclidean displacement of palm center between consecutive frames.

        High velocity → agitation, urgency, manic behaviour.
        """
        if len(hand_landmarks_sequence) < 2:
            return 0.0

        velocities: list[float] = []
        prev_pos: Optional[np.ndarray] = None

        for frame_lms in hand_landmarks_sequence:
            if len(frame_lms) > _PALM_CENTER:
                palm = frame_lms[_PALM_CENTER]
                current_pos = np.array([palm.x, palm.y])
                if prev_pos is not None:
                    velocity = float(np.linalg.norm(current_pos - prev_pos))
                    velocities.append(velocity)
                prev_pos = current_pos

        return float(np.mean(velocities)) if velocities else 0.0

    def analyze(
        self,
        hand_landmarks: list[Any],
        hand_sequence: list[list[Any]],
        face_bbox: Optional[tuple[float, float, float, float]] = None,
        current_time_sec: float = 0.0,
    ) -> HandAnalysisResult:
        """Full hand/gesture analysis pipeline."""
        self._current_time_sec = current_time_sec

        self_touch = self.detect_self_touching(hand_landmarks, face_bbox)
        fidget_score = self.calculate_fidget_score(hand_sequence)
        is_clenched = self.detect_hand_tension(hand_landmarks)
        velocity = self.gesture_velocity(hand_sequence)

        emotion_signals: list[str] = []

        if self_touch.is_touching_face and self_touch.touch_frequency_per_min > 5:
            emotion_signals.append("ANXIETY_SELF_TOUCH: frequent face-touching")

        if fidget_score > 0.6:
            emotion_signals.append("AGITATION: high fidgeting score")

        if is_clenched:
            emotion_signals.append("ANGER_TENSION: clenched fist detected")

        if velocity > 0.015:
            emotion_signals.append("RESTLESSNESS: high gesture velocity")

        if fidget_score < 0.05 and velocity < 0.002:
            emotion_signals.append("STILLNESS: minimal hand movement — possible blunting")

        return HandAnalysisResult(
            self_touch=self_touch,
            fidget_score=fidget_score,
            is_clenched=is_clenched,
            gesture_velocity=velocity,
            emotion_signals=emotion_signals,
        )

    def _touch_freq(self) -> float:
        """Touches per minute over the last 60 seconds."""
        if not self._touch_timestamps:
            return 0.0
        cutoff = self._current_time_sec - 60.0
        recent = [ts for ts in self._touch_timestamps if ts > cutoff]
        return float(len(recent))
