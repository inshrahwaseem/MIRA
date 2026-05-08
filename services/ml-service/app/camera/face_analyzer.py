"""
MIRA Face Analyzer.

Uses MediaPipe FaceMesh (468 landmarks) + DeepFace for:
  - 10 Action Unit extraction (AU1, AU4, AU6, AU12, AU15, AU17, AU20, AU23, AU24, AU28)
  - Duchenne vs polite smile detection
  - Micro-expression spotting (200ms window)
  - DeepFace 7-emotion classification

PRIVACY: Raw frames are NEVER saved — only extracted features persist.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SmileAnalysis:
    """Duchenne vs polite smile detection."""

    smile_type: str  # "genuine" | "polite" | "none"
    authenticity_score: float  # 0.0 – 1.0
    confidence: float


@dataclass
class MicroExpression:
    """A fleeting AU spike detected within ~200ms."""

    emotion_name: str
    duration_ms: float
    peak_intensity: float
    au_triggered: str


@dataclass
class FaceAnalysisResult:
    """Complete face analysis output."""

    action_units: dict[str, float]
    smile_analysis: SmileAnalysis
    micro_expressions: list[MicroExpression]
    deepface_emotions: dict[str, float]
    combined_emotions: dict[str, float]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Helper: Euclidean distance between two landmarks
# ---------------------------------------------------------------------------
def _lm_dist(landmarks: list[Any], idx_a: int, idx_b: int) -> float:
    """Euclidean distance between two MediaPipe NormalizedLandmarks."""
    point_a = landmarks[idx_a]
    point_b = landmarks[idx_b]
    return float(np.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    ))


def _lm_y_diff(landmarks: list[Any], idx_a: int, idx_b: int) -> float:
    """Signed vertical (y) displacement: a.y - b.y. Negative = a is above b."""
    return float(landmarks[idx_a].y - landmarks[idx_b].y)


# ---------------------------------------------------------------------------
# FaceAnalyzer
# ---------------------------------------------------------------------------
class FaceAnalyzer:
    """
    MediaPipe FaceMesh + DeepFace for facial emotion analysis.

    Landmark indices follow the 468-point FaceMesh topology:
      https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    """

    # Key landmark indices
    _LEFT_BROW_INNER = 107
    _RIGHT_BROW_INNER = 336
    _LEFT_BROW_OUTER = 70
    _RIGHT_BROW_OUTER = 300
    _LEFT_EYE_TOP = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_TOP = 386
    _RIGHT_EYE_BOTTOM = 374
    _LEFT_CHEEK = 117
    _RIGHT_CHEEK = 346
    _UPPER_LIP_CENTER = 13
    _LOWER_LIP_CENTER = 14
    _LEFT_LIP_CORNER = 61
    _RIGHT_LIP_CORNER = 291
    _CHIN = 152
    _NOSE_TIP = 1
    _LEFT_LIP_LOWER = 78
    _RIGHT_LIP_LOWER = 308

    def __init__(self) -> None:
        self._face_mesh: Any = None
        self._au_history: deque[dict[str, float]] = deque(maxlen=20)  # 2s at 10fps

    def _ensure_face_mesh(self) -> None:
        """Lazy-load MediaPipe FaceMesh."""
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def extract_action_units(self, landmarks: list[Any]) -> dict[str, float]:
        """
        Compute 10 FACS Action Units from FaceMesh 468 landmarks.

        Returns scores in [0.0, 1.0] for each AU.
        """
        # Normalisation reference: inter-pupil distance
        ipd = _lm_dist(landmarks, self._LEFT_EYE_TOP, self._RIGHT_EYE_TOP)
        if ipd < 1e-6:
            ipd = 0.06  # fallback

        # AU1 — Inner Brow Raise
        left_brow_eye = _lm_y_diff(landmarks, self._LEFT_EYE_TOP, self._LEFT_BROW_INNER)
        right_brow_eye = _lm_y_diff(landmarks, self._RIGHT_EYE_TOP, self._RIGHT_BROW_INNER)
        au1 = float(np.clip((left_brow_eye + right_brow_eye) / (2 * ipd) * 5, 0, 1))

        # AU4 — Brow Lowerer (convergence of inner brows)
        brow_convergence = _lm_dist(landmarks, self._LEFT_BROW_INNER, self._RIGHT_BROW_INNER)
        au4 = float(np.clip(1.0 - (brow_convergence / ipd), 0, 1))

        # AU6 — Cheek Raiser (key for Duchenne smile)
        left_cheek_lift = _lm_y_diff(landmarks, self._LEFT_EYE_BOTTOM, self._LEFT_CHEEK)
        right_cheek_lift = _lm_y_diff(landmarks, self._RIGHT_EYE_BOTTOM, self._RIGHT_CHEEK)
        au6 = float(np.clip((left_cheek_lift + right_cheek_lift) / (2 * ipd) * 8, 0, 1))

        # AU12 — Lip Corner Puller (smile)
        lip_width = _lm_dist(landmarks, self._LEFT_LIP_CORNER, self._RIGHT_LIP_CORNER)
        au12 = float(np.clip(lip_width / ipd - 0.8, 0, 1))

        # AU15 — Lip Corner Depressor (frown)
        left_corner_y = landmarks[self._LEFT_LIP_CORNER].y
        right_corner_y = landmarks[self._RIGHT_LIP_CORNER].y
        center_lip_y = landmarks[self._UPPER_LIP_CENTER].y
        corner_avg = (left_corner_y + right_corner_y) / 2
        au15 = float(np.clip((corner_avg - center_lip_y) / ipd * 10, 0, 1))

        # AU17 — Chin Raiser
        chin_lip = _lm_dist(landmarks, self._CHIN, self._LOWER_LIP_CENTER)
        au17 = float(np.clip(1.0 - chin_lip / ipd * 3, 0, 1))

        # AU20 — Lip Stretcher (fear)
        lip_height = _lm_dist(landmarks, self._UPPER_LIP_CENTER, self._LOWER_LIP_CENTER)
        au20 = float(np.clip(lip_width / (lip_height + 1e-6) / 8 - 0.3, 0, 1))

        # AU23 — Lip Tightener (anger)
        au23 = float(np.clip(1.0 - lip_height / ipd * 5, 0, 1))

        # AU24 — Lip Pressor
        au24 = float(np.clip(au23 * 0.8, 0, 1))

        # AU28 — Lip Suck (anxiety)
        lower_lip_tuck = _lm_y_diff(landmarks, self._LOWER_LIP_CENTER, self._CHIN)
        au28 = float(np.clip(1.0 - lower_lip_tuck / ipd * 4, 0, 1))

        action_units = {
            "AU1": au1, "AU4": au4, "AU6": au6, "AU12": au12,
            "AU15": au15, "AU17": au17, "AU20": au20,
            "AU23": au23, "AU24": au24, "AU28": au28,
        }
        return action_units

    def detect_genuine_vs_masked_smile(self, landmarks: list[Any]) -> SmileAnalysis:
        """
        Duchenne smile (genuine) vs polite/masked smile.

        Duchenne = AU6 > 0.4 AND AU12 > 0.5 (orbicularis oculi + zygomaticus major)
        Polite   = AU12 > 0.5 but AU6 < 0.25
        """
        action_units = self.extract_action_units(landmarks)
        au6 = action_units["AU6"]
        au12 = action_units["AU12"]

        if au12 > 0.5 and au6 > 0.4:
            return SmileAnalysis(
                smile_type="genuine",
                authenticity_score=min((au6 + au12) / 2, 1.0),
                confidence=0.85,
            )
        elif au12 > 0.5 and au6 < 0.25:
            return SmileAnalysis(
                smile_type="polite",
                authenticity_score=max(0.0, au12 - au6),
                confidence=0.75,
            )
        else:
            return SmileAnalysis(
                smile_type="none",
                authenticity_score=0.0,
                confidence=0.9,
            )

    def detect_micro_expressions(self, frame_buffer: deque[dict[str, float]], window_ms: float = 200.0) -> list[MicroExpression]:
        """
        Detect AUs that spike and vanish within 3-5 frames (~200ms at 10fps).
        """
        if len(frame_buffer) < 5:
            return []

        micro_expressions: list[MicroExpression] = []
        au_names = list(frame_buffer[0].keys())
        fps = 10
        frame_ms = 1000.0 / fps

        for au_name in au_names:
            values = [frame_aus.get(au_name, 0.0) for frame_aus in frame_buffer]

            for start_idx in range(len(values) - 4):
                window = values[start_idx : start_idx + 5]
                peak_val = max(window)
                baseline = (window[0] + window[-1]) / 2

                # Spike: peak > baseline + 0.3 and returns to baseline within 5 frames
                if peak_val > baseline + 0.3 and peak_val > 0.4:
                    duration_ms = (window.index(peak_val) + 1) * frame_ms
                    if duration_ms <= window_ms:
                        emotion_name = self._au_to_emotion(au_name)
                        micro_expressions.append(MicroExpression(
                            emotion_name=emotion_name,
                            duration_ms=duration_ms,
                            peak_intensity=peak_val,
                            au_triggered=au_name,
                        ))

        return micro_expressions

    def classify_emotion_deepface(self, frame: np.ndarray) -> dict[str, float]:
        """
        Run DeepFace emotion analysis on a single frame.

        Returns 7 emotion scores: angry, disgust, fear, happy, sad, surprise, neutral.
        """
        try:
            from deepface import DeepFace

            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(results, list) and len(results) > 0:
                emotions = results[0].get("emotion", {})
                # Normalise to 0-1
                total = sum(emotions.values()) or 1.0
                return {key: val / total for key, val in emotions.items()}
            return {}
        except Exception as deepface_error:
            logger.warning(f"DeepFace analysis failed: {deepface_error}")
            return {}

    def analyze_frame(self, frame: np.ndarray) -> Optional[FaceAnalysisResult]:
        """
        Full face analysis pipeline for a single BGR frame.

        PRIVACY: frame is processed in-memory only; never written to disk.
        """
        start_time = time.perf_counter()

        self._ensure_face_mesh()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        action_units = self.extract_action_units(landmarks)
        smile = self.detect_genuine_vs_masked_smile(landmarks)

        # Update AU history for micro-expression detection
        self._au_history.append(action_units)
        micro_expressions = self.detect_micro_expressions(self._au_history)

        # DeepFace emotions
        deepface_emotions = self.classify_emotion_deepface(frame)

        # Combined: DeepFace 60% + AU-derived 40%
        au_emotions = self._aus_to_emotion_scores(action_units)
        combined: dict[str, float] = {}
        all_keys = set(list(deepface_emotions.keys()) + list(au_emotions.keys()))
        for key in all_keys:
            df_score = deepface_emotions.get(key, 0.0)
            au_score = au_emotions.get(key, 0.0)
            combined[key] = df_score * 0.6 + au_score * 0.4

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FaceAnalysisResult(
            action_units=action_units,
            smile_analysis=smile,
            micro_expressions=micro_expressions,
            deepface_emotions=deepface_emotions,
            combined_emotions=combined,
            processing_time_ms=elapsed_ms,
        )

    # ── Helpers ──

    def _au_to_emotion(self, au_name: str) -> str:
        """Map a single AU to its primary associated emotion."""
        mapping = {
            "AU1": "Sadness", "AU4": "Anger", "AU6": "Joy",
            "AU12": "Joy", "AU15": "Sadness", "AU17": "Disgust",
            "AU20": "Fear", "AU23": "Anger", "AU24": "Anger",
            "AU28": "Fear",
        }
        return mapping.get(au_name, "Surprise")

    def _aus_to_emotion_scores(self, action_units: dict[str, float]) -> dict[str, float]:
        """Derive emotion probabilities from AU activations."""
        scores: dict[str, float] = {
            "happy": (action_units.get("AU6", 0) + action_units.get("AU12", 0)) / 2,
            "sad": (action_units.get("AU1", 0) + action_units.get("AU15", 0)) / 2,
            "angry": (action_units.get("AU4", 0) + action_units.get("AU23", 0)) / 2,
            "fear": (action_units.get("AU20", 0) + action_units.get("AU28", 0)) / 2,
            "disgust": action_units.get("AU17", 0),
            "surprise": action_units.get("AU1", 0) * 0.5,
            "neutral": max(0.0, 1.0 - sum(action_units.values()) / 10),
        }
        return scores
