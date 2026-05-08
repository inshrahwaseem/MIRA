"""
MIRA Multimodal Fusion Engine.

Combines text, voice, and camera EmotionVectors via weighted Bayesian
fusion with cross-modal conflict detection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.core.emotion_taxonomy import EmotionVector, get_taxonomy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class ConflictReport:
    """Describes cross-modal emotional conflict."""

    has_conflict: bool
    conflict_type: str = ""
    masked_emotion: str = ""
    genuine_emotion: str = ""
    confidence: float = 0.0
    explanation: str = ""
    llm_adjustment_instruction: str = ""


@dataclass
class FusedEmotionResult:
    """Final output of the fusion pipeline."""

    fused_vector: EmotionVector
    primary_emotion: str
    secondary_emotion: str
    intensity_label: str  # mild | moderate | intense
    conflict_report: ConflictReport
    modalities_used: list[str]
    weights_applied: dict[str, float]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# CrossModalConflictDetector
# ---------------------------------------------------------------------------
class CrossModalConflictDetector:
    """
    Detects semantic disagreement between modalities.

    Specific clinical patterns:
      - Masked Positivity:  smiling + low-energy voice + hopeless text
      - Suppressed Anxiety: calm text + high pitch + fidgeting
      - Controlled Anger:   angry text + neutral face + tense posture
      - Emotional Blunting:  near-zero affect in all channels
    """

    CONFLICT_THRESHOLD: float = 0.55

    def detect(
        self,
        text_vector: Optional[EmotionVector],
        voice_vector: Optional[EmotionVector],
        camera_vector: Optional[EmotionVector],
        raw_context: Optional[dict[str, Any]] = None,
    ) -> ConflictReport:
        """Compare pairs of modality vectors for conflict."""
        context = raw_context or {}
        pairs: list[tuple[str, str, float]] = []

        vectors = {
            "text": text_vector,
            "voice": voice_vector,
            "camera": camera_vector,
        }

        modality_keys = [key for key, vec in vectors.items() if vec is not None]
        for i, key_a in enumerate(modality_keys):
            for key_b in modality_keys[i + 1:]:
                distance = vectors[key_a].cosine_distance(vectors[key_b])
                pairs.append((key_a, key_b, distance))

        # Any pair above threshold?
        conflicting = [(pair_a, pair_b, dist) for pair_a, pair_b, dist in pairs if dist > self.CONFLICT_THRESHOLD]
        if not conflicting:
            return ConflictReport(has_conflict=False)

        # Try to match known clinical patterns
        pattern = self._match_pattern(vectors, context)

        return ConflictReport(
            has_conflict=True,
            conflict_type=pattern.get("type", "general_conflict"),
            masked_emotion=pattern.get("masked", ""),
            genuine_emotion=pattern.get("genuine", ""),
            confidence=max(dist for _, _, dist in conflicting),
            explanation=pattern.get("explanation", "Cross-modal disagreement detected."),
            llm_adjustment_instruction=pattern.get("llm_instruction", ""),
        )

    def _match_pattern(
        self,
        vectors: dict[str, Optional[EmotionVector]],
        context: dict[str, Any],
    ) -> dict[str, str]:
        """Match known clinical conflict patterns."""
        text_vec = vectors.get("text")
        voice_vec = vectors.get("voice")
        camera_vec = vectors.get("camera")

        # Masked Positivity: smiling face + low-energy voice + hopeless text
        if (
            camera_vec is not None and voice_vec is not None and text_vec is not None
            and text_vec.to_pad()[0] < -0.3  # negative text valence
            and context.get("smile_type") == "polite"
            and context.get("voice_energy", 1.0) < 0.3
        ):
            return {
                "type": "masked_positivity",
                "masked": text_vec.dominant_emotion,
                "genuine": "Sadness",
                "explanation": (
                    "User is smiling but voice energy is low and text expresses negativity. "
                    "Likely masking genuine distress with a social smile."
                ),
                "llm_instruction": (
                    "Gently acknowledge the contrast. Do NOT say 'you seem fine'. "
                    "Use: 'I notice what you're saying feels heavy, even though you're smiling.'"
                ),
            }

        # Suppressed Anxiety: calm text + high pitch + rapid speech + fidgeting
        if (
            text_vec is not None and voice_vec is not None
            and text_vec.to_pad()[0] > 0.0  # positive/neutral text
            and voice_vec.to_pad()[1] > 0.6  # high voice arousal
            and context.get("fidget_score", 0) > 0.5
        ):
            return {
                "type": "suppressed_anxiety",
                "masked": "Apprehension",
                "genuine": "Anxiety",
                "explanation": (
                    "Text reads calm but voice pitch/pace and fidgeting suggest underlying anxiety."
                ),
                "llm_instruction": (
                    "Address possible anxiety without labelling. "
                    "Use: 'Your body seems a bit restless — how are you really feeling right now?'"
                ),
            }

        # Controlled Anger: angry text + neutral face + tense posture
        if (
            text_vec is not None and camera_vec is not None
            and text_vec._vector[9] > 0.5  # anger dim
            and context.get("face_neutral", False)
            and context.get("posture_tension", 0) > 0.5
        ):
            return {
                "type": "controlled_anger",
                "masked": "Anger",
                "genuine": "Anger",
                "explanation": "Text expresses anger but face is neutral — controlled anger.",
                "llm_instruction": "Validate frustration. Offer anger-management reframing.",
            }

        # Emotional Blunting: near-zero everything
        all_vecs = [v for v in vectors.values() if v is not None]
        if all_vecs and all(v.intensity_score < 0.15 for v in all_vecs):
            return {
                "type": "emotional_blunting",
                "masked": "Numb",
                "genuine": "Numb",
                "explanation": "Very low emotional signal across all modalities — possible emotional blunting.",
                "llm_instruction": (
                    "This is clinically significant. Gently ask about numbness: "
                    "'Sometimes feeling nothing can be harder than feeling something. Has it been like this for a while?'"
                ),
            }

        return {"type": "general_conflict", "explanation": "Modalities show conflicting emotional signals."}


# ---------------------------------------------------------------------------
# BayesianFusion
# ---------------------------------------------------------------------------
class BayesianFusion:
    """Weighted Bayesian fusion of modality EmotionVectors."""

    def fuse(
        self,
        vectors: dict[str, EmotionVector],
        weights: dict[str, float],
        prior: Optional[np.ndarray] = None,
    ) -> EmotionVector:
        """
        Combine modality vectors with confidence-weighted Bayesian update.
        `prior` is the user's 30-day baseline as a 64-dim array (optional).
        """
        if not vectors:
            return EmotionVector()

        # Start from prior or uniform
        posterior = prior.copy() if prior is not None else np.ones(64, dtype=np.float32) / 64.0

        for modality_name, emotion_vector in vectors.items():
            modality_weight = weights.get(modality_name, 0.33)
            likelihood = np.abs(emotion_vector._vector) + 1e-8  # avoid zeros
            posterior = posterior * (likelihood ** modality_weight)

        # Normalize
        total = posterior.sum()
        if total > 0:
            posterior = posterior / total

        return EmotionVector(posterior)


# ---------------------------------------------------------------------------
# MultimodalFusionEngine — top-level orchestrator
# ---------------------------------------------------------------------------
class MultimodalFusionEngine:
    """Orchestrates text/voice/camera analysis and fuses results."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "text": 0.45,
        "voice": 0.35,
        "camera": 0.20,
    }

    def __init__(self) -> None:
        self._bayesian = BayesianFusion()
        self._conflict_detector = CrossModalConflictDetector()
        self._taxonomy = get_taxonomy()

    def _rebalance_weights(self, available: set[str]) -> dict[str, float]:
        """Proportionally redistribute weights when a modality is missing."""
        raw = {key: weight for key, weight in self.DEFAULT_WEIGHTS.items() if key in available}
        total = sum(raw.values())
        if total == 0:
            return {key: 1.0 / len(raw) for key in raw}
        return {key: weight / total for key, weight in raw.items()}

    async def analyze_session(
        self,
        text: str,
        audio_features: Optional[dict[str, Any]] = None,
        camera_features: Optional[dict[str, Any]] = None,
        session_id: str = "",
        user_id: str = "",
        user_prior: Optional[np.ndarray] = None,
    ) -> FusedEmotionResult:
        """
        Full multimodal analysis — runs text/voice/camera in PARALLEL
        then fuses with Bayesian update.
        """
        start_time = time.perf_counter()

        # Parallel analysis
        text_task = asyncio.to_thread(self._analyze_text, text)
        voice_task = asyncio.to_thread(self._analyze_voice, audio_features)
        camera_task = asyncio.to_thread(self._analyze_camera, camera_features)

        text_result, voice_result, camera_result = await asyncio.gather(
            text_task, voice_task, camera_task
        )

        # Collect available vectors
        vectors: dict[str, EmotionVector] = {}
        modalities_used: list[str] = []
        raw_context: dict[str, Any] = {}

        if text_result is not None:
            vectors["text"] = text_result.emotion_vector
            modalities_used.append("text")

        if voice_result is not None:
            vectors["voice"] = voice_result
            modalities_used.append("voice")
            if audio_features:
                raw_context["voice_energy"] = audio_features.get("energy", 0.5)

        if camera_result is not None:
            vectors["camera"] = camera_result
            modalities_used.append("camera")
            if camera_features:
                raw_context["smile_type"] = camera_features.get("smile_type", "none")
                raw_context["fidget_score"] = camera_features.get("fidget_score", 0.0)
                raw_context["face_neutral"] = camera_features.get("face_neutral", False)
                raw_context["posture_tension"] = camera_features.get("posture_tension", 0.0)

        # Rebalance weights
        weights = self._rebalance_weights(set(modalities_used))

        # Conflict detection
        conflict_report = self._conflict_detector.detect(
            vectors.get("text"), vectors.get("voice"), vectors.get("camera"), raw_context,
        )

        # Bayesian fusion
        fused_vector = self._bayesian.fuse(vectors, weights, prior=user_prior)

        # Determine primary / secondary emotions
        primary = fused_vector.dominant_emotion
        vector_array = fused_vector._vector[3:11]
        sorted_indices = np.argsort(vector_array)[::-1]
        primary_names = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
        secondary = primary_names[sorted_indices[1]] if len(sorted_indices) > 1 else ""

        # Intensity label
        intensity = fused_vector.intensity_score
        if intensity < 0.33:
            intensity_label = "mild"
        elif intensity < 0.66:
            intensity_label = "moderate"
        else:
            intensity_label = "intense"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FusedEmotionResult(
            fused_vector=fused_vector,
            primary_emotion=primary,
            secondary_emotion=secondary,
            intensity_label=intensity_label,
            conflict_report=conflict_report,
            modalities_used=modalities_used,
            weights_applied=weights,
            processing_time_ms=elapsed_ms,
        )

    # -- Internal modality handlers --
    def _analyze_text(self, text: str):
        """Synchronous text analysis (runs in thread)."""
        from app.text.analyzer import TextEmotionAnalyzer
        analyzer = TextEmotionAnalyzer()
        return analyzer.analyze(text)

    def _analyze_voice(self, audio_features: Optional[dict[str, Any]]) -> Optional[EmotionVector]:
        """Placeholder — returns voice EmotionVector from pre-extracted features."""
        if audio_features is None:
            return None
        vector = np.zeros(64, dtype=np.float32)
        vector[0] = audio_features.get("valence", 0.0)
        vector[1] = audio_features.get("arousal", 0.0)
        vector[2] = audio_features.get("dominance", 0.5)
        return EmotionVector(vector)

    def _analyze_camera(self, camera_features: Optional[dict[str, Any]]) -> Optional[EmotionVector]:
        """Placeholder — returns camera EmotionVector from pre-extracted features."""
        if camera_features is None:
            return None
        vector = np.zeros(64, dtype=np.float32)
        vector[0] = camera_features.get("valence", 0.0)
        vector[1] = camera_features.get("arousal", 0.0)
        vector[2] = camera_features.get("dominance", 0.5)
        return EmotionVector(vector)
