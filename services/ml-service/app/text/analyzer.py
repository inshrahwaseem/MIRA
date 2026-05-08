"""
MIRA Text Emotion Analyzer.

Uses HuggingFace DistilRoBERTa (j-hartmann, 28 emotions) to classify
free-form user text into Plutchik emotion vectors.  Also detects cognitive
distortions, masking language, and crisis-risk keywords.
"""

from __future__ import annotations

import html
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Local imports
from app.core.emotion_taxonomy import EmotionVector, get_taxonomy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TextAnalysisResult:
    """Output of a single text-analysis pass."""

    emotion_vector: EmotionVector
    cognitive_distortions: dict[str, bool]
    masking_probability: float
    crisis_keywords_found: list[str]
    crisis_risk_score: float
    key_phrases: list[str]
    valence: float
    arousal: float
    raw_scores: dict[str, float]
    model_confidence: float
    processing_time_ms: float


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """Cleaning, distortion detection, masking detection, and crisis signals."""

    # -- Crisis word lists --
    _HIGH_RISK: tuple[str, ...] = (
        "suicide", "kill myself", "end it all", "no reason to live",
        "goodbye forever", "want to die", "better off dead",
        "ending my life", "can't go on", "take my own life",
    )
    _MEDIUM_RISK: tuple[str, ...] = (
        "hopeless", "worthless", "burden", "disappear",
        "nobody would care", "give up", "no point",
        "can't take it anymore", "nothing matters",
    )

    # -- Masking phrases --
    _MASKING_PHRASES: tuple[str, ...] = (
        "i'm fine", "it's fine", "it's nothing", "never mind",
        "forget it", "doesn't matter", "don't worry about it",
        "i'm okay", "i'm good", "no big deal", "whatever",
        "it is what it is", "i'll be fine", "not a big deal",
    )

    # -- Distortion regex patterns --
    _CATASTROPHE_RE = re.compile(
        r"\b(always|never|everything|nothing|everyone|no\s*one|"
        r"worst|terrible|awful|horrible|disaster|ruined|destroyed)\b",
        re.IGNORECASE,
    )
    _BLACK_WHITE_RE = re.compile(
        r"\b(perfect|completely|totally|absolute|entirely|"
        r"all\s+or\s+nothing|100\s*%|0\s*%)\b",
        re.IGNORECASE,
    )
    _MIND_READING_RE = re.compile(
        r"\b(they\s+think|everyone\s+believes|she\s+must\s+hate|"
        r"he\s+probably\s+thinks|people\s+think|they\s+all\s+know)\b",
        re.IGNORECASE,
    )
    _FORTUNE_TELLING_RE = re.compile(
        r"\b(will\s+definitely|going\s+to\s+fail|i\s+know\s+it('ll|'s\s+going\s+to)|"
        r"it\s+won't\s+work|bound\s+to|guaranteed\s+to|never\s+going\s+to)\b",
        re.IGNORECASE,
    )
    _PERSONALIZATION_RE = re.compile(
        r"\b(my\s+fault|because\s+of\s+me|i\s+caused|i\s+ruined|"
        r"all\s+because\s+of\s+me|i'm\s+the\s+reason)\b",
        re.IGNORECASE,
    )
    _EMOTIONAL_REASONING_RE = re.compile(
        r"\b(i\s+feel\s+\w+\s+therefore|i\s+feel\s+like\s+a|"
        r"i\s+feel\s+so\s+\w+\s+that|because\s+i\s+feel)\b",
        re.IGNORECASE,
    )
    _SHOULD_RE = re.compile(
        r"\bi\s+(should|must|ought\s+to|have\s+to|need\s+to)\b",
        re.IGNORECASE,
    )
    _LABELING_RE = re.compile(
        r"\bi\s+am\s+a\s+(failure|loser|idiot|worthless|waste|"
        r"terrible\s+person|bad\s+person|fraud|disaster)\b",
        re.IGNORECASE,
    )
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    # ── public API ──

    def clean_text(self, text: str) -> str:
        """Normalize Unicode, strip HTML tags, collapse whitespace."""
        cleaned = html.unescape(text)
        cleaned = self._HTML_TAG_RE.sub("", cleaned)
        cleaned = unicodedata.normalize("NFKD", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def detect_cognitive_distortions(self, text: str) -> dict[str, bool]:
        """Return a dict of 8 CBT distortion types → detected True/False."""
        lower_text = text.lower()
        return {
            "catastrophizing": bool(self._CATASTROPHE_RE.search(lower_text)),
            "black_white_thinking": bool(self._BLACK_WHITE_RE.search(lower_text)),
            "mind_reading": bool(self._MIND_READING_RE.search(lower_text)),
            "fortune_telling": bool(self._FORTUNE_TELLING_RE.search(lower_text)),
            "personalization": bool(self._PERSONALIZATION_RE.search(lower_text)),
            "emotional_reasoning": bool(self._EMOTIONAL_REASONING_RE.search(lower_text)),
            "should_statements": bool(self._SHOULD_RE.search(lower_text)),
            "labeling": bool(self._LABELING_RE.search(lower_text)),
        }

    def detect_masking_language(self, text: str) -> float:
        """Score 0.0-1.0 for how much the text contains masking phrases."""
        lower_text = text.lower()
        hits = sum(1 for phrase in self._MASKING_PHRASES if phrase in lower_text)
        return min(hits / 3.0, 1.0)  # saturate at 3 hits

    def extract_crisis_signals(self, text: str) -> list[str]:
        """Return matched crisis keywords sorted HIGH first."""
        lower_text = text.lower()
        found: list[str] = []
        for phrase in self._HIGH_RISK:
            if phrase in lower_text:
                found.append(f"HIGH:{phrase}")
        for phrase in self._MEDIUM_RISK:
            if phrase in lower_text:
                found.append(f"MEDIUM:{phrase}")
        return found


# ---------------------------------------------------------------------------
# TextEmotionAnalyzer
# ---------------------------------------------------------------------------

# Mapping from j-hartmann model labels → Plutchik categories
_HARTMANN_TO_PLUTCHIK: dict[str, str] = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Joy",
    "neutral": "Serenity",
    "sadness": "Sadness",
    "surprise": "Surprise",
}

# Mapping from the bhadresh backup model
_BACKUP_TO_PLUTCHIK: dict[str, str] = {
    "anger": "Anger",
    "fear": "Fear",
    "joy": "Joy",
    "love": "Love",
    "sadness": "Sadness",
    "surprise": "Surprise",
}

# Dimension indices in the 64-dim vector for the 8 primary emotions
_EMOTION_DIM_MAP: dict[str, int] = {
    "Joy": 3, "Trust": 4, "Fear": 5, "Surprise": 6,
    "Sadness": 7, "Disgust": 8, "Anger": 9, "Anticipation": 10,
}

# Extended dims for dyads/social (11-34)
_EXTENDED_DIM_MAP: dict[str, int] = {
    "Serenity": 11, "Ecstasy": 12, "Acceptance": 13, "Admiration": 14,
    "Apprehension": 15, "Terror": 16, "Distraction": 17, "Amazement": 18,
    "Pensiveness": 19, "Grief": 20, "Boredom": 21, "Loathing": 22,
    "Annoyance": 23, "Rage": 24, "Interest": 25, "Vigilance": 26,
    "Optimism": 27, "Love": 28, "Submission": 29, "Awe": 30,
    "Disapproval": 31, "Remorse": 32, "Contempt": 33, "Aggressiveness": 34,
}


class TextEmotionAnalyzer:
    """Lazy-loaded HuggingFace pipeline for text → EmotionVector."""

    def __init__(self) -> None:
        self._preprocessor = TextPreprocessor()

    # -- Lazy model loading --
    @cached_property
    def classifier(self):
        """Primary DistilRoBERTa emotion classifier (28-label)."""
        from transformers import pipeline as hf_pipeline

        logger.info("Loading primary text model: j-hartmann/emotion-english-distilroberta-base")
        return hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,  # CPU
        )

    @cached_property
    def backup_classifier(self):
        """Fallback DistilBERT 6-label classifier."""
        from transformers import pipeline as hf_pipeline

        logger.info("Loading backup text model: bhadresh-savani/distilbert-base-uncased-emotion")
        return hf_pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
            device=-1,
        )

    # -- Public API --
    def analyze(self, text: str) -> TextAnalysisResult:
        """Full analysis of a single text input."""
        start_time = time.perf_counter()

        cleaned = self._preprocessor.clean_text(text)
        distortions = self._preprocessor.detect_cognitive_distortions(cleaned)
        masking_prob = self._preprocessor.detect_masking_language(cleaned)
        crisis_signals = self._preprocessor.extract_crisis_signals(cleaned)

        # Classify with primary model, fall back if it fails
        try:
            raw_output = self.classifier(cleaned)
            raw_scores = {item["label"]: item["score"] for item in raw_output[0]}
            label_map = _HARTMANN_TO_PLUTCHIK
        except Exception as primary_error:
            logger.warning(f"Primary model failed ({primary_error}), using backup")
            raw_output = self.backup_classifier(cleaned)
            raw_scores = {item["label"]: item["score"] for item in raw_output[0]}
            label_map = _BACKUP_TO_PLUTCHIK

        # Build emotion vector
        emotion_vector = self._map_to_plutchik(raw_scores, label_map)
        valence, arousal, _dominance = emotion_vector.to_pad()

        # Model confidence = max score among all labels
        model_confidence = max(raw_scores.values()) if raw_scores else 0.0

        # Crisis risk score
        high_count = sum(1 for signal in crisis_signals if signal.startswith("HIGH"))
        medium_count = sum(1 for signal in crisis_signals if signal.startswith("MEDIUM"))
        crisis_risk_score = min((high_count * 0.5 + medium_count * 0.2), 1.0)

        # Key phrases — simple: top distortions that fired
        key_phrases = [
            distortion_name
            for distortion_name, detected in distortions.items()
            if detected
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return TextAnalysisResult(
            emotion_vector=emotion_vector,
            cognitive_distortions=distortions,
            masking_probability=masking_prob,
            crisis_keywords_found=crisis_signals,
            crisis_risk_score=crisis_risk_score,
            key_phrases=key_phrases,
            valence=valence,
            arousal=arousal,
            raw_scores=raw_scores,
            model_confidence=model_confidence,
            processing_time_ms=elapsed_ms,
        )

    def batch_analyze(self, texts: list[str]) -> list[TextAnalysisResult]:
        """Analyze a list of texts sequentially."""
        return [self.analyze(text) for text in texts]

    # -- Internal mapping --
    def _map_to_plutchik(
        self,
        raw_scores: dict[str, float],
        label_map: dict[str, str],
    ) -> EmotionVector:
        """Map model output labels to a 64-dim Plutchik EmotionVector."""
        vector = np.zeros(64, dtype=np.float32)

        taxonomy = get_taxonomy()

        for label, score in raw_scores.items():
            plutchik_name = label_map.get(label)
            if plutchik_name is None:
                continue

            # Set PAD values (dims 0-2) weighted by score
            try:
                definition = taxonomy.get_by_name(plutchik_name)
                vector[0] += definition.valence * score
                vector[1] += definition.arousal * score
                vector[2] += definition.dominance * score
            except KeyError:
                pass

            # Primary 8 category dims
            if plutchik_name in _EMOTION_DIM_MAP:
                vector[_EMOTION_DIM_MAP[plutchik_name]] = score

            # Extended dims
            if plutchik_name in _EXTENDED_DIM_MAP:
                vector[_EXTENDED_DIM_MAP[plutchik_name]] = score

        return EmotionVector(vector)
