"""
MIRA Crisis Detection & Response Protocol.

3 independent detection layers (ALL run on every message):
  Layer 1: Keyword matching (<5ms)
  Layer 2: ML composite scoring (valence, arousal, outlier, future refs)
  Layer 3: Behavioral (voice + camera signals, if enabled)

Response protocol: 4 levels (0=none → 3=immediate override).
Level 3 uses hardcoded Urdu safety templates — NEVER LLM-generated.

DESIGN PRINCIPLE: False positive >> false negative (always err toward caution).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGH_RISK_WORDS: list[str] = [
    "suicide", "kill myself", "end it all", "can't go on",
    "no reason to live", "goodbye forever", "want to die",
    "not be here anymore", "make it stop", "no way out",
]

MEDIUM_RISK_WORDS: list[str] = [
    "hopeless", "worthless", "burden to everyone",
    "disappear", "nobody would care", "give up on everything",
    "what's the point of anything", "better off without me",
    "can't take it anymore", "nothing matters",
]

# Context words that negate crisis (gaming, casual speech)
CONTEXT_NEGATORS: dict[str, list[str]] = {
    "kill": ["game", "level", "boss", "test", "exam", "time", "vibe", "it at"],
    "end": ["movie", "show", "season", "episode", "story", "book", "semester"],
    "die": ["laughing", "funny", "hilarious", "bored", "dying to try"],
}


class CrisisLevel(IntEnum):
    NONE = 0
    CONCERN = 1
    ELEVATED = 2
    IMMEDIATE = 3


@dataclass
class CrisisAssessment:
    """Complete crisis assessment output."""
    level: CrisisLevel
    triggered_words: list[str]
    keyword_score: float
    ml_score: float
    behavioral_score: float
    composite_score: float
    flags: list[str]
    response_instruction: str


# ---------------------------------------------------------------------------
# Layer 1: Keyword Matching (<5ms)
# ---------------------------------------------------------------------------
class KeywordCrisisDetector:
    """Fast keyword-based crisis screening."""

    def detect(self, text: str) -> tuple[list[str], CrisisLevel, float]:
        text_lower = text.lower()
        triggered: list[str] = []

        # Check high-risk
        for word in HIGH_RISK_WORDS:
            if word in text_lower and not self._is_negated(word, text_lower):
                triggered.append(f"HIGH:{word}")

        # Check medium-risk
        for word in MEDIUM_RISK_WORDS:
            if word in text_lower:
                triggered.append(f"MED:{word}")

        high_count = sum(1 for t in triggered if t.startswith("HIGH:"))
        med_count = sum(1 for t in triggered if t.startswith("MED:"))

        if high_count >= 1:
            return triggered, CrisisLevel.IMMEDIATE, 1.0
        elif med_count >= 2:
            return triggered, CrisisLevel.ELEVATED, 0.7
        elif med_count >= 1:
            return triggered, CrisisLevel.CONCERN, 0.4
        return triggered, CrisisLevel.NONE, 0.0

    def _is_negated(self, word: str, text: str) -> bool:
        """Check if crisis word is used in non-crisis context."""
        negators = CONTEXT_NEGATORS.get(word, [])
        for negator in negators:
            # Check if negator appears within 5 words of the keyword
            pattern = rf"{re.escape(word)}.{{0,30}}{re.escape(negator)}"
            if re.search(pattern, text):
                return True
            pattern_rev = rf"{re.escape(negator)}.{{0,30}}{re.escape(word)}"
            if re.search(pattern_rev, text):
                return True
        return False


# ---------------------------------------------------------------------------
# Layer 2: ML Composite Scoring
# ---------------------------------------------------------------------------
class MLCrisisScorer:
    """Weighted composite scoring from emotion and behavioral signals."""

    def score(
        self,
        valence: float = 0.0,
        arousal: float = 0.5,
        arousal_delta: float = 0.0,
        is_dbscan_outlier: bool = False,
        keyword_count: int = 0,
        has_future_references: bool = True,
    ) -> tuple[CrisisLevel, float]:
        components: list[tuple[float, float]] = [
            (1.0 if valence < -0.75 else 0.0, 0.30),
            (1.0 if arousal_delta < -0.35 else 0.0, 0.20),  # paradoxical calm
            (1.0 if is_dbscan_outlier else 0.0, 0.20),
            (1.0 if keyword_count > 0 else 0.0, 0.15),
            (1.0 if not has_future_references else 0.0, 0.15),
        ]

        score = sum(val * weight for val, weight in components)

        if score >= 0.75:
            return CrisisLevel.IMMEDIATE, score
        elif score >= 0.45:
            return CrisisLevel.ELEVATED, score
        elif score >= 0.20:
            return CrisisLevel.CONCERN, score
        return CrisisLevel.NONE, score


# ---------------------------------------------------------------------------
# Layer 3: Behavioral (voice + camera)
# ---------------------------------------------------------------------------
class BehavioralCrisisDetector:
    """Crisis signals from voice and camera modalities."""

    def detect(
        self,
        voice_features: Optional[dict[str, Any]] = None,
        camera_features: Optional[dict[str, Any]] = None,
        previous_crisis_level: int = 0,
    ) -> tuple[int, list[str]]:
        """Returns (additional_crisis_points, flags)."""
        additional = 0
        flags: list[str] = []

        if voice_features:
            f0_var = voice_features.get("pitch", {}).get("std_f0", 999)
            rms = voice_features.get("energy", {}).get("rms_mean", 999)
            speech_rate = voice_features.get("temporal", {}).get("speech_rate_estimate", 999)

            # Flat affect: extremely low variation in all voice parameters
            if f0_var < 10 and rms < 0.01 and speech_rate < 1.5:
                additional += 1
                flags.append("VOICE_FLAT_AFFECT: minimal pitch/energy/rate variation")

            # Paradoxical calm after previous crisis
            if (
                previous_crisis_level >= 2
                and rms > 0.08
                and f0_var < 15
            ):
                additional += 1
                flags.append("PARADOXICAL_CALM: sudden energy increase after crisis — may indicate decision")

        if camera_features:
            aus = camera_features.get("action_units", {})
            if aus:
                max_au = max(aus.values(), default=0.5)
                if max_au < 0.15:
                    additional += 1
                    flags.append("CAMERA_BLUNTING: all AU scores < 0.15 — emotional blunting")

        return min(additional, 1), flags  # cap at +1


# ---------------------------------------------------------------------------
# Crisis Response Protocol
# ---------------------------------------------------------------------------
class CrisisResponseProtocol:
    """
    Hardcoded crisis response templates.

    TONE RULES (absolute, non-negotiable):
      NEVER: clinical language, alarming phrases, bullet-point resource lists
      ALWAYS: warm, slow, present, one question at a time
    """

    _RESPONSES: dict[int, list[str]] = {
        0: [],  # no special action
        1: [
            "Main yeh feel kar rahi hoon ke aap thoda mushkil waqt se guzar rahe hain. "
            "Kya aap mujhe thoda aur bata sakte hain?",
        ],
        2: [
            "Aap jo feel kar rahe hain woh valid hai. Main yahaan hoon aur kahin nahi ja rahi. "
            "Kya hum ek grounding exercise try karein?",
            "Aao ek cheez try karein — 5 cheezein jo aap abhi DEKH sakte hain bataiye.",
            "Kabhi kabhi aise moments mein kisi se baat karna madad karta hai. "
            "Umang Helpline (0311-7786264) available hai — bilkul free aur confidential.",
        ],
        3: [
            "Aap abhi safe hain. Main yahaan hoon.",
            "Kya aap mujhe thoda aur bata sakte hain ke ab kaisa feel ho raha hai?",
            "Kabhi kabhi aise moments mein kisi se baat karna help karta hai — "
            "Pakistan mein Umang helpline (0311-7786264) available hai, "
            "aur aap unhe abhi call kar sakte hain. Main bhi yahaan hoon.",
        ],
    }

    @classmethod
    def get_response(cls, level: CrisisLevel, exchange_count: int = 0) -> Optional[str]:
        """Get appropriate crisis response for current level and exchange count."""
        responses = cls._RESPONSES.get(level.value, [])
        if not responses:
            return None
        idx = min(exchange_count, len(responses) - 1)
        return responses[idx]

    @classmethod
    def get_instruction(cls, level: CrisisLevel) -> str:
        """Get LLM instruction adjustment for the crisis level."""
        instructions = {
            CrisisLevel.NONE: "Normal therapeutic response.",
            CrisisLevel.CONCERN: (
                "Embed a warm gentle check-in naturally in response. "
                "Do NOT use clinical language. Ask one caring question."
            ),
            CrisisLevel.ELEVATED: (
                "Offer grounding exercise. After 2 exchanges, naturally mention resources. "
                "No bullet-point lists. Resources should feel like a caring suggestion."
            ),
            CrisisLevel.IMMEDIATE: (
                "OVERRIDE LLM — use hardcoded safety response template. "
                "Log to crisis_events table. Trigger n8n followup workflow."
            ),
        }
        return instructions[level]


# ---------------------------------------------------------------------------
# CrisisDetectionSystem — runs all 3 layers
# ---------------------------------------------------------------------------
class CrisisDetectionSystem:
    """
    3-layer crisis detection pipeline.

    All layers run on EVERY message. Results are combined with
    max-of-layers logic (highest crisis level wins).
    """

    def __init__(self) -> None:
        self._keyword_detector = KeywordCrisisDetector()
        self._ml_scorer = MLCrisisScorer()
        self._behavioral_detector = BehavioralCrisisDetector()

    def assess(
        self,
        text: str,
        valence: float = 0.0,
        arousal: float = 0.5,
        arousal_delta: float = 0.0,
        is_dbscan_outlier: bool = False,
        has_future_references: bool = True,
        voice_features: Optional[dict[str, Any]] = None,
        camera_features: Optional[dict[str, Any]] = None,
        previous_crisis_level: int = 0,
    ) -> CrisisAssessment:
        """Run all 3 detection layers and return composite assessment."""
        # Layer 1: Keywords
        triggered, kw_level, kw_score = self._keyword_detector.detect(text)

        # Layer 2: ML composite
        ml_level, ml_score = self._ml_scorer.score(
            valence=valence,
            arousal=arousal,
            arousal_delta=arousal_delta,
            is_dbscan_outlier=is_dbscan_outlier,
            keyword_count=len(triggered),
            has_future_references=has_future_references,
        )

        # Layer 3: Behavioral
        behav_add, behav_flags = self._behavioral_detector.detect(
            voice_features=voice_features,
            camera_features=camera_features,
            previous_crisis_level=previous_crisis_level,
        )
        behav_score = behav_add * 0.5

        # Composite: max of layers + behavioral addon
        base_level = max(kw_level, ml_level)
        final_level_int = min(base_level.value + behav_add, 3)
        final_level = CrisisLevel(final_level_int)

        composite = max(kw_score, ml_score) + behav_score
        all_flags = [f"keyword:{t}" for t in triggered] + behav_flags

        response_instruction = CrisisResponseProtocol.get_instruction(final_level)

        if final_level >= CrisisLevel.ELEVATED:
            logger.warning(
                f"CRISIS DETECTED level={final_level.value} "
                f"composite={composite:.2f} flags={all_flags}"
            )

        return CrisisAssessment(
            level=final_level,
            triggered_words=triggered,
            keyword_score=kw_score,
            ml_score=ml_score,
            behavioral_score=behav_score,
            composite_score=min(composite, 1.0),
            flags=all_flags,
            response_instruction=response_instruction,
        )
