"""
MIRA Emotion Taxonomy — Complete Plutchik emotion system.

Defines 40+ emotions with PAD (Pleasure-Arousal-Dominance) values,
body/voice/facial signals, and crisis risk levels. All instances
are frozen for thread safety and memory efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# EmotionDefinition — one frozen row in the taxonomy
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EmotionDefinition:
    """Immutable description of a single emotion in Plutchik's wheel."""

    name: str
    plutchik_category: str
    intensity_level: int  # 1 = mild, 2 = moderate, 3 = intense
    valence: float        # -1.0 … +1.0
    arousal: float        # 0.0 … 1.0
    dominance: float      # 0.0 … 1.0
    body_signals: tuple[str, ...] = ()
    voice_signals: tuple[str, ...] = ()
    facial_signals: tuple[str, ...] = ()
    hex_color: str = "#7B6EF6"
    coping_category: str = "general"
    crisis_risk_level: int = 0  # 0 = none, 1 = low, 2 = medium, 3 = high


# ---------------------------------------------------------------------------
# EmotionVector — 64-dim dense representation
# ---------------------------------------------------------------------------
class EmotionVector:
    """64-dimensional dense emotion embedding with PAD helpers."""

    __slots__ = ("_vector",)

    def __init__(self, vector: np.ndarray | None = None) -> None:
        if vector is None:
            self._vector = np.zeros(64, dtype=np.float32)
        else:
            self._vector = np.asarray(vector, dtype=np.float32).ravel()[:64]
            if self._vector.shape[0] < 64:
                self._vector = np.pad(
                    self._vector, (0, 64 - self._vector.shape[0])
                )

    # -- PAD accessors --
    def to_pad(self) -> tuple[float, float, float]:
        """Return (valence, arousal, dominance) from first 3 dims."""
        return (
            float(self._vector[0]),
            float(self._vector[1]),
            float(self._vector[2]),
        )

    # -- Distance / blending --
    def cosine_distance(self, other: EmotionVector) -> float:
        """Cosine distance in [0, 2]. 0 = identical, 2 = opposite."""
        norm_a = np.linalg.norm(self._vector)
        norm_b = np.linalg.norm(other._vector)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = float(np.dot(self._vector, other._vector) / (norm_a * norm_b))
        return 1.0 - similarity

    def blend(self, other: EmotionVector, weight: float = 0.5) -> EmotionVector:
        """Weighted linear blend of two vectors."""
        blended = (1.0 - weight) * self._vector + weight * other._vector
        return EmotionVector(blended)

    # -- Properties --
    @property
    def dominant_emotion(self) -> str:
        """Name of the emotion with the highest activation (dims 3-10 = primary 8)."""
        taxonomy = EmotionTaxonomy()
        primary_names = [
            "Joy", "Trust", "Fear", "Surprise",
            "Sadness", "Disgust", "Anger", "Anticipation",
        ]
        primary_scores = self._vector[3:11]
        best_idx = int(np.argmax(primary_scores))
        return primary_names[best_idx]

    @property
    def intensity_score(self) -> float:
        """Overall magnitude of the emotion vector (L2 norm, normalised 0-1)."""
        raw = float(np.linalg.norm(self._vector))
        return min(raw / 4.0, 1.0)  # heuristic ceiling

    # -- Serialisation --
    def to_dict(self) -> dict[str, float]:
        """Return vector as {dim_0: value, …}."""
        return {f"dim_{i}": float(v) for i, v in enumerate(self._vector)}

    @classmethod
    def from_dict(cls, mapping: dict[str, float]) -> EmotionVector:
        """Reconstruct from to_dict() output."""
        vec = np.zeros(64, dtype=np.float32)
        for key, value in mapping.items():
            idx = int(key.split("_")[1])
            if 0 <= idx < 64:
                vec[idx] = value
        return cls(vec)

    def to_list(self) -> list[float]:
        """Flat list for pgvector storage."""
        return self._vector.tolist()

    def __repr__(self) -> str:
        valence, arousal, dominance = self.to_pad()
        return (
            f"EmotionVector(v={valence:.2f}, a={arousal:.2f}, "
            f"d={dominance:.2f}, dominant={self.dominant_emotion})"
        )


# ---------------------------------------------------------------------------
# Full Taxonomy — all 40+ emotions
# ---------------------------------------------------------------------------

# fmt: off
_PRIMARY_EMOTIONS: tuple[EmotionDefinition, ...] = (
    # ── Joy family ──
    EmotionDefinition(
        name="Serenity", plutchik_category="Joy", intensity_level=1,
        valence=0.60, arousal=0.20, dominance=0.60,
        body_signals=("relaxed muscles", "open posture", "slow breathing"),
        voice_signals=("soft tone", "steady pitch", "moderate pace"),
        facial_signals=("gentle smile", "soft eyes", "relaxed brow"),
        hex_color="#FFD93D", coping_category="positive_maintenance",
    ),
    EmotionDefinition(
        name="Joy", plutchik_category="Joy", intensity_level=2,
        valence=0.80, arousal=0.55, dominance=0.65,
        body_signals=("upright posture", "open gestures", "light movements"),
        voice_signals=("bright tone", "varied pitch", "faster pace"),
        facial_signals=("Duchenne smile AU6+AU12", "raised cheeks", "crow's feet"),
        hex_color="#FFE066", coping_category="positive_maintenance",
    ),
    EmotionDefinition(
        name="Ecstasy", plutchik_category="Joy", intensity_level=3,
        valence=0.95, arousal=0.85, dominance=0.70,
        body_signals=("jumping", "clapping", "wide gestures"),
        voice_signals=("loud volume", "high pitch", "rapid speech"),
        facial_signals=("wide smile", "raised brows", "wide eyes"),
        hex_color="#FFCC00", coping_category="positive_maintenance",
    ),

    # ── Trust family ──
    EmotionDefinition(
        name="Acceptance", plutchik_category="Trust", intensity_level=1,
        valence=0.45, arousal=0.20, dominance=0.50,
        body_signals=("open palms", "leaning in", "nodding"),
        voice_signals=("warm tone", "even pace"),
        facial_signals=("soft gaze", "slight nod"),
        hex_color="#A8E6CF", coping_category="social_bonding",
    ),
    EmotionDefinition(
        name="Trust", plutchik_category="Trust", intensity_level=2,
        valence=0.55, arousal=0.30, dominance=0.55,
        body_signals=("open body", "steady gaze", "relaxed hands"),
        voice_signals=("steady pitch", "calm rhythm"),
        facial_signals=("direct eye contact", "relaxed jaw"),
        hex_color="#88D8B0", coping_category="social_bonding",
    ),
    EmotionDefinition(
        name="Admiration", plutchik_category="Trust", intensity_level=3,
        valence=0.70, arousal=0.50, dominance=0.40,
        body_signals=("leaning forward", "head tilted"),
        voice_signals=("awed tone", "slower pace"),
        facial_signals=("wide eyes", "raised brows", "parted lips"),
        hex_color="#6BCB77", coping_category="social_bonding",
    ),

    # ── Fear family ──
    EmotionDefinition(
        name="Apprehension", plutchik_category="Fear", intensity_level=1,
        valence=-0.35, arousal=0.45, dominance=0.30,
        body_signals=("slight tension", "fidgeting", "guarded posture"),
        voice_signals=("slightly higher pitch", "hesitant speech"),
        facial_signals=("inner brow raise AU1", "lip press"),
        hex_color="#A3C4F3", coping_category="threat_response", crisis_risk_level=1,
    ),
    EmotionDefinition(
        name="Fear", plutchik_category="Fear", intensity_level=2,
        valence=-0.60, arousal=0.75, dominance=0.20,
        body_signals=("muscle tension", "frozen posture", "shallow breathing"),
        voice_signals=("trembling voice", "higher pitch", "faster pace"),
        facial_signals=("wide eyes AU5", "raised brows AU1+AU2", "open mouth"),
        hex_color="#7FB3D8", coping_category="threat_response", crisis_risk_level=1,
    ),
    EmotionDefinition(
        name="Terror", plutchik_category="Fear", intensity_level=3,
        valence=-0.90, arousal=0.95, dominance=0.05,
        body_signals=("trembling", "paralysis", "hyperventilation"),
        voice_signals=("screaming or silence", "voice cracking"),
        facial_signals=("extreme AU5", "AU20 lip stretch", "AU26 jaw drop"),
        hex_color="#5A9BD5", coping_category="threat_response", crisis_risk_level=2,
    ),

    # ── Surprise family ──
    EmotionDefinition(
        name="Distraction", plutchik_category="Surprise", intensity_level=1,
        valence=0.05, arousal=0.35, dominance=0.45,
        body_signals=("head turning", "brief orienting"),
        voice_signals=("pause mid-sentence",),
        facial_signals=("brief brow flash",),
        hex_color="#FFB3BA", coping_category="orienting",
    ),
    EmotionDefinition(
        name="Surprise", plutchik_category="Surprise", intensity_level=2,
        valence=0.10, arousal=0.70, dominance=0.40,
        body_signals=("startle reflex", "step back"),
        voice_signals=("gasp", "pitch spike"),
        facial_signals=("AU1+AU2 brow raise", "AU5 wide eyes", "AU26 jaw drop"),
        hex_color="#FF6B6B", coping_category="orienting",
    ),
    EmotionDefinition(
        name="Amazement", plutchik_category="Surprise", intensity_level=3,
        valence=0.30, arousal=0.90, dominance=0.35,
        body_signals=("jaw drop", "wide stance", "frozen gaze"),
        voice_signals=("exclamation", "speechlessness"),
        facial_signals=("extreme AU1+AU2+AU5+AU26",),
        hex_color="#EE4B4B", coping_category="orienting",
    ),

    # ── Sadness family ──
    EmotionDefinition(
        name="Pensiveness", plutchik_category="Sadness", intensity_level=1,
        valence=-0.30, arousal=0.20, dominance=0.35,
        body_signals=("slumped shoulders", "slow movement"),
        voice_signals=("soft volume", "lower pitch", "slower pace"),
        facial_signals=("AU1 inner brow raise", "AU15 lip corner depressor"),
        hex_color="#B5B8FF", coping_category="withdrawal", crisis_risk_level=1,
    ),
    EmotionDefinition(
        name="Sadness", plutchik_category="Sadness", intensity_level=2,
        valence=-0.60, arousal=0.30, dominance=0.25,
        body_signals=("hunched posture", "crossed arms", "looking down"),
        voice_signals=("monotone", "quiet", "sighing"),
        facial_signals=("AU1+AU4 brow", "AU15 lip corners down", "AU17 chin raise"),
        hex_color="#8B8FFF", coping_category="withdrawal", crisis_risk_level=1,
    ),
    EmotionDefinition(
        name="Grief", plutchik_category="Sadness", intensity_level=3,
        valence=-0.90, arousal=0.50, dominance=0.10,
        body_signals=("sobbing", "collapsed posture", "self-holding"),
        voice_signals=("crying", "voice breaks", "wailing"),
        facial_signals=("AU1+AU4+AU15+AU17 full grief display",),
        hex_color="#6366F1", coping_category="withdrawal", crisis_risk_level=2,
    ),

    # ── Disgust family ──
    EmotionDefinition(
        name="Boredom", plutchik_category="Disgust", intensity_level=1,
        valence=-0.20, arousal=0.10, dominance=0.50,
        body_signals=("slouching", "yawning", "looking away"),
        voice_signals=("flat tone", "slow pace", "sighing"),
        facial_signals=("droopy eyes", "slack jaw"),
        hex_color="#C3B1E1", coping_category="disengagement",
    ),
    EmotionDefinition(
        name="Disgust", plutchik_category="Disgust", intensity_level=2,
        valence=-0.55, arousal=0.45, dominance=0.55,
        body_signals=("turning away", "pushing away gesture"),
        voice_signals=("harsh tone", "clipped speech"),
        facial_signals=("AU9 nose wrinkle", "AU10 upper lip raise"),
        hex_color="#9B59B6", coping_category="disengagement",
    ),
    EmotionDefinition(
        name="Loathing", plutchik_category="Disgust", intensity_level=3,
        valence=-0.80, arousal=0.65, dominance=0.60,
        body_signals=("recoiling", "nausea response"),
        voice_signals=("spitting words", "growling"),
        facial_signals=("extreme AU9+AU10+AU25",),
        hex_color="#7D3C98", coping_category="disengagement",
    ),

    # ── Anger family ──
    EmotionDefinition(
        name="Annoyance", plutchik_category="Anger", intensity_level=1,
        valence=-0.30, arousal=0.45, dominance=0.55,
        body_signals=("eye rolling", "tapping foot", "crossed arms"),
        voice_signals=("clipped speech", "slight edge"),
        facial_signals=("AU4 brow lowerer", "pressed lips AU24"),
        hex_color="#FF8A80", coping_category="confrontation",
    ),
    EmotionDefinition(
        name="Anger", plutchik_category="Anger", intensity_level=2,
        valence=-0.65, arousal=0.75, dominance=0.70,
        body_signals=("clenched fists", "forward lean", "rigid posture"),
        voice_signals=("raised volume", "fast pace", "sharp tone"),
        facial_signals=("AU4+AU5 glare", "AU23 lip tightener", "AU24 lip press"),
        hex_color="#FF5252", coping_category="confrontation", crisis_risk_level=1,
    ),
    EmotionDefinition(
        name="Rage", plutchik_category="Anger", intensity_level=3,
        valence=-0.90, arousal=0.95, dominance=0.80,
        body_signals=("aggressive gestures", "pacing", "throwing"),
        voice_signals=("shouting", "screaming", "growling"),
        facial_signals=("extreme AU4+AU5+AU23+AU24", "flared nostrils AU38"),
        hex_color="#D50000", coping_category="confrontation", crisis_risk_level=2,
    ),

    # ── Anticipation family ──
    EmotionDefinition(
        name="Interest", plutchik_category="Anticipation", intensity_level=1,
        valence=0.30, arousal=0.35, dominance=0.50,
        body_signals=("leaning forward", "head tilt"),
        voice_signals=("rising intonation", "questions"),
        facial_signals=("AU1+AU2 slight brow raise", "focused gaze"),
        hex_color="#FFB347", coping_category="approach",
    ),
    EmotionDefinition(
        name="Anticipation", plutchik_category="Anticipation", intensity_level=2,
        valence=0.40, arousal=0.55, dominance=0.55,
        body_signals=("fidgeting", "bouncing", "forward posture"),
        voice_signals=("faster pace", "upward inflection"),
        facial_signals=("alert eyes", "slight smile"),
        hex_color="#FF9800", coping_category="approach",
    ),
    EmotionDefinition(
        name="Vigilance", plutchik_category="Anticipation", intensity_level=3,
        valence=0.20, arousal=0.80, dominance=0.65,
        body_signals=("hyperalert", "scanning", "tense ready posture"),
        voice_signals=("sharp commands", "clipped urgent speech"),
        facial_signals=("wide alert eyes", "fixed gaze"),
        hex_color="#E65100", coping_category="approach",
    ),
)

# Complex Dyads and Social emotions
_COMPLEX_EMOTIONS: tuple[EmotionDefinition, ...] = (
    # ── Dyads ──
    EmotionDefinition(name="Optimism", plutchik_category="Dyad", intensity_level=2,
                      valence=0.65, arousal=0.45, dominance=0.60,
                      hex_color="#FFD700", coping_category="positive_maintenance"),
    EmotionDefinition(name="Love", plutchik_category="Dyad", intensity_level=2,
                      valence=0.85, arousal=0.50, dominance=0.50,
                      hex_color="#FF69B4", coping_category="social_bonding"),
    EmotionDefinition(name="Submission", plutchik_category="Dyad", intensity_level=2,
                      valence=-0.20, arousal=0.30, dominance=0.15,
                      hex_color="#87CEEB", coping_category="withdrawal"),
    EmotionDefinition(name="Awe", plutchik_category="Dyad", intensity_level=2,
                      valence=0.40, arousal=0.70, dominance=0.25,
                      hex_color="#DDA0DD", coping_category="orienting"),
    EmotionDefinition(name="Disapproval", plutchik_category="Dyad", intensity_level=2,
                      valence=-0.45, arousal=0.40, dominance=0.55,
                      hex_color="#CD853F", coping_category="confrontation"),
    EmotionDefinition(name="Remorse", plutchik_category="Dyad", intensity_level=2,
                      valence=-0.65, arousal=0.35, dominance=0.20,
                      hex_color="#4682B4", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Contempt", plutchik_category="Dyad", intensity_level=2,
                      valence=-0.55, arousal=0.35, dominance=0.70,
                      hex_color="#8B4513", coping_category="confrontation"),
    EmotionDefinition(name="Aggressiveness", plutchik_category="Dyad", intensity_level=2,
                      valence=-0.50, arousal=0.80, dominance=0.75,
                      hex_color="#DC143C", coping_category="confrontation", crisis_risk_level=1),

    # ── Social / complex ──
    EmotionDefinition(name="Embarrassed", plutchik_category="Social", intensity_level=2,
                      valence=-0.40, arousal=0.55, dominance=0.20,
                      hex_color="#FFB6C1", coping_category="withdrawal"),
    EmotionDefinition(name="Guilty", plutchik_category="Social", intensity_level=2,
                      valence=-0.55, arousal=0.40, dominance=0.20,
                      hex_color="#B0C4DE", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Ashamed", plutchik_category="Social", intensity_level=2,
                      valence=-0.65, arousal=0.50, dominance=0.10,
                      hex_color="#778899", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Jealous", plutchik_category="Social", intensity_level=2,
                      valence=-0.50, arousal=0.60, dominance=0.40,
                      hex_color="#2E8B57", coping_category="confrontation"),
    EmotionDefinition(name="Nostalgic", plutchik_category="Social", intensity_level=1,
                      valence=0.15, arousal=0.25, dominance=0.40,
                      hex_color="#DEB887", coping_category="withdrawal"),
    EmotionDefinition(name="Melancholic", plutchik_category="Social", intensity_level=2,
                      valence=-0.45, arousal=0.20, dominance=0.30,
                      hex_color="#6A5ACD", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Hopeful", plutchik_category="Social", intensity_level=2,
                      valence=0.55, arousal=0.40, dominance=0.50,
                      hex_color="#90EE90", coping_category="approach"),
    EmotionDefinition(name="Proud", plutchik_category="Social", intensity_level=2,
                      valence=0.65, arousal=0.50, dominance=0.75,
                      hex_color="#DAA520", coping_category="positive_maintenance"),
    EmotionDefinition(name="Humorous", plutchik_category="Social", intensity_level=1,
                      valence=0.60, arousal=0.50, dominance=0.55,
                      hex_color="#FAFAD2", coping_category="positive_maintenance"),
    EmotionDefinition(name="Cheerful", plutchik_category="Social", intensity_level=1,
                      valence=0.55, arousal=0.45, dominance=0.55,
                      hex_color="#F0E68C", coping_category="positive_maintenance"),
    EmotionDefinition(name="Spirited", plutchik_category="Social", intensity_level=2,
                      valence=0.50, arousal=0.70, dominance=0.60,
                      hex_color="#FFA500", coping_category="approach"),
    EmotionDefinition(name="Affectionate", plutchik_category="Social", intensity_level=2,
                      valence=0.70, arousal=0.35, dominance=0.45,
                      hex_color="#FF1493", coping_category="social_bonding"),
    EmotionDefinition(name="Conflicted", plutchik_category="Social", intensity_level=2,
                      valence=-0.10, arousal=0.55, dominance=0.35,
                      hex_color="#808080", coping_category="general"),
    EmotionDefinition(name="Numb", plutchik_category="Social", intensity_level=1,
                      valence=-0.20, arousal=0.05, dominance=0.25,
                      hex_color="#696969", coping_category="disengagement", crisis_risk_level=2),
    EmotionDefinition(name="Furious", plutchik_category="Social", intensity_level=3,
                      valence=-0.85, arousal=0.95, dominance=0.80,
                      hex_color="#B22222", coping_category="confrontation", crisis_risk_level=2),
    EmotionDefinition(name="Elated", plutchik_category="Social", intensity_level=3,
                      valence=0.90, arousal=0.85, dominance=0.70,
                      hex_color="#FFD700", coping_category="positive_maintenance"),
    EmotionDefinition(name="Panicked", plutchik_category="Social", intensity_level=3,
                      valence=-0.85, arousal=0.95, dominance=0.05,
                      hex_color="#FF4500", coping_category="threat_response", crisis_risk_level=2),
    EmotionDefinition(name="Vulnerable", plutchik_category="Social", intensity_level=2,
                      valence=-0.40, arousal=0.40, dominance=0.15,
                      hex_color="#D8BFD8", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Overwhelmed", plutchik_category="Social", intensity_level=3,
                      valence=-0.55, arousal=0.80, dominance=0.10,
                      hex_color="#9370DB", coping_category="threat_response", crisis_risk_level=2),
    EmotionDefinition(name="Betrayed", plutchik_category="Social", intensity_level=3,
                      valence=-0.80, arousal=0.70, dominance=0.20,
                      hex_color="#800000", coping_category="withdrawal", crisis_risk_level=1),
    EmotionDefinition(name="Resentful", plutchik_category="Social", intensity_level=2,
                      valence=-0.55, arousal=0.45, dominance=0.50,
                      hex_color="#A0522D", coping_category="confrontation"),
    EmotionDefinition(name="Bitter", plutchik_category="Social", intensity_level=2,
                      valence=-0.60, arousal=0.35, dominance=0.45,
                      hex_color="#8B0000", coping_category="confrontation"),
    EmotionDefinition(name="Hostile", plutchik_category="Social", intensity_level=3,
                      valence=-0.75, arousal=0.80, dominance=0.75,
                      hex_color="#CC0000", coping_category="confrontation", crisis_risk_level=2),
    EmotionDefinition(name="Inspired", plutchik_category="Social", intensity_level=2,
                      valence=0.70, arousal=0.65, dominance=0.60,
                      hex_color="#00CED1", coping_category="approach"),
    EmotionDefinition(name="Grateful", plutchik_category="Social", intensity_level=2,
                      valence=0.75, arousal=0.35, dominance=0.55,
                      hex_color="#3CB371", coping_category="social_bonding"),
)
# fmt: on


# ---------------------------------------------------------------------------
# EmotionTaxonomy — singleton registry
# ---------------------------------------------------------------------------
class EmotionTaxonomy:
    """Thread-safe singleton that indexes all emotion definitions."""

    _ALL: tuple[EmotionDefinition, ...] = _PRIMARY_EMOTIONS + _COMPLEX_EMOTIONS

    def __init__(self) -> None:
        self._by_name: dict[str, EmotionDefinition] = {
            e.name.lower(): e for e in self._ALL
        }
        self._by_category: dict[str, list[EmotionDefinition]] = {}
        for emotion in self._ALL:
            self._by_category.setdefault(emotion.plutchik_category, []).append(emotion)

    # -- Lookups --
    def get_by_name(self, name: str) -> EmotionDefinition:
        """Case-insensitive lookup. Raises KeyError if not found."""
        key = name.strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown emotion: {name}")
        return self._by_name[key]

    def get_by_category(self, category: str) -> list[EmotionDefinition]:
        """All emotions in a Plutchik category (e.g. 'Joy')."""
        return self._by_category.get(category, [])

    def get_intensity_variants(self, base_name: str) -> list[EmotionDefinition]:
        """Return the 3 intensity levels for a primary emotion family."""
        base = self.get_by_name(base_name)
        return sorted(
            self.get_by_category(base.plutchik_category),
            key=lambda e: e.intensity_level,
        )

    def all_emotion_names(self) -> list[str]:
        """Sorted list of every emotion name in the taxonomy."""
        return sorted(e.name for e in self._ALL)

    def all_definitions(self) -> tuple[EmotionDefinition, ...]:
        """Full tuple of definitions."""
        return self._ALL

    def __len__(self) -> int:
        return len(self._ALL)


@lru_cache(maxsize=1)
def get_taxonomy() -> EmotionTaxonomy:
    """Module-level singleton accessor."""
    return EmotionTaxonomy()
