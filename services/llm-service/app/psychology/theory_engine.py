"""
MIRA Psychology Theory Engine.

Evidence-based therapeutic framework implementations:
  - MaslowNeedsAnalyzer: 5-level hierarchy detection + interventions
  - AttachmentStyleDetector: 4-style inference + communication adaptation
  - CBTInterventionEngine: NATs, thought records, Socratic questioning, behavioral activation
  - ACTTechniqueEngine: defusion, values clarification, RAIN acceptance
  - BigFiveProfiler: OCEAN inference from text + response style adaptation
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------
class MaslowLevel(IntEnum):
    PHYSIOLOGICAL = 1
    SAFETY = 2
    BELONGING = 3
    ESTEEM = 4
    SELF_ACTUALIZATION = 5


class AttachmentStyle(str, Enum):
    SECURE = "secure"
    ANXIOUS_PREOCCUPIED = "anxious_preoccupied"
    DISMISSIVE_AVOIDANT = "dismissive_avoidant"
    FEARFUL_AVOIDANT = "fearful_avoidant"


@dataclass
class NegativeAutomaticThought:
    thought_content: str
    distortion_type: str
    emotion_triggered: str
    intensity_estimate: float  # 0.0-1.0


@dataclass
class ThoughtRecord:
    situation: str
    emotion: str
    emotion_intensity: float
    nat: str
    evidence_for: list[str]
    evidence_against: list[str]
    balanced_thought: str
    outcome_emotion: str
    outcome_intensity: float


@dataclass
class ScheduledActivity:
    name: str
    category: str  # "mastery" | "pleasure" | "connection"
    duration_minutes: int
    time_of_day: str
    difficulty: int


@dataclass
class CommunicationAdaptations:
    tone: str
    pacing: str
    validation_level: str
    challenge_approach: str
    suggestions: list[str]


@dataclass
class BigFiveScores:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    confidence: float = 0.0  # increases with session count


@dataclass
class ResponseAdaptations:
    complexity_level: str
    social_suggestions: bool
    validation_weight: float
    structure_level: str
    notes: list[str]


# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------
_MASLOW_KEYWORDS: dict[int, list[str]] = {
    1: ["exhausted", "tired", "can't sleep", "insomnia", "hungry", "pain", "sick",
        "headache", "dizzy", "nauseous", "no energy", "physical", "body hurts"],
    2: ["job loss", "fired", "money", "rent", "homeless", "unsafe", "threat",
        "danger", "financial", "eviction", "debt", "scared for my safety",
        "stability", "security", "afraid of losing"],
    3: ["lonely", "alone", "rejected", "breakup", "nobody cares", "isolated",
        "no friends", "miss them", "abandoned", "left out", "don't belong",
        "disconnected", "no one understands"],
    4: ["failure", "worthless", "not good enough", "ashamed", "embarrassed",
        "incompetent", "stupid", "everyone is better", "can't do anything right",
        "disappointment", "imposter", "shame", "useless"],
    5: ["meaningless", "what's the point", "no purpose", "unfulfilled",
        "empty inside", "is this all there is", "wasting my life", "potential",
        "stuck", "going nowhere", "legacy", "contribution"],
}

_ATTACHMENT_KEYWORDS: dict[str, list[str]] = {
    "anxious_preoccupied": [
        "they don't care", "what if they leave", "are you still there",
        "please don't go", "do you even like me", "abandonment",
        "reassure me", "clingy", "need constant contact", "fear of rejection",
        "overthinking their texts", "what did I do wrong",
    ],
    "dismissive_avoidant": [
        "i don't need anyone", "better alone", "emotions are weakness",
        "i'm fine on my own", "too close", "suffocating", "independence",
        "don't need help", "feelings don't matter", "waste of time",
    ],
    "fearful_avoidant": [
        "want closeness but scared", "push people away", "hot and cold",
        "love scares me", "want connection but", "afraid of getting hurt",
        "trust issues", "approach avoidance", "mixed feelings about people",
    ],
    "secure": [
        "comfortable with closeness", "trust my partner", "healthy boundaries",
        "can be alone and okay", "open communication", "support each other",
    ],
}

_BIG_FIVE_MARKERS: dict[str, list[str]] = {
    "openness": ["imagine", "what if", "creative", "curious", "new idea",
                  "explore", "metaphor", "abstract", "philosophical", "art"],
    "conscientiousness": ["plan", "schedule", "deadline", "organize", "goal",
                          "discipline", "responsible", "on time", "priority", "structure"],
    "extraversion": ["party", "friends", "social", "people", "gathering",
                     "talk to", "hang out", "energy from", "group", "outgoing"],
    "agreeableness": ["feel bad for", "help them", "compromise", "harmony",
                      "caring", "empathy", "their feelings", "others first", "kind"],
    "neuroticism": ["worry", "anxious", "what if", "always", "worst case",
                    "can't stop thinking", "ruminate", "mood swings", "overwhelmed", "panic"],
}


# ---------------------------------------------------------------------------
# MaslowNeedsAnalyzer
# ---------------------------------------------------------------------------
class MaslowNeedsAnalyzer:
    """Detect which Maslow hierarchy level is most activated."""

    def analyze_need_level(
        self,
        text: str,
        emotion_valence: float = 0.0,
        session_history: Optional[list[str]] = None,
    ) -> tuple[MaslowLevel, float]:
        text_lower = text.lower()
        scores: dict[int, float] = {level: 0.0 for level in range(1, 6)}

        # Keyword scoring
        for level, keywords in _MASLOW_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            scores[level] = hits / max(len(keywords), 1)

        # Session history boost
        if session_history:
            combined = " ".join(session_history[-5:]).lower()
            for level, keywords in _MASLOW_KEYWORDS.items():
                history_hits = sum(1 for kw in keywords if kw in combined)
                scores[level] += (history_hits / max(len(keywords), 1)) * 0.3

        # Emotion vector influence
        if emotion_valence < -0.6:
            scores[1] += 0.1
            scores[2] += 0.15

        # Lower needs take priority (Maslow's principle)
        for level in range(1, 6):
            if scores[level] > 0.2:
                confidence = min(scores[level], 1.0)
                return MaslowLevel(level), confidence

        return MaslowLevel.BELONGING, 0.3  # default

    def get_intervention(self, level: MaslowLevel) -> list[str]:
        interventions: dict[int, list[str]] = {
            1: [
                "Let's do a quick body scan — notice your breathing, tension, comfort.",
                "Have you eaten and had water recently? Basic needs come first.",
                "A 5-minute grounding exercise can help reconnect with your body.",
            ],
            2: [
                "Let's identify one concrete step you can take for stability today.",
                "Problem-solving: what resources are available to you right now?",
                "Can we list what IS stable in your life to build from there?",
            ],
            3: [
                "You matter, and feeling disconnected doesn't mean you are alone.",
                "A micro-connection: text one person today — even 'thinking of you.'",
                "Self-compassion: you deserve the care you give others.",
            ],
            4: [
                "Let's identify 3 strengths you've shown this week, however small.",
                "That critical voice isn't the truth — let's examine the evidence.",
                "You're comparing your inside to everyone else's outside.",
            ],
            5: [
                "What would your life look like if it aligned with your deepest values?",
                "Purpose isn't found — it's built, one meaningful action at a time.",
                "Acceptance: the search for meaning IS meaning.",
            ],
        }
        return interventions.get(level.value, interventions[3])


# ---------------------------------------------------------------------------
# AttachmentStyleDetector
# ---------------------------------------------------------------------------
class AttachmentStyleDetector:
    """Infer attachment style from text history."""

    def infer_style(
        self, text_history: list[str],
    ) -> tuple[AttachmentStyle, float]:
        combined = " ".join(text_history).lower()
        scores: dict[str, float] = {}

        for style, keywords in _ATTACHMENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in combined)
            scores[style] = hits / max(len(keywords), 1)

        if not scores or max(scores.values()) < 0.1:
            return AttachmentStyle.SECURE, 0.2

        best_style = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = min(scores[best_style] * 2, 1.0)
        # Confidence scales with history length
        confidence *= min(len(text_history) / 10, 1.0)

        return AttachmentStyle(best_style), confidence

    def adapt_communication(self, style: AttachmentStyle) -> CommunicationAdaptations:
        adaptations: dict[AttachmentStyle, CommunicationAdaptations] = {
            AttachmentStyle.ANXIOUS_PREOCCUPIED: CommunicationAdaptations(
                tone="explicitly warm and reassuring",
                pacing="consistent, predictable check-ins",
                validation_level="high — validate before ANY redirect",
                challenge_approach="gentle, always sandwiched with reassurance",
                suggestions=[
                    "Start every response with explicit emotional validation.",
                    "Avoid long silences — they trigger abandonment fear.",
                    "Use 'I'm here' language frequently.",
                ],
            ),
            AttachmentStyle.DISMISSIVE_AVOIDANT: CommunicationAdaptations(
                tone="respectful, skills-focused, not emotionally pushy",
                pacing="respect their emotional distance",
                validation_level="moderate — validate without overwhelming",
                challenge_approach="intellectual, through skills and thoughts, not feelings",
                suggestions=[
                    "Focus on thoughts and actions, not 'how does that make you feel?'",
                    "Never pressure to share or open up.",
                    "Frame therapy as skill-building, not emotional processing.",
                ],
            ),
            AttachmentStyle.FEARFUL_AVOIDANT: CommunicationAdaptations(
                tone="extra safe, slow, no confrontation",
                pacing="very slow — celebrate small disclosures",
                validation_level="very high — safety is paramount",
                challenge_approach="minimal — build trust first over many sessions",
                suggestions=[
                    "Normalize ambivalence about closeness.",
                    "Never interpret their behavior as resistance.",
                    "Small steps are huge wins — acknowledge them.",
                ],
            ),
            AttachmentStyle.SECURE: CommunicationAdaptations(
                tone="warm, direct, collaborative",
                pacing="normal therapeutic pacing",
                validation_level="balanced",
                challenge_approach="direct Socratic questioning is appropriate",
                suggestions=[
                    "Standard warm supportive therapeutic tone.",
                    "Can explore deeper themes directly.",
                ],
            ),
        }
        return adaptations[style]


# ---------------------------------------------------------------------------
# CBTInterventionEngine
# ---------------------------------------------------------------------------
_DISTORTION_PATTERNS: dict[str, list[str]] = {
    "catastrophizing": [r"worst.*happen", r"disaster", r"ruined", r"end of", r"everything.*fall apart"],
    "all_or_nothing": [r"always", r"never", r"everyone", r"nobody", r"completely", r"totally"],
    "mind_reading": [r"they think", r"they must", r"everyone knows", r"people judge", r"they hate"],
    "fortune_telling": [r"will definitely", r"going to fail", r"won't work", r"never going to"],
    "personalization": [r"my fault", r"because of me", r"i caused", r"i ruined"],
    "should_statements": [r"i should", r"i must", r"i have to", r"ought to"],
    "emotional_reasoning": [r"i feel.*therefore", r"feels true", r"because i feel"],
    "labeling": [r"i am (a |an )?(failure|loser|idiot|stupid|worthless|pathetic)"],
}


class CBTInterventionEngine:
    """CBT-based therapeutic interventions."""

    def identify_nats(self, text: str) -> list[NegativeAutomaticThought]:
        nats: list[NegativeAutomaticThought] = []
        text_lower = text.lower()

        for distortion, patterns in _DISTORTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Extract surrounding context as the thought content
                    match = re.search(pattern, text_lower)
                    if match:
                        start = max(0, match.start() - 30)
                        end = min(len(text_lower), match.end() + 30)
                        thought = text[start:end].strip()

                        emotion_map = {
                            "catastrophizing": "Fear",
                            "all_or_nothing": "Sadness",
                            "mind_reading": "Anxiety",
                            "fortune_telling": "Fear",
                            "personalization": "Guilt",
                            "should_statements": "Shame",
                            "emotional_reasoning": "Distress",
                            "labeling": "Shame",
                        }

                        nats.append(NegativeAutomaticThought(
                            thought_content=thought,
                            distortion_type=distortion,
                            emotion_triggered=emotion_map.get(distortion, "Distress"),
                            intensity_estimate=0.6,
                        ))
                    break  # one hit per distortion type

        return nats

    def generate_thought_record(self, nat: NegativeAutomaticThought) -> ThoughtRecord:
        evidence_against_templates = {
            "catastrophizing": ["Has the worst case actually happened before?", "What happened last time you feared this?"],
            "all_or_nothing": ["Can you think of one exception?", "Is there a middle ground?"],
            "mind_reading": ["Have you verified what they actually think?", "Could there be another explanation?"],
            "fortune_telling": ["What evidence do you have for this prediction?", "How accurate were past predictions?"],
            "personalization": ["What other factors contributed?", "Would you blame a friend in the same situation?"],
            "should_statements": ["Where does this rule come from?", "Is this standard realistic for everyone?"],
            "emotional_reasoning": ["Do feelings always equal facts?", "Have you felt this way and been wrong before?"],
            "labeling": ["Does one moment define your entire identity?", "What would a kind friend call you instead?"],
        }

        balanced_thoughts = {
            "catastrophizing": "The worst case is possible but unlikely. I can handle what actually happens.",
            "all_or_nothing": "Life has shades of grey. Partial success is still success.",
            "mind_reading": "I don't know what others think unless I ask. My assumptions aren't facts.",
            "fortune_telling": "I cannot predict the future. I'll deal with what comes when it comes.",
            "personalization": "Many factors contribute to outcomes. I am not solely responsible.",
            "should_statements": "I can prefer things without demanding them. 'I would like to' is kinder than 'I should.'",
            "emotional_reasoning": "Feeling something strongly doesn't make it true.",
            "labeling": "I am a whole person, not a single label. One moment doesn't define me.",
        }

        return ThoughtRecord(
            situation=f"Triggered by: {nat.thought_content[:80]}",
            emotion=nat.emotion_triggered,
            emotion_intensity=nat.intensity_estimate,
            nat=nat.thought_content,
            evidence_for=["This thought feels very real right now."],
            evidence_against=evidence_against_templates.get(nat.distortion_type, ["Consider the evidence."]),
            balanced_thought=balanced_thoughts.get(nat.distortion_type, "A more balanced view exists."),
            outcome_emotion="Relief",
            outcome_intensity=max(0.0, nat.intensity_estimate - 0.3),
        )

    def socratic_questions(self, belief: str) -> list[str]:
        return [
            f"What evidence supports the belief that '{belief}'?",
            f"What evidence contradicts it? Is there another explanation?",
            f"What is the worst that could happen? The best? The most realistic?",
            f"What is the effect of believing this thought? What would change if you didn't?",
            f"If a close friend told you they believed '{belief}', what would you say to them?",
        ]

    def reframe_suggestion(self, nat: NegativeAutomaticThought) -> str:
        reframes = {
            "catastrophizing": f"Instead of '{nat.thought_content[:60]}…', try: 'This is hard, but I've survived hard things before.'",
            "all_or_nothing": f"Instead of absolutes, try: 'Sometimes this happens, and sometimes it doesn't.'",
            "labeling": f"Instead of labeling yourself, try: 'I made a mistake, and that's human — it doesn't define me.'",
            "personalization": f"Try: 'I played a part, but I'm not the only factor. I can learn without blaming myself.'",
        }
        return reframes.get(
            nat.distortion_type,
            f"A compassionate alternative: 'This is painful, and I deserve kindness right now.'",
        )

    def behavioral_activation_plan(self, anhedonia_score: float) -> list[ScheduledActivity]:
        if anhedonia_score <= 0.3:
            return [ScheduledActivity("Continue current pleasant activities", "pleasure", 30, "any", 1)]

        if anhedonia_score <= 0.7:
            return [
                ScheduledActivity("15-minute walk outside", "pleasure", 15, "morning", 1),
                ScheduledActivity("Call or text one person", "connection", 10, "afternoon", 1),
                ScheduledActivity("One small hobby activity", "pleasure", 20, "evening", 1),
            ]

        return [
            ScheduledActivity("Morning stretch routine", "mastery", 10, "morning", 1),
            ScheduledActivity("20-minute walk", "pleasure", 20, "morning", 1),
            ScheduledActivity("Prepare one meal mindfully", "mastery", 30, "midday", 2),
            ScheduledActivity("Contact one person", "connection", 15, "afternoon", 1),
            ScheduledActivity("Journaling: 3 observations", "mastery", 10, "evening", 1),
            ScheduledActivity("One enjoyable activity (music, show, read)", "pleasure", 30, "evening", 1),
        ]


# ---------------------------------------------------------------------------
# ACTTechniqueEngine
# ---------------------------------------------------------------------------
class ACTTechniqueEngine:
    """Acceptance and Commitment Therapy techniques."""

    _DEFUSION_METHODS = [
        (
            "notice",
            "Notice that you are having the thought: '{thought}'. "
            "Just notice it — like watching a cloud pass. You are not the thought. "
            "You are the sky it passes through.",
        ),
        (
            "name_it",
            "Your mind is telling you '{thought}' again. "
            "Thank your mind: 'Thanks, mind, for trying to protect me.' "
            "Then gently return to what matters to you right now.",
        ),
        (
            "leaves_on_stream",
            "Imagine sitting by a gentle stream. Each thought is a leaf floating on the water. "
            "Place the thought '{thought}' on a leaf and watch it drift away. "
            "If it comes back, place it on another leaf. No judgment.",
        ),
        (
            "silly_voice",
            "Try saying the thought '{thought}' in a silly cartoon voice, or sing it to the tune "
            "of 'Happy Birthday'. Notice how the words lose their grip when the delivery changes. "
            "The content is the same — but the power shifts.",
        ),
    ]

    def defusion_exercise(self, stuck_thought: str) -> str:
        method = random.choice(self._DEFUSION_METHODS)
        return method[1].format(thought=stuck_thought[:100])

    def values_clarification_prompt(self) -> list[str]:
        return [
            "If no one was watching and nothing could fail — what would you spend your time doing?",
            "What do you want your life to stand for when you look back?",
            "Who do you want to be in your most important relationships?",
            "What matters most to you when you're at your best?",
            "If you could write one sentence on your gravestone, what would it say?",
        ]

    def acceptance_script(self, emotion: str, intensity: int) -> str:
        intensity_word = {1: "gently", 2: "noticeably", 3: "strongly"}.get(intensity, "")
        return (
            f"Let's practice RAIN together.\n\n"
            f"**Recognize**: You're feeling {emotion} {intensity_word} right now. Name it: 'I feel {emotion.lower()}.'\n\n"
            f"**Allow**: Give it permission to be here. It's not dangerous — it's information. "
            f"Say: 'I can make space for this.'\n\n"
            f"**Investigate**: Where do you feel {emotion.lower()} in your body? "
            f"Chest? Stomach? Throat? Just notice, without trying to fix.\n\n"
            f"**Nurture**: Place a hand where you feel it. Ask gently: "
            f"'What do you need right now?' Listen for the answer."
        )


# ---------------------------------------------------------------------------
# BigFiveProfiler
# ---------------------------------------------------------------------------
class BigFiveProfiler:
    """Infer OCEAN personality traits from text history."""

    def infer_from_history(self, text_history: list[str]) -> BigFiveScores:
        if not text_history:
            return BigFiveScores()

        combined = " ".join(text_history).lower()
        total_words = max(len(combined.split()), 1)

        scores = BigFiveScores()
        for trait, markers in _BIG_FIVE_MARKERS.items():
            hits = sum(1 for m in markers if m in combined)
            raw = hits / max(len(markers), 1)
            setattr(scores, trait, min(0.3 + raw * 1.4, 1.0))  # baseline 0.3

        scores.confidence = min(len(text_history) / 15, 1.0)
        return scores

    def adapt_response_style(self, scores: BigFiveScores) -> ResponseAdaptations:
        notes: list[str] = []
        complexity = "moderate"
        social = True
        validation = 0.5
        structure = "moderate"

        if scores.neuroticism > 0.7:
            validation = 0.8
            notes.append("High neuroticism: more validation, gentler challenges, slower pacing.")
        if scores.extraversion < 0.3:
            social = False
            notes.append("Low extraversion: avoid 'go socialize' suggestions. Suggest solo activities.")
        if scores.openness > 0.7:
            complexity = "high"
            notes.append("High openness: can handle complex reframes, metaphors, and abstract concepts.")
        if scores.conscientiousness < 0.3:
            structure = "low"
            notes.append("Low conscientiousness: smaller action steps, looser structure, no rigid plans.")
        if scores.agreeableness > 0.7:
            notes.append("High agreeableness: may suppress own needs. Gently explore boundaries.")

        return ResponseAdaptations(
            complexity_level=complexity,
            social_suggestions=social,
            validation_weight=validation,
            structure_level=structure,
            notes=notes,
        )
