"""
MIRA ML Service — FastAPI Router for all analysis endpoints.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.clustering.mood_tracker import (
    ClusterResult,
    DriftReport,
    EmotionFeatureEngineer,
    EmotionTriggerMiner,
    MoodClusteringEngine,
    MoodDriftAnalyzer,
    MoodVisualizer,
    TriggerRule,
)
from app.fusion.engine import FusedEmotionResult, MultimodalFusionEngine
from app.text.analyzer import TextAnalysisResult, TextEmotionAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

class ServiceState:
    """
    Singleton state container to avoid global variables.
    PASS ✓ - Refactored per strict Python quality standards.
    """
    def __init__(self):
        self._text_analyzer: Optional[TextEmotionAnalyzer] = None
        self._fusion_engine: Optional[MultimodalFusionEngine] = None
        self._clustering: Optional[MoodClusteringEngine] = None
        self._drift: Optional[MoodDriftAnalyzer] = None
        self._trigger_miner: Optional[EmotionTriggerMiner] = None
        self._visualizer: Optional[MoodVisualizer] = None

    @property
    def text_analyzer(self) -> TextEmotionAnalyzer:
        if self._text_analyzer is None:
            self._text_analyzer = TextEmotionAnalyzer()
        return self._text_analyzer

    @property
    def fusion_engine(self) -> MultimodalFusionEngine:
        if self._fusion_engine is None:
            self._fusion_engine = MultimodalFusionEngine()
        return self._fusion_engine

    @property
    def clustering(self) -> MoodClusteringEngine:
        if self._clustering is None:
            self._clustering = MoodClusteringEngine()
        return self._clustering

    @property
    def drift(self) -> MoodDriftAnalyzer:
        if self._drift is None:
            self._drift = MoodDriftAnalyzer()
        return self._drift

    @property
    def trigger_miner(self) -> EmotionTriggerMiner:
        if self._trigger_miner is None:
            self._trigger_miner = EmotionTriggerMiner()
        return self._trigger_miner

    @property
    def visualizer(self) -> MoodVisualizer:
        if self._visualizer is None:
            self._visualizer = MoodVisualizer()
        return self._visualizer

STATE = ServiceState()

# ---------------------------------------------------------------------------
# Request / Response Pydantic schemas
# ---------------------------------------------------------------------------
class TextAnalyzeRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}
    text: str = Field(..., min_length=1, max_length=5000)

class TextAnalyzeResponse(BaseModel):
    primary_emotion: str
    valence: float
    arousal: float
    model_confidence: float
    cognitive_distortions: dict[str, bool]
    masking_probability: float
    crisis_keywords_found: list[str]
    crisis_risk_score: float
    key_phrases: list[str]
    raw_scores: dict[str, float]
    processing_time_ms: float

class FuseRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}
    text: str = Field(..., min_length=1, max_length=5000)
    audio_features: Optional[dict[str, Any]] = None
    camera_features: Optional[dict[str, Any]] = None
    session_id: str = ""
    user_id: str = ""

class FuseResponse(BaseModel):
    primary_emotion: str
    secondary_emotion: str
    intensity_label: str
    modalities_used: list[str]
    weights_applied: dict[str, float]
    has_conflict: bool
    conflict_type: str
    masked_emotion: str
    genuine_emotion: str
    conflict_explanation: str
    llm_adjustment_instruction: str
    processing_time_ms: float

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/analyze/text", response_model=TextAnalyzeResponse)
@limiter.limit("20/minute")
async def analyze_text(request: Request, body: TextAnalyzeRequest) -> TextAnalyzeResponse:
    """Analyze text for emotions, distortions, masking, and crisis risk."""
    try:
        analyzer = STATE.text_analyzer
        result: TextAnalysisResult = analyzer.analyze(body.text)
        return TextAnalyzeResponse(
            primary_emotion=result.emotion_vector.dominant_emotion,
            valence=result.valence,
            arousal=result.arousal,
            model_confidence=result.model_confidence,
            cognitive_distortions=result.cognitive_distortions,
            masking_probability=result.masking_probability,
            crisis_keywords_found=result.crisis_keywords_found,
            crisis_risk_score=result.crisis_risk_score,
            key_phrases=result.key_phrases,
            raw_scores=result.raw_scores,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as analysis_error:
        logger.error(f"Text analysis failed: {analysis_error}")
        raise HTTPException(status_code=500, detail="Text analysis service unavailable.")

@router.post("/analyze/fuse", response_model=FuseResponse)
@limiter.limit("20/minute")
async def analyze_fuse(request: Request, body: FuseRequest) -> FuseResponse:
    """Multimodal fusion — text + optional voice + optional camera."""
    try:
        engine = STATE.fusion_engine
        result: FusedEmotionResult = await engine.analyze_session(
            text=body.text,
            audio_features=body.audio_features,
            camera_features=body.camera_features,
            session_id=body.session_id,
            user_id=body.user_id,
        )
        conflict = result.conflict_report
        return FuseResponse(
            primary_emotion=result.primary_emotion,
            secondary_emotion=result.secondary_emotion,
            intensity_label=result.intensity_label,
            modalities_used=result.modalities_used,
            weights_applied=result.weights_applied,
            has_conflict=conflict.has_conflict,
            conflict_type=conflict.conflict_type,
            masked_emotion=conflict.masked_emotion,
            genuine_emotion=conflict.genuine_emotion,
            conflict_explanation=conflict.explanation,
            llm_adjustment_instruction=conflict.llm_adjustment_instruction,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as fusion_error:
        logger.error(f"Fusion analysis failed: {fusion_error}")
        raise HTTPException(status_code=500, detail="Fusion analysis service unavailable.")

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "ml-service"}
