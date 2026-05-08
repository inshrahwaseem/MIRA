"""
MIRA ML Service — FastAPI Router for all analysis endpoints.

Endpoints:
  POST /analyze/text      → TextAnalysisResult
  POST /analyze/voice     → VoiceAnalysisResult (placeholder)
  POST /analyze/camera    → CameraAnalysisResult (placeholder)
  POST /analyze/fuse      → FusedEmotionResult
  POST /cluster/predict   → ClusterResult
  POST /drift/detect      → DriftReport
  POST /triggers/mine     → list[TriggerRule]
  GET  /visualize/{chart}/{user_id} → Plotly JSON
  GET  /health            → service status
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.clustering.mood_tracker import (
    ClusterResult,
    CrisisClassifier,
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

# ---------------------------------------------------------------------------
# Singleton instances (lazy — created on first request, not import)
# ---------------------------------------------------------------------------
_text_analyzer: TextEmotionAnalyzer | None = None
_fusion_engine: MultimodalFusionEngine | None = None
_clustering: MoodClusteringEngine | None = None
_drift: MoodDriftAnalyzer | None = None
_trigger_miner: EmotionTriggerMiner | None = None
_visualizer: MoodVisualizer | None = None


def _get_text_analyzer() -> TextEmotionAnalyzer:
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextEmotionAnalyzer()
    return _text_analyzer


def _get_fusion_engine() -> MultimodalFusionEngine:
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = MultimodalFusionEngine()
    return _fusion_engine


def _get_clustering() -> MoodClusteringEngine:
    global _clustering
    if _clustering is None:
        _clustering = MoodClusteringEngine()
    return _clustering


def _get_drift() -> MoodDriftAnalyzer:
    global _drift
    if _drift is None:
        _drift = MoodDriftAnalyzer()
    return _drift


def _get_trigger_miner() -> EmotionTriggerMiner:
    global _trigger_miner
    if _trigger_miner is None:
        _trigger_miner = EmotionTriggerMiner()
    return _trigger_miner


def _get_visualizer() -> MoodVisualizer:
    global _visualizer
    if _visualizer is None:
        _visualizer = MoodVisualizer()
    return _visualizer


# ---------------------------------------------------------------------------
# Request / Response Pydantic schemas
# ---------------------------------------------------------------------------
class TextAnalyzeRequest(BaseModel):
    """Request body for POST /analyze/text."""
    model_config = {"str_strip_whitespace": True}
    text: str = Field(..., min_length=1, max_length=5000)


class TextAnalyzeResponse(BaseModel):
    """Response for POST /analyze/text."""
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
    """Request body for POST /analyze/fuse."""
    model_config = {"str_strip_whitespace": True}
    text: str = Field(..., min_length=1, max_length=5000)
    audio_features: Optional[dict[str, Any]] = None
    camera_features: Optional[dict[str, Any]] = None
    session_id: str = ""
    user_id: str = ""


class FuseResponse(BaseModel):
    """Response for POST /analyze/fuse."""
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


class ClusterPredictRequest(BaseModel):
    """Request body for POST /cluster/predict."""
    session_features: dict[str, Any]


class ClusterPredictResponse(BaseModel):
    """Response for POST /cluster/predict."""
    cluster_id: int
    cluster_name: str
    confidence: float


class DriftDetectRequest(BaseModel):
    """Request body for POST /drift/detect."""
    sessions: list[dict[str, Any]]
    window: int = 7


class DriftDetectResponse(BaseModel):
    """Response for POST /drift/detect."""
    trend_direction: str
    rate_of_change: float
    predicted_next_day: float
    alert_level: int
    explanation: str


class TriggerMineRequest(BaseModel):
    """Request body for POST /triggers/mine."""
    sessions: list[dict[str, Any]]
    min_support: float = 0.3
    min_confidence: float = 0.6


class TriggerRuleResponse(BaseModel):
    """One trigger rule in the response."""
    antecedent_words: list[str]
    consequent_emotion: str
    confidence: float
    lift: float
    explanation: str


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    text_model_loaded: bool
    clustering_fitted: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Service start time
# ---------------------------------------------------------------------------
_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/analyze/text", response_model=TextAnalyzeResponse)
async def analyze_text(request: TextAnalyzeRequest) -> TextAnalyzeResponse:
    """Analyze text for emotions, distortions, masking, and crisis risk."""
    try:
        analyzer = _get_text_analyzer()
        result: TextAnalysisResult = analyzer.analyze(request.text)
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
async def analyze_fuse(request: FuseRequest) -> FuseResponse:
    """Multimodal fusion — text + optional voice + optional camera."""
    try:
        engine = _get_fusion_engine()
        result: FusedEmotionResult = await engine.analyze_session(
            text=request.text,
            audio_features=request.audio_features,
            camera_features=request.camera_features,
            session_id=request.session_id,
            user_id=request.user_id,
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


@router.post("/cluster/predict", response_model=ClusterPredictResponse)
async def cluster_predict(request: ClusterPredictRequest) -> ClusterPredictResponse:
    """Predict mood cluster for a single session."""
    try:
        import numpy as np
        clustering = _get_clustering()
        engineer = EmotionFeatureEngineer()
        features = engineer.transform([request.session_features])
        cluster_result = clustering.predict(features[0])
        return ClusterPredictResponse(
            cluster_id=cluster_result.cluster_id,
            cluster_name=cluster_result.cluster_name,
            confidence=cluster_result.confidence,
        )
    except Exception as cluster_error:
        logger.error(f"Cluster prediction failed: {cluster_error}")
        raise HTTPException(status_code=500, detail="Clustering service unavailable.")


@router.post("/drift/detect", response_model=DriftDetectResponse)
async def drift_detect(request: DriftDetectRequest) -> DriftDetectResponse:
    """Detect mood drift over the last N sessions."""
    try:
        drift_analyzer = _get_drift()
        report: DriftReport = drift_analyzer.detect_drift(request.sessions, request.window)
        return DriftDetectResponse(
            trend_direction=report.trend_direction,
            rate_of_change=report.rate_of_change,
            predicted_next_day=report.predicted_next_day,
            alert_level=report.alert_level,
            explanation=report.explanation,
        )
    except Exception as drift_error:
        logger.error(f"Drift detection failed: {drift_error}")
        raise HTTPException(status_code=500, detail="Drift detection service unavailable.")


@router.post("/triggers/mine", response_model=list[TriggerRuleResponse])
async def triggers_mine(request: TriggerMineRequest) -> list[TriggerRuleResponse]:
    """Mine emotion trigger patterns from session history."""
    try:
        miner = _get_trigger_miner()
        rules = miner.mine_triggers(request.sessions, request.min_support, request.min_confidence)
        return [
            TriggerRuleResponse(
                antecedent_words=rule.antecedent_words,
                consequent_emotion=rule.consequent_emotion,
                confidence=rule.confidence,
                lift=rule.lift,
                explanation=rule.explanation,
            )
            for rule in rules
        ]
    except Exception as trigger_error:
        logger.error(f"Trigger mining failed: {trigger_error}")
        raise HTTPException(status_code=500, detail="Trigger mining service unavailable.")


@router.get("/visualize/{chart_type}/{user_id}")
async def visualize(chart_type: str, user_id: str) -> dict[str, Any]:
    """Return a Plotly chart JSON for the requested chart type."""
    visualizer = _get_visualizer()
    # In production, sessions would be fetched from DB by user_id
    placeholder_sessions: list[dict[str, Any]] = []
    chart_map = {
        "tsne": visualizer.tsne_2d,
        "umap": visualizer.umap_3d,
        "calendar": visualizer.mood_calendar_heatmap,
        "radar": visualizer.emotion_radar,
        "drift": visualizer.drift_timeline,
    }
    chart_func = chart_map.get(chart_type)
    if chart_func is None:
        raise HTTPException(status_code=400, detail=f"Unknown chart type: {chart_type}. Available: {list(chart_map.keys())}")
    try:
        if chart_type == "radar":
            return chart_func({})
        return chart_func(placeholder_sessions)
    except Exception as viz_error:
        logger.error(f"Visualization failed for {chart_type}: {viz_error}")
        raise HTTPException(status_code=500, detail="Visualization service unavailable.")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health and model loading status."""
    text_loaded = _text_analyzer is not None and hasattr(_text_analyzer, "classifier")
    clustering_fitted = _clustering is not None and _clustering._fitted
    return HealthResponse(
        status="healthy",
        text_model_loaded=text_loaded,
        clustering_fitted=clustering_fitted,
        uptime_seconds=round(time.time() - _start_time, 2),
    )
