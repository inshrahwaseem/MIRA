"""
MIRA LLM Service — FastAPI Router.

Endpoints:
  POST /chat          → StreamingResponse (SSE token stream)
  POST /chat/sync     → ChatResponse
  POST /memory/add    → MemoryAddResult
  POST /knowledge/ingest → IngestResult
  GET  /health        → service status
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.companion.llm_client import (
    ChatMessage,
    CrisisResponseTemplate,
    MIRACompanion,
)
from app.memory.long_term_memory import ConversationContextBuilder, ConversationMessage, LongTermMemoryStore
from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
from app.rag.retriever import TherapyKnowledgeRetriever

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Singletons (lazy)
# ---------------------------------------------------------------------------
_companion: MIRACompanion | None = None
_memory: LongTermMemoryStore | None = None
_retriever: TherapyKnowledgeRetriever | None = None
_ingester: PsychologyKnowledgeIngester | None = None


def _get_companion() -> MIRACompanion:
    global _companion
    if _companion is None:
        _companion = MIRACompanion()
    return _companion


def _get_memory() -> LongTermMemoryStore:
    global _memory
    if _memory is None:
        _memory = LongTermMemoryStore()
    return _memory


def _get_retriever() -> TherapyKnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TherapyKnowledgeRetriever()
    return _retriever


def _get_ingester() -> PsychologyKnowledgeIngester:
    global _ingester
    if _ingester is None:
        _ingester = PsychologyKnowledgeIngester()
    return _ingester


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Request body for POST /chat and /chat/sync."""
    model_config = {"str_strip_whitespace": True}

    message: str = Field(..., min_length=1, max_length=5000)
    user_id: str = Field(..., min_length=1)
    session_id: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # Emotion context (from ML service)
    dominant_emotion: str = "neutral"
    intensity: int = 1
    valence: float = 0.0
    arousal: float = 0.0
    cluster_name: str = "unknown"
    drift_alert_level: int = 0
    maslow_level: int = 3
    attachment_style: str = "unknown"
    theory_applied: str = "CBT"
    cognitive_distortions: dict[str, bool] = Field(default_factory=dict)
    crisis_level: int = 0


class ChatSyncResponse(BaseModel):
    """Response for POST /chat/sync."""

    response: str
    model_used: str
    processing_time_ms: float
    crisis_override: bool


class MemoryAddRequest(BaseModel):
    """Request body for POST /memory/add."""

    session_id: str
    user_id: str
    summary: str = Field(..., min_length=1, max_length=5000)
    dominant_emotion: str
    key_topics: list[str] = Field(default_factory=list)


class MemoryAddResponse(BaseModel):
    """Response for POST /memory/add."""

    doc_id: str
    status: str


class IngestRequest(BaseModel):
    """Request body for POST /knowledge/ingest."""

    text: str = Field(..., min_length=1)
    source_title: str
    theory_category: str
    evidence_level: str = "curated"


class IngestResponse(BaseModel):
    """Response for POST /knowledge/ingest."""

    chunks_created: int
    status: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    ollama_ok: bool
    groq_available: bool
    chroma_ok: bool
    knowledge_chunks: int
    memory_collection_ok: bool


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------
async def _sse_stream(request: ChatRequest):
    """Generate SSE events for streaming chat."""
    companion = _get_companion()
    memory = _get_memory()
    retriever = _get_retriever()

    # Build RAG context
    rag_context = retriever.build_rag_context(
        dominant_emotion=request.dominant_emotion,
        distortions=request.cognitive_distortions,
        maslow_level=request.maslow_level,
        k=3,
    )

    # Build conversation context
    context_builder = ConversationContextBuilder(memory)
    conv_history = [
        ConversationMessage(role=m.get("role", "user"), content=m.get("content", ""))
        for m in request.conversation_history
    ]
    context = context_builder.build_context(
        dominant_emotion=request.dominant_emotion,
        intensity=request.intensity,
        valence=request.valence,
        arousal=request.arousal,
        cluster_name=request.cluster_name,
        drift_alert_level=request.drift_alert_level,
        distortions=request.cognitive_distortions,
        maslow_level=request.maslow_level,
        rag_context=rag_context,
        conversation_history=conv_history,
        user_id=request.user_id,
        attachment_style=request.attachment_style,
        theory_applied=request.theory_applied,
    )

    # Build ChatMessage history
    chat_history = [
        ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
        for m in request.conversation_history
    ]

    # Stream tokens
    full_response: list[str] = []
    async for token in companion.generate_response(
        user_message=request.message,
        context=context,
        conversation_history=chat_history,
        crisis_level=request.crisis_level,
        user_id=request.user_id,
        theory_applied=request.theory_applied,
        stream=True,
    ):
        full_response.append(token)
        event = json.dumps({"type": "token", "content": token})
        yield f"data: {event}\n\n"

    # Emotion insight event
    insight = json.dumps({
        "type": "emotion_insight",
        "content": {
            "dominant_emotion": request.dominant_emotion,
            "intensity": request.intensity,
            "valence": request.valence,
            "cluster": request.cluster_name,
        },
    })
    yield f"data: {insight}\n\n"

    # Done event
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat endpoint."""
    try:
        return StreamingResponse(
            _sse_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as chat_error:
        logger.error(f"Chat stream failed: {chat_error}")
        raise HTTPException(status_code=500, detail="Chat service unavailable.")


@router.post("/chat/sync", response_model=ChatSyncResponse)
async def chat_sync(request: ChatRequest) -> ChatSyncResponse:
    """Non-streaming chat endpoint."""
    start_time = time.perf_counter()

    try:
        companion = _get_companion()
        memory = _get_memory()
        retriever = _get_retriever()

        rag_context = retriever.build_rag_context(
            dominant_emotion=request.dominant_emotion,
            distortions=request.cognitive_distortions,
            maslow_level=request.maslow_level,
            k=3,
        )

        context_builder = ConversationContextBuilder(memory)
        conv_history = [
            ConversationMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in request.conversation_history
        ]
        context = context_builder.build_context(
            dominant_emotion=request.dominant_emotion,
            intensity=request.intensity,
            valence=request.valence,
            arousal=request.arousal,
            cluster_name=request.cluster_name,
            drift_alert_level=request.drift_alert_level,
            distortions=request.cognitive_distortions,
            maslow_level=request.maslow_level,
            rag_context=rag_context,
            conversation_history=conv_history,
            user_id=request.user_id,
            attachment_style=request.attachment_style,
            theory_applied=request.theory_applied,
        )

        chat_history = [
            ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in request.conversation_history
        ]

        tokens: list[str] = []
        async for token in companion.generate_response(
            user_message=request.message,
            context=context,
            conversation_history=chat_history,
            crisis_level=request.crisis_level,
            user_id=request.user_id,
            theory_applied=request.theory_applied,
            stream=False,
        ):
            tokens.append(token)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ChatSyncResponse(
            response="".join(tokens),
            model_used="ollama/llama3",
            processing_time_ms=elapsed_ms,
            crisis_override=request.crisis_level >= 3,
        )
    except Exception as sync_error:
        logger.error(f"Sync chat failed: {sync_error}")
        raise HTTPException(status_code=500, detail="Chat service unavailable.")


@router.post("/memory/add", response_model=MemoryAddResponse)
async def memory_add(request: MemoryAddRequest) -> MemoryAddResponse:
    """Add a session memory to long-term storage."""
    try:
        memory = _get_memory()
        doc_id = memory.add_memory(
            session_id=request.session_id,
            summary=request.summary,
            emotion_dominant=request.dominant_emotion,
            key_topics=request.key_topics,
            user_id=request.user_id,
        )
        return MemoryAddResponse(doc_id=doc_id, status="stored")
    except Exception as memory_error:
        logger.error(f"Memory add failed: {memory_error}")
        raise HTTPException(status_code=500, detail="Memory service unavailable.")


@router.post("/knowledge/ingest", response_model=IngestResponse)
async def knowledge_ingest(request: IngestRequest) -> IngestResponse:
    """Ingest a text document into the psychology knowledge base."""
    try:
        ingester = _get_ingester()
        chunks = ingester.add_text_directly(
            text=request.text,
            source_title=request.source_title,
            theory_category=request.theory_category,
            evidence_level=request.evidence_level,
        )
        return IngestResponse(chunks_created=chunks, status="ingested")
    except Exception as ingest_error:
        logger.error(f"Knowledge ingestion failed: {ingest_error}")
        raise HTTPException(status_code=500, detail="Ingestion service unavailable.")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check."""
    from app.companion.llm_client import GroqFallbackClient, OllamaClient

    ollama_ok = False
    try:
        ollama = OllamaClient()
        ollama_ok = await ollama.health_check()
    except Exception:
        pass

    groq_available = bool(GroqFallbackClient().available)

    chroma_ok = False
    knowledge_chunks = 0
    try:
        ingester = _get_ingester()
        stats = ingester.get_collection_stats()
        knowledge_chunks = stats.get("total_chunks", 0)
        chroma_ok = True
    except Exception:
        pass

    memory_ok = False
    try:
        _get_memory()
        memory_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (ollama_ok or groq_available) else "degraded",
        ollama_ok=ollama_ok,
        groq_available=groq_available,
        chroma_ok=chroma_ok,
        knowledge_chunks=knowledge_chunks,
        memory_collection_ok=memory_ok,
    )
