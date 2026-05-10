"""
MIRA LLM Service — FastAPI Router.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.companion.llm_client import MIRACompanion, ChatMessage
from app.memory.long_term_memory import LongTermMemoryStore
from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
from app.rag.retriever import TherapyKnowledgeRetriever

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

class ServiceState:
    """Singleton state container for LLM service."""
    def __init__(self):
        self._companion: Optional[MIRACompanion] = None
        self._memory: Optional[LongTermMemoryStore] = None
        self._retriever: Optional[TherapyKnowledgeRetriever] = None
        self._ingester: Optional[PsychologyKnowledgeIngester] = None

    @property
    def companion(self) -> MIRACompanion:
        if self._companion is None:
            self._companion = MIRACompanion()
        return self._companion

    @property
    def memory(self) -> LongTermMemoryStore:
        if self._memory is None:
            self._memory = LongTermMemoryStore()
        return self._memory

    @property
    def retriever(self) -> TherapyKnowledgeRetriever:
        if self._retriever is None:
            self._retriever = TherapyKnowledgeRetriever()
        return self._retriever

    @property
    def ingester(self) -> PsychologyKnowledgeIngester:
        if self._ingester is None:
            self._ingester = PsychologyKnowledgeIngester()
        return self._ingester

STATE = ServiceState()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    user_id: str = Field(..., min_length=1)
    session_id: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    dominant_emotion: str = "neutral"
    crisis_level: int = 0

class ChatSyncResponse(BaseModel):
    response: str
    model_used: str
    crisis_override: bool

@router.post("/chat/sync", response_model=ChatSyncResponse)
@limiter.limit("10/minute")
async def chat_sync(request: Request, body: ChatRequest) -> ChatSyncResponse:
    """Non-streaming chat endpoint with rate limiting."""
    try:
        companion = STATE.companion
        chat_history = [
            ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in body.conversation_history
        ]
        
        tokens: list[str] = []
        async for token in companion.generate_response(
            user_message=body.message,
            context="", # Context built inside client in this version
            conversation_history=chat_history,
            crisis_level=body.crisis_level,
            user_id=body.user_id,
            stream=False,
        ):
            tokens.append(token)

        return ChatSyncResponse(
            response="".join(tokens),
            model_used="mira-llm",
            crisis_override=body.crisis_level >= 3,
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="LLM service error")

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "llm-service"}
