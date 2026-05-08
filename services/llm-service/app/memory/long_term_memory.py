"""
MIRA Long-Term Memory Store + Conversation Context Builder.

ChromaDB-backed session memory with per-user isolation,
similarity-based recall, and auto-compression of old memories.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "session_memories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Maslow level descriptions
_MASLOW_LABELS: dict[int, str] = {
    1: "Physiological (survival)",
    2: "Safety (security)",
    3: "Love & Belonging",
    4: "Esteem (self-worth)",
    5: "Self-Actualization",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MemoryDoc:
    """A single retrieved memory document."""

    session_id: str
    summary: str
    dominant_emotion: str
    key_topics: list[str]
    created_at: str
    relevance_score: float


@dataclass
class ConversationMessage:
    """One message in the conversation history."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[float] = None


# ---------------------------------------------------------------------------
# LongTermMemoryStore
# ---------------------------------------------------------------------------
class LongTermMemoryStore:
    """
    ChromaDB-backed memory store with per-user isolation.

    Each memory is a session summary stored with metadata:
    user_id, session_id, dominant_emotion, key_topics, created_at.
    """

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = MEMORY_COLLECTION,
    ) -> None:
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
        try:
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        except Exception:
            logger.warning("ChromaDB HTTP unavailable, using ephemeral client.")
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        session_id: str,
        summary: str,
        emotion_dominant: str,
        key_topics: list[str],
        user_id: str,
    ) -> str:
        """
        Store a session memory in ChromaDB.

        Returns the generated document ID.
        """
        doc_id = hashlib.md5(
            f"{user_id}:{session_id}:{summary[:50]}".encode()
        ).hexdigest()

        self._collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{
                "user_id": user_id,
                "session_id": session_id,
                "dominant_emotion": emotion_dominant,
                "key_topics": ",".join(key_topics),
                "created_at": str(int(time.time())),
            }],
        )

        logger.info(f"Memory added: user={user_id[:8]}… session={session_id[:8]}… emotion={emotion_dominant}")
        return doc_id

    def retrieve_similar(
        self,
        current_emotion_text: str,
        user_id: str,
        k: int = 5,
    ) -> list[MemoryDoc]:
        """
        Retrieve k most similar past sessions for this user.
        """
        try:
            results = self._collection.query(
                query_texts=[current_emotion_text],
                n_results=k,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            memories: list[MemoryDoc] = []
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                memories.append(MemoryDoc(
                    session_id=meta.get("session_id", ""),
                    summary=doc,
                    dominant_emotion=meta.get("dominant_emotion", ""),
                    key_topics=meta.get("key_topics", "").split(","),
                    created_at=meta.get("created_at", ""),
                    relevance_score=float(distance),
                ))
            return memories

        except Exception as retrieval_error:
            logger.warning(f"Memory retrieval failed: {retrieval_error}")
            return []

    def retrieve_by_topic(
        self,
        topic: str,
        user_id: str,
        k: int = 3,
    ) -> list[MemoryDoc]:
        """Retrieve memories matching a specific topic."""
        return self.retrieve_similar(
            current_emotion_text=topic,
            user_id=user_id,
            k=k,
        )

    def get_user_memory_count(self, user_id: str) -> int:
        """Count total memories for a user."""
        try:
            results = self._collection.get(
                where={"user_id": user_id},
                include=[],
            )
            return len(results["ids"])
        except Exception:
            return 0

    async def auto_compress_old(
        self,
        user_id: str,
        older_than_days: int = 30,
        ollama_base_url: str = "http://localhost:11434",
    ) -> int:
        """
        Compress old memories (>30 days) by summarizing clusters
        with Ollama and replacing them with a single summary.

        Returns number of memories compressed.
        """
        import httpx

        cutoff_timestamp = str(int(time.time()) - older_than_days * 86400)

        try:
            all_memories = self._collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            return 0

        old_ids: list[str] = []
        old_summaries: list[str] = []

        for doc_id, doc, meta in zip(
            all_memories["ids"],
            all_memories["documents"],
            all_memories["metadatas"],
        ):
            created = meta.get("created_at", "0")
            if created < cutoff_timestamp:
                old_ids.append(doc_id)
                old_summaries.append(doc)

        if len(old_summaries) < 3:
            return 0  # not enough to compress

        # Ask Ollama to summarize
        combined_text = "\n\n---\n\n".join(old_summaries[:20])  # cap at 20
        prompt = (
            "Summarize the following therapy session notes into a single concise paragraph "
            "that captures the key emotional themes, progress, and recurring patterns:\n\n"
            f"{combined_text}"
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{ollama_base_url}/api/generate",
                    json={"model": "llama3", "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                compressed_summary = response.json().get("response", combined_text[:500])
        except Exception as ollama_error:
            logger.warning(f"Ollama compression failed: {ollama_error}")
            compressed_summary = f"Compressed summary of {len(old_summaries)} sessions."

        # Delete old memories
        self._collection.delete(ids=old_ids)

        # Add compressed memory
        self.add_memory(
            session_id="compressed",
            summary=compressed_summary,
            emotion_dominant="mixed",
            key_topics=["compressed", "historical"],
            user_id=user_id,
        )

        logger.info(f"Compressed {len(old_ids)} old memories for user {user_id[:8]}…")
        return len(old_ids)


# ---------------------------------------------------------------------------
# ConversationContextBuilder
# ---------------------------------------------------------------------------
class ConversationContextBuilder:
    """
    Builds structured context for the LLM system prompt by combining:
      - Short-term: last 10 messages (sliding window)
      - Long-term: top 5 similar past sessions from ChromaDB
      - RAG knowledge: therapy techniques from knowledge base
      - User profile: attachment style, Big Five, history
    """

    MAX_SHORT_TERM_MESSAGES = 10

    def __init__(self, memory_store: LongTermMemoryStore) -> None:
        self._memory = memory_store

    def build_context(
        self,
        dominant_emotion: str,
        intensity: int,
        valence: float,
        arousal: float,
        cluster_name: str,
        drift_alert_level: int,
        distortions: dict[str, bool],
        maslow_level: int,
        rag_context: str,
        conversation_history: list[ConversationMessage],
        user_id: str,
        attachment_style: str = "unknown",
        theory_applied: str = "CBT",
    ) -> str:
        """
        Build the complete structured context string for injection
        into the LLM system prompt.
        """
        sections: list[str] = []

        # [1] Current emotional state
        sections.append(self._build_current_state(
            dominant_emotion, intensity, valence, arousal,
            cluster_name, drift_alert_level,
        ))

        # [2] User profile
        sections.append(self._build_user_profile(
            attachment_style, maslow_level, theory_applied,
        ))

        # [3] Cognitive distortions
        detected = [name for name, active in distortions.items() if active]
        if detected:
            sections.append(f"[Cognitive Distortions Detected]\n" + ", ".join(detected))

        # [4] RAG knowledge
        if rag_context:
            sections.append(f"[Relevant Therapy Knowledge]\n{rag_context}")

        # [5] Similar past sessions
        past_sessions = self._memory.retrieve_similar(
            current_emotion_text=dominant_emotion,
            user_id=user_id,
            k=5,
        )
        if past_sessions:
            past_text = "\n".join(
                f"- ({mem.dominant_emotion}) {mem.summary[:150]}"
                for mem in past_sessions[:3]
            )
            sections.append(f"[Similar Past Sessions]\n{past_text}")

        # [6] Recent conversation (sliding window)
        recent = conversation_history[-self.MAX_SHORT_TERM_MESSAGES:]
        if recent:
            conv_text = "\n".join(
                f"{msg.role.upper()}: {msg.content[:200]}" for msg in recent
            )
            sections.append(f"[Recent Conversation]\n{conv_text}")

        return "\n\n".join(sections)

    def _build_current_state(
        self,
        emotion: str, intensity: int, valence: float, arousal: float,
        cluster: str, drift: int,
    ) -> str:
        """Format current emotional state section."""
        drift_text = f"Level {drift} — declining trend detected" if drift > 0 else "None"
        return (
            f"[Current Emotional State]\n"
            f"Dominant emotion: {emotion} at intensity {intensity}/3\n"
            f"Valence: {valence:.2f}, Arousal: {arousal:.2f}\n"
            f"Mood cluster: {cluster}\n"
            f"Drift alert: {drift_text}"
        )

    def _build_user_profile(
        self,
        attachment: str, maslow: int, theory: str,
    ) -> str:
        """Format user profile section."""
        maslow_desc = _MASLOW_LABELS.get(maslow, "Unknown")
        return (
            f"[User Profile]\n"
            f"Attachment style: {attachment}\n"
            f"Maslow level: {maslow} — {maslow_desc}\n"
            f"Theory to apply: {theory}"
        )
