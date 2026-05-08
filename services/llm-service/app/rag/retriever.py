"""
MIRA Therapy Knowledge Retriever.

Multi-query RAG retrieval that searches the psychology knowledge base
using emotion, distortion, and Maslow-level derived queries, then
merges, deduplicates, and reranks results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "psychology_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Maslow level descriptions for query enrichment
_MASLOW_DESCRIPTIONS: dict[int, str] = {
    1: "physiological needs: sleep, food, safety, basic survival",
    2: "safety needs: stability, security, freedom from fear, routine",
    3: "love and belonging: relationships, connection, community, intimacy",
    4: "esteem needs: confidence, achievement, recognition, self-worth",
    5: "self-actualization: growth, purpose, meaning, creativity, peak experiences",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class KnowledgeChunk:
    """A single retrieved knowledge chunk with source metadata."""

    text: str
    source_title: str
    theory_category: str
    theory_name: str
    author: str
    year_published: str
    evidence_level: str
    relevance_score: float  # 0.0 = perfect match (cosine distance)


# ---------------------------------------------------------------------------
# TherapyKnowledgeRetriever
# ---------------------------------------------------------------------------
class TherapyKnowledgeRetriever:
    """
    Multi-query RAG retriever for psychology knowledge.

    Generates 3 parallel queries based on:
      1. Dominant emotion → therapy intervention
      2. Detected cognitive distortion → CBT technique
      3. Maslow level → support strategy
    Merges, deduplicates, and reranks by relevance.
    """

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = CHROMA_COLLECTION,
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

    def retrieve(
        self,
        query: str,
        dominant_emotion: str = "",
        distortions: Optional[dict[str, bool]] = None,
        maslow_level: int = 3,
        k: int = 5,
    ) -> list[KnowledgeChunk]:
        """
        Multi-query retrieval with dedup and reranking.

        Args:
            query: user's raw text or question
            dominant_emotion: primary detected emotion (e.g. "Sadness")
            distortions: dict of distortion_name → detected (True/False)
            maslow_level: user's current Maslow hierarchy level (1-5)
            k: number of chunks to return
        """
        queries: list[str] = []

        # Query 1: Emotion-based
        if dominant_emotion:
            queries.append(f"{dominant_emotion} therapy intervention technique")

        # Query 2: Distortion-based
        if distortions:
            detected = [name for name, active in distortions.items() if active]
            if detected:
                primary_distortion = detected[0]
                queries.append(f"{primary_distortion} CBT technique cognitive behavioral therapy")

        # Query 3: Maslow-level based
        maslow_desc = _MASLOW_DESCRIPTIONS.get(maslow_level, "")
        if maslow_desc:
            queries.append(f"maslow level {maslow_level} {maslow_desc} support strategy")

        # Fallback: use raw query
        if not queries:
            queries.append(query)

        # Run all queries and merge
        all_results: dict[str, tuple[KnowledgeChunk, float]] = {}

        for sub_query in queries:
            try:
                results = self._collection.query(
                    query_texts=[sub_query],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )

                if not results["documents"] or not results["documents"][0]:
                    continue

                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunk_key = doc[:100]  # dedup by first 100 chars
                    chunk = KnowledgeChunk(
                        text=doc,
                        source_title=meta.get("source_title", "unknown"),
                        theory_category=meta.get("theory_category", "general"),
                        theory_name=meta.get("theory_name", "unknown"),
                        author=meta.get("author", "unknown"),
                        year_published=meta.get("year_published", "unknown"),
                        evidence_level=meta.get("evidence_level", "unknown"),
                        relevance_score=float(distance),
                    )
                    # Keep the best (lowest distance) per chunk
                    if chunk_key not in all_results or distance < all_results[chunk_key][1]:
                        all_results[chunk_key] = (chunk, float(distance))

            except Exception as query_error:
                logger.warning(f"RAG query failed for '{sub_query[:50]}': {query_error}")

        # Sort by relevance (lowest distance first) and take top k
        sorted_chunks = sorted(all_results.values(), key=lambda x: x[1])
        return [chunk for chunk, _ in sorted_chunks[:k]]

    def build_rag_context(
        self,
        dominant_emotion: str,
        distortions: Optional[dict[str, bool]] = None,
        maslow_level: int = 3,
        k: int = 3,
    ) -> str:
        """
        Build a formatted RAG context string for injection into the LLM
        system prompt.
        """
        chunks = self.retrieve(
            query=dominant_emotion,
            dominant_emotion=dominant_emotion,
            distortions=distortions,
            maslow_level=maslow_level,
            k=k,
        )

        if not chunks:
            return "No specific therapy knowledge retrieved for this context."

        context_parts: list[str] = []
        for idx, chunk in enumerate(chunks, 1):
            citation = f"[{chunk.theory_name}, {chunk.author} ({chunk.year_published})]"
            context_parts.append(
                f"--- Knowledge {idx} {citation} ---\n{chunk.text}"
            )

        return "\n\n".join(context_parts)

    def retrieve_by_theory(self, theory_category: str, k: int = 5) -> list[KnowledgeChunk]:
        """Retrieve chunks filtered by a specific theory category."""
        try:
            results = self._collection.query(
                query_texts=[f"{theory_category} therapy technique"],
                n_results=k,
                where={"theory_category": theory_category},
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            chunks: list[KnowledgeChunk] = []
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(KnowledgeChunk(
                    text=doc,
                    source_title=meta.get("source_title", "unknown"),
                    theory_category=meta.get("theory_category", "general"),
                    theory_name=meta.get("theory_name", "unknown"),
                    author=meta.get("author", "unknown"),
                    year_published=meta.get("year_published", "unknown"),
                    evidence_level=meta.get("evidence_level", "unknown"),
                    relevance_score=float(distance),
                ))
            return chunks

        except Exception as retrieve_error:
            logger.warning(f"Theory retrieval failed: {retrieve_error}")
            return []
