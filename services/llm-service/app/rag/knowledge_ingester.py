"""
MIRA Psychology Knowledge Ingester.

Uses LlamaIndex for data ingestion + chunking, Sentence Transformers for
embeddings (FREE, local), and ChromaDB for vector storage.

Knowledge categories cover 9 evidence-based psychology frameworks used
by the MIRA companion for RAG-based therapeutic responses.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"
CHROMA_COLLECTION = "psychology_knowledge"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Evidence-based knowledge categories with metadata
KNOWLEDGE_CATEGORIES: dict[str, dict[str, str]] = {
    "cbt": {
        "theory": "Cognitive Behavioral Therapy",
        "author": "Aaron Beck",
        "year": "1979",
        "focus": "cognitive distortions, thought records, behavioral experiments",
    },
    "act": {
        "theory": "Acceptance and Commitment Therapy",
        "author": "Russ Harris",
        "year": "2009",
        "focus": "defusion, acceptance, values, committed action",
    },
    "maslow": {
        "theory": "Hierarchy of Needs",
        "author": "Abraham Maslow",
        "year": "1943",
        "focus": "5-level hierarchy, interventions per level",
    },
    "attachment": {
        "theory": "Attachment Theory",
        "author": "Bowlby & Ainsworth",
        "year": "1969",
        "focus": "4 attachment styles, adaptation strategies",
    },
    "plutchik": {
        "theory": "Wheel of Emotions",
        "author": "Robert Plutchik",
        "year": "1980",
        "focus": "40 emotions, PAD model, dyads, intensity levels",
    },
    "behaviourism": {
        "theory": "Behavioral Psychology",
        "author": "B.F. Skinner",
        "year": "1938",
        "focus": "reinforcement, behavioral activation, habit loops",
    },
    "ekman": {
        "theory": "Facial Action Coding System",
        "author": "Paul Ekman",
        "year": "1972",
        "focus": "FACS, AU mapping, micro-expressions, deception detection",
    },
    "rogers": {
        "theory": "Person-Centered Therapy",
        "author": "Carl Rogers",
        "year": "1959",
        "focus": "ideal vs real self, congruence, self-compassion, unconditional positive regard",
    },
    "crisis": {
        "theory": "Crisis Intervention",
        "author": "Stanley & Brown",
        "year": "2012",
        "focus": "safety planning, Stanley-Brown protocol, grounding techniques",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IngestionReport:
    """Result of a knowledge ingestion run."""

    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    theories_covered: list[str] = field(default_factory=list)
    documents_processed: int = 0


# ---------------------------------------------------------------------------
# PsychologyKnowledgeIngester
# ---------------------------------------------------------------------------
class PsychologyKnowledgeIngester:
    """
    Ingests psychology knowledge documents into ChromaDB via
    LlamaIndex readers and Sentence Transformer embeddings.
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
            logger.warning("ChromaDB HTTP client unavailable, using ephemeral client.")
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_all(self, knowledge_dir: Path | None = None) -> IngestionReport:
        """
        Ingest all supported files from the knowledge directory.

        Reads PDFs and TXT files, chunks them, embeds, and stores in ChromaDB.
        """
        source_dir = knowledge_dir or KNOWLEDGE_DIR
        report = IngestionReport()

        if not source_dir.exists():
            logger.warning(f"Knowledge directory does not exist: {source_dir}")
            report.errors.append(f"Directory not found: {source_dir}")
            return report

        # Collect all files
        supported_extensions = {".txt", ".md", ".pdf"}
        files = [
            f for f in source_dir.rglob("*")
            if f.suffix.lower() in supported_extensions and f.is_file()
        ]

        if not files:
            logger.info("No knowledge files found to ingest.")
            return report

        for file_path in files:
            try:
                # Auto-detect category from parent folder name
                category = self._detect_category(file_path)
                chunks_added = self.add_document(
                    file_path=file_path,
                    theory_category=category,
                    evidence_level="peer_reviewed",
                )
                report.chunks_created += chunks_added
                report.documents_processed += 1
                if category not in report.theories_covered:
                    report.theories_covered.append(category)
            except Exception as ingest_error:
                error_msg = f"Failed to ingest {file_path.name}: {ingest_error}"
                logger.error(error_msg)
                report.errors.append(error_msg)

        logger.info(
            f"Ingestion complete: {report.chunks_created} chunks from "
            f"{report.documents_processed} documents covering {report.theories_covered}"
        )
        return report

    def add_document(
        self,
        file_path: Path,
        theory_category: str,
        evidence_level: str = "peer_reviewed",
    ) -> int:
        """
        Add a single document: read → chunk → embed → store.

        Returns number of chunks created.
        """
        text = self._read_file(file_path)
        if not text.strip():
            return 0

        chunks = self._chunk_text(text)
        category_meta = KNOWLEDGE_CATEGORIES.get(theory_category, {})

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{file_path.name}:{chunk_index}:{chunk_text[:50]}".encode()
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "source_title": file_path.stem,
                "source_file": file_path.name,
                "theory_category": theory_category,
                "theory_name": category_meta.get("theory", theory_category),
                "author": category_meta.get("author", "unknown"),
                "year_published": category_meta.get("year", "unknown"),
                "evidence_level": evidence_level,
                "chunk_index": str(chunk_index),
            })

        if ids:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(f"Ingested {len(ids)} chunks from {file_path.name} [{theory_category}]")
        return len(ids)

    def add_text_directly(
        self,
        text: str,
        source_title: str,
        theory_category: str,
        evidence_level: str = "curated",
    ) -> int:
        """Add raw text content directly (no file needed)."""
        chunks = self._chunk_text(text)
        category_meta = KNOWLEDGE_CATEGORIES.get(theory_category, {})

        ids = []
        documents = []
        metadatas = []

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{source_title}:{chunk_index}:{chunk_text[:50]}".encode()
            ).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "source_title": source_title,
                "source_file": "direct_input",
                "theory_category": theory_category,
                "theory_name": category_meta.get("theory", theory_category),
                "author": category_meta.get("author", "unknown"),
                "year_published": category_meta.get("year", "unknown"),
                "evidence_level": evidence_level,
                "chunk_index": str(chunk_index),
            })

        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return len(ids)

    def get_collection_stats(self) -> dict[str, Any]:
        """Return collection count and metadata."""
        return {
            "total_chunks": self._collection.count(),
            "collection_name": self._collection.name,
        }

    # ── Private helpers ──

    def _read_file(self, file_path: Path) -> str:
        """Read text from TXT/MD files. PDF uses LlamaIndex reader."""
        if file_path.suffix.lower() == ".pdf":
            return self._read_pdf(file_path)
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF using LlamaIndex SimpleDirectoryReader."""
        try:
            from llama_index.core import SimpleDirectoryReader

            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            documents = reader.load_data()
            return "\n\n".join(doc.text for doc in documents)
        except Exception as pdf_error:
            logger.warning(f"PDF read failed for {file_path}: {pdf_error}")
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > CHUNK_SIZE // 2:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            stripped = chunk.strip()
            if stripped:
                chunks.append(stripped)

            start = end - CHUNK_OVERLAP

        return chunks

    def _detect_category(self, file_path: Path) -> str:
        """Auto-detect theory category from folder or filename."""
        path_lower = str(file_path).lower()
        for category_key in KNOWLEDGE_CATEGORIES:
            if category_key in path_lower:
                return category_key
        return "general"
