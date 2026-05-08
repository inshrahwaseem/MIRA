"""
MIRA LLM Service — Test Suite.

Covers: knowledge ingester, retriever, long-term memory, context builder,
LLM client, crisis templates, agent tools, and router schemas.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===================================================================
# 1. Knowledge Ingester Tests
# ===================================================================
class TestKnowledgeIngester:
    """Tests for PsychologyKnowledgeIngester."""

    def test_knowledge_categories_complete(self):
        from app.rag.knowledge_ingester import KNOWLEDGE_CATEGORIES
        required = {"cbt", "act", "maslow", "attachment", "plutchik", "behaviourism", "ekman", "rogers", "crisis"}
        assert required.issubset(set(KNOWLEDGE_CATEGORIES.keys()))

    def test_category_has_required_fields(self):
        from app.rag.knowledge_ingester import KNOWLEDGE_CATEGORIES
        for key, cat in KNOWLEDGE_CATEGORIES.items():
            assert "theory" in cat, f"{key} missing 'theory'"
            assert "author" in cat, f"{key} missing 'author'"
            assert "year" in cat, f"{key} missing 'year'"
            assert "focus" in cat, f"{key} missing 'focus'"

    def test_chunk_text_splits_correctly(self):
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester.__new__(PsychologyKnowledgeIngester)
        text = "A" * 1200  # should produce 3+ chunks at 512 chunk size
        chunks = ingester._chunk_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 520  # small overshoot ok at sentence boundary

    def test_chunk_text_handles_empty(self):
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester.__new__(PsychologyKnowledgeIngester)
        chunks = ingester._chunk_text("")
        assert chunks == []

    def test_detect_category_from_path(self):
        from pathlib import Path
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester.__new__(PsychologyKnowledgeIngester)
        assert ingester._detect_category(Path("/data/cbt/beck_chapter1.txt")) == "cbt"
        assert ingester._detect_category(Path("/data/maslow/hierarchy.txt")) == "maslow"
        assert ingester._detect_category(Path("/data/random/file.txt")) == "general"

    def test_add_text_directly(self):
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester()
        count = ingester.add_text_directly(
            text="CBT thought records help identify automatic negative thoughts.",
            source_title="test_doc",
            theory_category="cbt",
        )
        assert count >= 1

    def test_collection_stats(self):
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester()
        stats = ingester.get_collection_stats()
        assert "total_chunks" in stats
        assert "collection_name" in stats


# ===================================================================
# 2. Retriever Tests
# ===================================================================
class TestTherapyRetriever:
    """Tests for TherapyKnowledgeRetriever."""

    def _seed_data(self):
        from app.rag.knowledge_ingester import PsychologyKnowledgeIngester
        ingester = PsychologyKnowledgeIngester()
        ingester.add_text_directly(
            text="Cognitive behavioral therapy uses thought records to challenge negative automatic thoughts.",
            source_title="cbt_basics",
            theory_category="cbt",
        )
        ingester.add_text_directly(
            text="Acceptance and commitment therapy teaches defusion techniques for anxiety.",
            source_title="act_basics",
            theory_category="act",
        )

    def test_retrieve_returns_chunks(self):
        from app.rag.retriever import TherapyKnowledgeRetriever
        self._seed_data()
        retriever = TherapyKnowledgeRetriever()
        chunks = retriever.retrieve(
            query="anxiety treatment",
            dominant_emotion="Fear",
            k=3,
        )
        assert isinstance(chunks, list)

    def test_build_rag_context_returns_string(self):
        from app.rag.retriever import TherapyKnowledgeRetriever
        self._seed_data()
        retriever = TherapyKnowledgeRetriever()
        context = retriever.build_rag_context(
            dominant_emotion="Sadness",
            distortions={"catastrophizing": True},
            maslow_level=3,
        )
        assert isinstance(context, str)
        assert len(context) > 0

    def test_retrieve_empty_query(self):
        from app.rag.retriever import TherapyKnowledgeRetriever
        retriever = TherapyKnowledgeRetriever()
        chunks = retriever.retrieve(query="", k=3)
        assert isinstance(chunks, list)


# ===================================================================
# 3. Long-Term Memory Tests
# ===================================================================
class TestLongTermMemory:
    """Tests for LongTermMemoryStore."""

    def test_add_and_retrieve_memory(self):
        from app.memory.long_term_memory import LongTermMemoryStore
        store = LongTermMemoryStore()
        doc_id = store.add_memory(
            session_id="test-session-001",
            summary="User expressed sadness about work stress and isolation.",
            emotion_dominant="Sadness",
            key_topics=["work", "isolation", "stress"],
            user_id="test-user-001",
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

        memories = store.retrieve_similar(
            current_emotion_text="feeling sad and lonely",
            user_id="test-user-001",
            k=3,
        )
        assert len(memories) >= 1
        assert memories[0].dominant_emotion == "Sadness"

    def test_retrieve_by_topic(self):
        from app.memory.long_term_memory import LongTermMemoryStore
        store = LongTermMemoryStore()
        store.add_memory(
            session_id="topic-test",
            summary="Discussion about relationship anxiety.",
            emotion_dominant="Fear",
            key_topics=["relationship", "anxiety"],
            user_id="test-user-002",
        )
        memories = store.retrieve_by_topic("relationship", user_id="test-user-002", k=3)
        assert isinstance(memories, list)

    def test_memory_count(self):
        from app.memory.long_term_memory import LongTermMemoryStore
        store = LongTermMemoryStore()
        store.add_memory(
            session_id="count-test",
            summary="Test session.",
            emotion_dominant="Joy",
            key_topics=["test"],
            user_id="test-user-003",
        )
        count = store.get_user_memory_count("test-user-003")
        assert count >= 1

    def test_no_memories_for_unknown_user(self):
        from app.memory.long_term_memory import LongTermMemoryStore
        store = LongTermMemoryStore()
        memories = store.retrieve_similar(
            current_emotion_text="anything",
            user_id="nonexistent-user-xyz",
            k=3,
        )
        assert memories == []


# ===================================================================
# 4. Context Builder Tests
# ===================================================================
class TestConversationContextBuilder:
    """Tests for ConversationContextBuilder."""

    def test_builds_structured_context(self):
        from app.memory.long_term_memory import (
            ConversationContextBuilder,
            ConversationMessage,
            LongTermMemoryStore,
        )
        store = LongTermMemoryStore()
        builder = ConversationContextBuilder(store)

        context = builder.build_context(
            dominant_emotion="Anxiety",
            intensity=2,
            valence=-0.4,
            arousal=0.7,
            cluster_name="anxiety + anticipation",
            drift_alert_level=1,
            distortions={"catastrophizing": True, "fortune_telling": True},
            maslow_level=2,
            rag_context="CBT thought records can help challenge catastrophic thinking.",
            conversation_history=[
                ConversationMessage(role="user", content="I feel anxious about tomorrow."),
                ConversationMessage(role="assistant", content="Tell me more about what worries you."),
            ],
            user_id="test-user-001",
            attachment_style="anxious_preoccupied",
            theory_applied="CBT",
        )

        assert "[Current Emotional State]" in context
        assert "Anxiety" in context
        assert "[User Profile]" in context
        assert "CBT" in context
        assert "[Cognitive Distortions Detected]" in context
        assert "catastrophizing" in context

    def test_context_without_distortions(self):
        from app.memory.long_term_memory import (
            ConversationContextBuilder,
            LongTermMemoryStore,
        )
        store = LongTermMemoryStore()
        builder = ConversationContextBuilder(store)
        context = builder.build_context(
            dominant_emotion="Joy",
            intensity=2,
            valence=0.8,
            arousal=0.5,
            cluster_name="joy + trust",
            drift_alert_level=0,
            distortions={},
            maslow_level=4,
            rag_context="",
            conversation_history=[],
            user_id="test-user-001",
        )
        assert "Cognitive Distortions" not in context


# ===================================================================
# 5. Crisis Response Template Tests
# ===================================================================
class TestCrisisResponse:
    """Tests for CrisisResponseTemplate."""

    def test_immediate_response(self):
        from app.companion.llm_client import CrisisResponseTemplate
        resp = CrisisResponseTemplate.get_response(0)
        assert "safe" in resp.lower() or "yahaan" in resp.lower()

    def test_grounding_exercise(self):
        from app.companion.llm_client import CrisisResponseTemplate
        resp = CrisisResponseTemplate.get_response(1)
        assert "5" in resp  # 5-4-3-2-1 exercise

    def test_resource_mention(self):
        from app.companion.llm_client import CrisisResponseTemplate
        resp = CrisisResponseTemplate.get_response(2)
        assert "Helpline" in resp or "helpline" in resp.lower()

    def test_responses_are_not_empty(self):
        from app.companion.llm_client import CrisisResponseTemplate
        for exchange_count in range(5):
            resp = CrisisResponseTemplate.get_response(exchange_count)
            assert len(resp) > 10


# ===================================================================
# 6. LLM Client Tests
# ===================================================================
class TestOllamaClient:
    """Tests for OllamaClient."""

    def test_client_creates(self):
        from app.companion.llm_client import OllamaClient
        client = OllamaClient(base_url="http://localhost:11434", model="llama3")
        assert client._base_url == "http://localhost:11434"
        assert client._model == "llama3"

    def test_chat_message_dataclass(self):
        from app.companion.llm_client import ChatMessage
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestGroqClient:
    """Tests for GroqFallbackClient."""

    def test_not_available_without_key(self):
        from app.companion.llm_client import GroqFallbackClient
        client = GroqFallbackClient(api_key="")
        assert client.available is False

    def test_available_with_key(self):
        from app.companion.llm_client import GroqFallbackClient
        client = GroqFallbackClient(api_key="test-key-123")
        assert client.available is True


class TestMIRACompanion:
    """Tests for MIRACompanion."""

    def test_system_prompt_template_has_placeholders(self):
        from app.companion.llm_client import MIRACompanion
        template = MIRACompanion.SYSTEM_PROMPT_TEMPLATE
        assert "{theory_applied}" in template
        assert "{context}" in template

    def test_build_messages_truncation(self):
        from app.companion.llm_client import ChatMessage, MIRACompanion
        companion = MIRACompanion()
        # Create a large history
        history = [
            ChatMessage(role="user", content="x" * 500)
            for _ in range(20)
        ]
        messages = companion._build_messages(
            system_prompt="System prompt" * 10,
            conversation_history=history,
            user_message="Current message",
        )
        # Should have system + some history + current
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        assert messages[-1].content == "Current message"
        # Should be truncated (not all 20 messages)
        assert len(messages) < 22

    def test_activity_prescription_dataclass(self):
        from app.companion.llm_client import ActivityPrescription
        rx = ActivityPrescription(
            name="Thought Record",
            theory_basis="CBT",
            target_emotion="Sadness",
            instructions="Write down the negative thought.",
            difficulty_level=2,
            estimated_minutes=15,
        )
        assert rx.difficulty_level == 2


# ===================================================================
# 7. Agent Tool Tests
# ===================================================================
class TestAgentTools:
    """Tests for MIRA agent tools."""

    def test_breathing_exercise_478(self):
        from app.agent.mira_agent import trigger_breathing_exercise
        result = trigger_breathing_exercise.invoke("4-7-8")
        data = json.loads(result)
        assert data["inhale_seconds"] == 4
        assert data["hold_seconds"] == 7
        assert data["exhale_seconds"] == 8

    def test_breathing_exercise_box(self):
        from app.agent.mira_agent import trigger_breathing_exercise
        result = trigger_breathing_exercise.invoke("box")
        data = json.loads(result)
        assert data["name"] == "Box Breathing"

    def test_breathing_exercise_grounding(self):
        from app.agent.mira_agent import trigger_breathing_exercise
        result = trigger_breathing_exercise.invoke("grounding")
        data = json.loads(result)
        assert "steps" in data
        assert len(data["steps"]) == 5

    def test_assess_crisis_level_high(self):
        from app.agent.mira_agent import assess_crisis_level
        result = assess_crisis_level.invoke({"text": "I want to kill myself", "valence": -0.9})
        assert "Level: 3" in result

    def test_assess_crisis_level_none(self):
        from app.agent.mira_agent import assess_crisis_level
        result = assess_crisis_level.invoke({"text": "I had a great day today", "valence": 0.8})
        assert "Level: 0" in result

    def test_assess_crisis_level_medium(self):
        from app.agent.mira_agent import assess_crisis_level
        result = assess_crisis_level.invoke({"text": "I feel hopeless and worthless", "valence": -0.6})
        assert "Level: 2" in result

    def test_recommend_prescription_cbt_sadness(self):
        from app.agent.mira_agent import recommend_prescription
        result = recommend_prescription.invoke({"emotion": "Sadness", "theory": "CBT"})
        data = json.loads(result)
        assert "Thought Record" in data["name"]

    def test_recommend_prescription_fallback(self):
        from app.agent.mira_agent import recommend_prescription
        result = recommend_prescription.invoke({"emotion": "Confusion", "theory": "Unknown"})
        data = json.loads(result)
        assert "name" in data
        assert "instructions" in data

    def test_mood_visualization_valid(self):
        from app.agent.mira_agent import generate_mood_visualization
        result = generate_mood_visualization.invoke({"user_id": "test-user", "chart_type": "radar"})
        assert "radar" in result

    def test_mood_visualization_invalid(self):
        from app.agent.mira_agent import generate_mood_visualization
        result = generate_mood_visualization.invoke({"user_id": "test-user", "chart_type": "invalid"})
        assert "Unknown" in result

    def test_all_tools_count(self):
        from app.agent.mira_agent import ALL_TOOLS
        assert len(ALL_TOOLS) == 7


# ===================================================================
# 8. Router Schema Tests
# ===================================================================
class TestRouterSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_chat_request_validation(self):
        from app.routers.llm import ChatRequest
        req = ChatRequest(message="Hello", user_id="user-001")
        assert req.dominant_emotion == "neutral"
        assert req.crisis_level == 0
        assert req.maslow_level == 3

    def test_chat_request_rejects_empty_message(self):
        from app.routers.llm import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="", user_id="user-001")

    def test_chat_request_rejects_empty_user(self):
        from app.routers.llm import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="Hello", user_id="")

    def test_memory_add_request(self):
        from app.routers.llm import MemoryAddRequest
        req = MemoryAddRequest(
            session_id="sess-001",
            user_id="user-001",
            summary="User discussed work stress.",
            dominant_emotion="Anxiety",
            key_topics=["work", "stress"],
        )
        assert req.dominant_emotion == "Anxiety"

    def test_health_response(self):
        from app.routers.llm import HealthResponse
        resp = HealthResponse(
            status="healthy",
            ollama_ok=True,
            groq_available=True,
            chroma_ok=True,
            knowledge_chunks=150,
            memory_collection_ok=True,
        )
        assert resp.status == "healthy"
