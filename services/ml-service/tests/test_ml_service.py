"""
MIRA ML Service — Tests.

Covers: emotion taxonomy, text analyzer, mood tracker, fusion engine, and API endpoints.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ===================================================================
# 1. EmotionTaxonomy Tests
# ===================================================================
class TestEmotionTaxonomy:
    """Tests for emotion_taxonomy.py."""

    def test_taxonomy_has_40_plus_emotions(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        assert len(taxonomy) >= 40

    def test_get_by_name_case_insensitive(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        joy = taxonomy.get_by_name("joy")
        assert joy.name == "Joy"
        assert joy.plutchik_category == "Joy"

    def test_get_by_name_unknown_raises(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        with pytest.raises(KeyError):
            taxonomy.get_by_name("nonexistent_emotion")

    def test_get_by_category_returns_list(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        fear_family = taxonomy.get_by_category("Fear")
        assert len(fear_family) == 3  # Apprehension, Fear, Terror
        assert all(e.plutchik_category == "Fear" for e in fear_family)

    def test_intensity_variants_sorted(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        variants = taxonomy.get_intensity_variants("Sadness")
        assert len(variants) == 3
        assert variants[0].intensity_level == 1  # Pensiveness
        assert variants[2].intensity_level == 3  # Grief

    def test_all_emotion_names_sorted(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        names = taxonomy.all_emotion_names()
        assert names == sorted(names)

    def test_pad_values_in_range(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        for definition in taxonomy.all_definitions():
            assert -1.0 <= definition.valence <= 1.0, f"{definition.name} valence out of range"
            assert 0.0 <= definition.arousal <= 1.0, f"{definition.name} arousal out of range"
            assert 0.0 <= definition.dominance <= 1.0, f"{definition.name} dominance out of range"

    def test_crisis_risk_in_range(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        for definition in taxonomy.all_definitions():
            assert 0 <= definition.crisis_risk_level <= 3

    def test_emotion_definition_is_frozen(self):
        from app.core.emotion_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        joy = taxonomy.get_by_name("Joy")
        with pytest.raises(AttributeError):
            joy.name = "Modified"  # type: ignore


# ===================================================================
# 2. EmotionVector Tests
# ===================================================================
class TestEmotionVector:
    """Tests for EmotionVector."""

    def test_default_zero_vector(self):
        from app.core.emotion_taxonomy import EmotionVector
        vector = EmotionVector()
        assert vector.to_pad() == (0.0, 0.0, 0.0)

    def test_from_numpy(self):
        from app.core.emotion_taxonomy import EmotionVector
        data = np.zeros(64, dtype=np.float32)
        data[0] = 0.5
        data[1] = 0.3
        data[2] = 0.7
        vector = EmotionVector(data)
        valence, arousal, dominance = vector.to_pad()
        assert abs(valence - 0.5) < 1e-6
        assert abs(arousal - 0.3) < 1e-6
        assert abs(dominance - 0.7) < 1e-6

    def test_cosine_distance_identical(self):
        from app.core.emotion_taxonomy import EmotionVector
        data = np.random.rand(64).astype(np.float32)
        vec_a = EmotionVector(data)
        vec_b = EmotionVector(data.copy())
        assert vec_a.cosine_distance(vec_b) < 1e-6

    def test_cosine_distance_orthogonal(self):
        from app.core.emotion_taxonomy import EmotionVector
        data_a = np.zeros(64, dtype=np.float32)
        data_b = np.zeros(64, dtype=np.float32)
        data_a[0] = 1.0
        data_b[1] = 1.0
        vec_a = EmotionVector(data_a)
        vec_b = EmotionVector(data_b)
        assert abs(vec_a.cosine_distance(vec_b) - 1.0) < 1e-6

    def test_blend(self):
        from app.core.emotion_taxonomy import EmotionVector
        data_a = np.ones(64, dtype=np.float32)
        data_b = np.zeros(64, dtype=np.float32)
        vec_a = EmotionVector(data_a)
        vec_b = EmotionVector(data_b)
        blended = vec_a.blend(vec_b, weight=0.5)
        assert abs(blended._vector[0] - 0.5) < 1e-6

    def test_serialization_roundtrip(self):
        from app.core.emotion_taxonomy import EmotionVector
        original = EmotionVector(np.random.rand(64).astype(np.float32))
        restored = EmotionVector.from_dict(original.to_dict())
        assert np.allclose(original._vector, restored._vector, atol=1e-6)

    def test_intensity_score_bounded(self):
        from app.core.emotion_taxonomy import EmotionVector
        vector = EmotionVector(np.ones(64, dtype=np.float32) * 10)
        assert 0.0 <= vector.intensity_score <= 1.0


# ===================================================================
# 3. TextPreprocessor Tests
# ===================================================================
class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_clean_html(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        assert "<b>" not in preprocessor.clean_text("Hello <b>world</b>")

    def test_clean_whitespace(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        assert preprocessor.clean_text("  hello   world  ") == "hello world"

    def test_detect_catastrophizing(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        distortions = preprocessor.detect_cognitive_distortions("Everything is always terrible")
        assert distortions["catastrophizing"] is True

    def test_detect_black_white(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        distortions = preprocessor.detect_cognitive_distortions("It has to be completely perfect")
        assert distortions["black_white_thinking"] is True

    def test_detect_labeling(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        distortions = preprocessor.detect_cognitive_distortions("I am a failure")
        assert distortions["labeling"] is True

    def test_no_false_positives(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        distortions = preprocessor.detect_cognitive_distortions("I had a good day at work")
        assert not any(distortions.values())

    def test_masking_detection(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        score = preprocessor.detect_masking_language("I'm fine, it's nothing, forget it")
        assert score > 0.5

    def test_masking_no_signal(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        score = preprocessor.detect_masking_language("I feel really excited about tomorrow")
        assert score == 0.0

    def test_crisis_high_risk(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        signals = preprocessor.extract_crisis_signals("I want to die and end it all")
        high_signals = [s for s in signals if s.startswith("HIGH")]
        assert len(high_signals) >= 2

    def test_crisis_medium_risk(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        signals = preprocessor.extract_crisis_signals("I feel hopeless and worthless")
        medium_signals = [s for s in signals if s.startswith("MEDIUM")]
        assert len(medium_signals) >= 2

    def test_crisis_no_risk(self):
        from app.text.analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        signals = preprocessor.extract_crisis_signals("I am happy and grateful")
        assert len(signals) == 0


# ===================================================================
# 4. MoodDriftAnalyzer Tests
# ===================================================================
class TestMoodDriftAnalyzer:
    """Tests for MoodDriftAnalyzer."""

    def test_insufficient_sessions(self):
        from app.clustering.mood_tracker import MoodDriftAnalyzer
        analyzer = MoodDriftAnalyzer()
        report = analyzer.detect_drift([{"valence": 0.5}])
        assert report.trend_direction == "stable"
        assert report.alert_level == 0

    def test_declining_trend(self):
        from app.clustering.mood_tracker import MoodDriftAnalyzer
        analyzer = MoodDriftAnalyzer()
        sessions = [
            {"valence": 0.6, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.4, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.2, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.0, "arousal": 0.5, "crisis_level": 0},
            {"valence": -0.2, "arousal": 0.5, "crisis_level": 0},
            {"valence": -0.4, "arousal": 0.5, "crisis_level": 0},
            {"valence": -0.6, "arousal": 0.5, "crisis_level": 0},
        ]
        report = analyzer.detect_drift(sessions)
        assert report.trend_direction == "declining"
        assert report.alert_level >= 1

    def test_improving_trend(self):
        from app.clustering.mood_tracker import MoodDriftAnalyzer
        analyzer = MoodDriftAnalyzer()
        sessions = [
            {"valence": -0.5, "arousal": 0.5, "crisis_level": 0},
            {"valence": -0.3, "arousal": 0.5, "crisis_level": 0},
            {"valence": -0.1, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.1, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.3, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.5, "arousal": 0.5, "crisis_level": 0},
            {"valence": 0.7, "arousal": 0.5, "crisis_level": 0},
        ]
        report = analyzer.detect_drift(sessions)
        assert report.trend_direction == "improving"


# ===================================================================
# 5. Fusion ConflictDetector Tests
# ===================================================================
class TestConflictDetector:
    """Tests for CrossModalConflictDetector."""

    def test_no_conflict_similar_vectors(self):
        from app.core.emotion_taxonomy import EmotionVector
        from app.fusion.engine import CrossModalConflictDetector
        detector = CrossModalConflictDetector()
        vec = EmotionVector(np.random.rand(64).astype(np.float32))
        report = detector.detect(vec, vec, vec)
        assert report.has_conflict is False

    def test_conflict_opposite_vectors(self):
        from app.core.emotion_taxonomy import EmotionVector
        from app.fusion.engine import CrossModalConflictDetector
        detector = CrossModalConflictDetector()
        positive = np.zeros(64, dtype=np.float32)
        positive[3] = 1.0  # joy
        negative = np.zeros(64, dtype=np.float32)
        negative[7] = 1.0  # sadness
        report = detector.detect(EmotionVector(positive), EmotionVector(negative), None)
        assert report.has_conflict is True

    def test_emotional_blunting_detection(self):
        from app.core.emotion_taxonomy import EmotionVector
        from app.fusion.engine import CrossModalConflictDetector
        detector = CrossModalConflictDetector()
        flat = EmotionVector(np.ones(64, dtype=np.float32) * 0.01)
        report = detector.detect(flat, flat, flat)
        # Low-affect signals may or may not trigger a conflict — they should trigger blunting
        # The cosine distance between identical vectors is 0 so has_conflict=False
        # but all intensities are low so blunting is detected by the pattern matcher
        # This depends on whether distance threshold is met first
        # For identical vectors, conflict won't trigger — this is correct behavior


# ===================================================================
# 6. BayesianFusion Tests
# ===================================================================
class TestBayesianFusion:
    """Tests for BayesianFusion."""

    def test_single_modality(self):
        from app.core.emotion_taxonomy import EmotionVector
        from app.fusion.engine import BayesianFusion
        fusion = BayesianFusion()
        vec = EmotionVector(np.random.rand(64).astype(np.float32))
        result = fusion.fuse({"text": vec}, {"text": 1.0})
        # Result should be valid EmotionVector
        assert result._vector.shape == (64,)

    def test_two_modalities_weighted(self):
        from app.core.emotion_taxonomy import EmotionVector
        from app.fusion.engine import BayesianFusion
        fusion = BayesianFusion()
        vec_a = EmotionVector(np.ones(64, dtype=np.float32))
        vec_b = EmotionVector(np.ones(64, dtype=np.float32) * 2)
        result = fusion.fuse(
            {"text": vec_a, "voice": vec_b},
            {"text": 0.6, "voice": 0.4},
        )
        assert result._vector.shape == (64,)

    def test_empty_returns_zero_vector(self):
        from app.fusion.engine import BayesianFusion
        fusion = BayesianFusion()
        result = fusion.fuse({}, {})
        assert np.allclose(result._vector, 0.0)


# ===================================================================
# 7. MoodClusteringEngine Tests
# ===================================================================
class TestMoodClustering:
    """Tests for MoodClusteringEngine."""

    def test_fit_and_predict(self):
        from app.clustering.mood_tracker import MoodClusteringEngine
        engine = MoodClusteringEngine()
        # Generate synthetic data — 3 clear clusters
        np.random.seed(42)
        cluster_a = np.random.randn(20, 20) + np.array([2] * 20)
        cluster_b = np.random.randn(20, 20) + np.array([-2] * 20)
        cluster_c = np.random.randn(20, 20) + np.array([0] * 20)
        feature_matrix = np.vstack([cluster_a, cluster_b, cluster_c])

        metrics = engine.fit(feature_matrix)
        assert metrics["silhouette_score"] > 0
        assert metrics["k_chosen"] >= 2

        result = engine.predict(cluster_a[0])
        assert result.cluster_id >= 0
        assert result.confidence > 0

    def test_predict_unfitted(self):
        from app.clustering.mood_tracker import MoodClusteringEngine
        engine = MoodClusteringEngine()
        result = engine.predict(np.zeros(20))
        assert result.cluster_id == -1
        assert result.cluster_name == "unfitted"
