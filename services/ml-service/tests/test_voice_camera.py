"""
MIRA Voice & Camera Module Tests.

Covers: voice feature extraction, emotion mapping, clinical flags,
face AU extraction, smile detection, gaze EAR, posture analysis,
hand gesture detection, and cross-modal conflict detection.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pytest


# ===================================================================
# Voice Tests
# ===================================================================
class TestVoiceFeatureExtractor:
    """Tests for VoiceFeatureExtractor."""

    def _make_tone(self, freq: float = 220.0, duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate a synthetic sine-wave tone."""
        time_array = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq * time_array)).astype(np.float32)

    def test_pitch_extraction_returns_valid(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=220, duration=1.0)
        pitch = extractor.extract_pitch_features(audio, 16000)
        assert "mean_f0" in pitch
        assert "std_f0" in pitch
        assert "f0_range" in pitch
        assert "voiced_fraction" in pitch
        assert pitch["voiced_fraction"] >= 0.0

    def test_energy_extraction(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=300, duration=1.0)
        energy = extractor.extract_energy_features(audio, 16000)
        assert energy["rms_mean"] > 0
        assert isinstance(energy["loudness_db"], float)

    def test_temporal_features_with_silence(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        # 1s tone + 1s silence + 1s tone
        tone = self._make_tone(freq=220, duration=1.0)
        silence = np.zeros(16000, dtype=np.float32)
        audio = np.concatenate([tone, silence, tone])
        temporal = extractor.extract_temporal_features(audio, 16000)
        assert temporal["pause_count"] >= 1
        assert temporal["total_pause_duration"] > 0.0

    def test_mfcc_shape(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=200, duration=1.0)
        mfcc = extractor.extract_mfcc(audio, 16000, n_mfcc=40)
        assert len(mfcc["mfcc_mean"]) == 40
        assert len(mfcc["mfcc_delta_mean"]) == 40
        assert len(mfcc["mfcc_delta2_mean"]) == 40

    def test_spectral_features(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=440, duration=1.0)
        spectral = extractor.extract_spectral_features(audio, 16000)
        assert spectral["spectral_centroid_mean"] > 0
        assert len(spectral["chroma_mean"]) == 12

    def test_jitter_non_negative(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=220, duration=2.0)
        jitter = extractor.extract_jitter(audio, 16000)
        assert jitter >= 0.0

    def test_shimmer_non_negative(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=220, duration=2.0)
        shimmer = extractor.extract_shimmer(audio, 16000)
        assert shimmer >= 0.0

    def test_hnr_returns_float(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=220, duration=2.0)
        hnr = extractor.extract_hnr(audio, 16000)
        assert isinstance(hnr, float)

    def test_extract_all_keys(self):
        from app.voice.feature_extractor import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        audio = self._make_tone(freq=220, duration=2.0)
        features = extractor.extract_all(audio, 16000)
        assert "pitch" in features
        assert "energy" in features
        assert "temporal" in features
        assert "mfcc" in features
        assert "spectral" in features
        assert "jitter" in features
        assert "shimmer" in features
        assert "hnr" in features


class TestVoiceEmotionMapper:
    """Tests for VoiceEmotionMapper."""

    def test_anxiety_pattern(self):
        from app.voice.feature_extractor import VoiceEmotionMapper
        mapper = VoiceEmotionMapper()
        features = {
            "pitch": {"mean_f0": 250, "std_f0": 50, "f0_range": 100, "voiced_fraction": 0.8},
            "energy": {"rms_mean": 0.08, "rms_std": 0.02, "rms_slope": 0.0, "loudness_db": -20},
            "temporal": {"speech_rate_estimate": 6.0, "pause_count": 0, "total_pause_duration": 0},
            "spectral": {"zero_crossing_rate_mean": 0.1},
            "jitter": 0.04, "shimmer": 0.05, "hnr": 10,
        }
        vector = mapper.map_to_emotion(features)
        # Fear dim (index 5) should be active
        assert vector._vector[5] > 0.3

    def test_depression_pattern(self):
        from app.voice.feature_extractor import VoiceEmotionMapper
        mapper = VoiceEmotionMapper()
        features = {
            "pitch": {"mean_f0": 100, "std_f0": 5, "f0_range": 20, "voiced_fraction": 0.3},
            "energy": {"rms_mean": 0.01, "rms_std": 0.002, "rms_slope": -0.001, "loudness_db": -40},
            "temporal": {"speech_rate_estimate": 1.5, "pause_count": 5, "total_pause_duration": 5.0},
            "spectral": {"zero_crossing_rate_mean": 0.02},
            "jitter": 0.01, "shimmer": 0.02, "hnr": 5,
        }
        vector = mapper.map_to_emotion(features)
        # Sadness dim (index 7) should be active, valence negative
        assert vector._vector[7] > 0.3
        assert vector.to_pad()[0] < 0

    def test_clinical_flags_depression(self):
        from app.voice.feature_extractor import VoiceEmotionMapper
        mapper = VoiceEmotionMapper()
        features = {
            "pitch": {"mean_f0": 100, "std_f0": 5},
            "energy": {"rms_mean": 0.01, "rms_std": 0.002},
            "temporal": {"speech_rate_estimate": 1.5},
        }
        flags = mapper.detect_clinical_flags(features)
        assert any("DEPRESSION" in flag for flag in flags)

    def test_clinical_flags_flat_affect(self):
        from app.voice.feature_extractor import VoiceEmotionMapper
        mapper = VoiceEmotionMapper()
        features = {
            "pitch": {"std_f0": 3.0},
            "energy": {"rms_std": 0.001},
        }
        flags = mapper.detect_clinical_flags(features)
        assert any("FLAT_AFFECT" in flag for flag in flags)


# ===================================================================
# Face Analyzer Tests (using mock landmarks)
# ===================================================================
class MockLandmark:
    """Mimics a MediaPipe NormalizedLandmark."""
    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_neutral_face(count: int = 468) -> list[MockLandmark]:
    """Generate a set of mock landmarks at neutral positions."""
    return [MockLandmark(x=0.5, y=0.5, z=0.0) for _ in range(count)]


class TestFaceAnalyzer:
    """Tests for FaceAnalyzer."""

    def test_extract_action_units_returns_10(self):
        from app.camera.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        landmarks = _make_neutral_face()
        aus = analyzer.extract_action_units(landmarks)
        assert len(aus) == 10
        assert all(0.0 <= val <= 1.0 for val in aus.values())

    def test_smile_none_on_neutral(self):
        from app.camera.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        landmarks = _make_neutral_face()
        smile = analyzer.detect_genuine_vs_masked_smile(landmarks)
        # Neutral face → unlikely to register strong AU12
        assert smile.smile_type in ("none", "polite", "genuine")
        assert 0.0 <= smile.authenticity_score <= 1.0

    def test_micro_expression_empty_buffer(self):
        from app.camera.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        buffer: deque[dict[str, float]] = deque(maxlen=20)
        micro = analyzer.detect_micro_expressions(buffer)
        assert micro == []

    def test_aus_to_emotion_scores(self):
        from app.camera.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        aus = {"AU1": 0.1, "AU4": 0.2, "AU6": 0.8, "AU12": 0.9,
               "AU15": 0.0, "AU17": 0.0, "AU20": 0.0, "AU23": 0.0,
               "AU24": 0.0, "AU28": 0.0}
        scores = analyzer._aus_to_emotion_scores(aus)
        assert scores["happy"] > scores["sad"]


# ===================================================================
# Gaze Analyzer Tests
# ===================================================================
class TestGazeAnalyzer:
    """Tests for GazeAnalyzer."""

    def test_ear_open_eye(self):
        from app.camera.gaze_analyzer import GazeAnalyzer
        analyzer = GazeAnalyzer()
        # Open eye: wide vertical, normal horizontal
        eye = [
            (0.3, 0.5),  # outer corner
            (0.35, 0.45),  # upper 1
            (0.4, 0.44),  # upper 2
            (0.45, 0.5),  # inner corner
            (0.4, 0.56),  # lower 2
            (0.35, 0.55),  # lower 1
        ]
        ear = analyzer.calculate_ear(eye)
        assert ear > 0.2  # open eye

    def test_ear_closed_eye(self):
        from app.camera.gaze_analyzer import GazeAnalyzer
        analyzer = GazeAnalyzer()
        # Closed eye: minimal vertical gap
        eye = [
            (0.3, 0.5),
            (0.35, 0.50),
            (0.4, 0.50),
            (0.45, 0.5),
            (0.4, 0.51),
            (0.35, 0.51),
        ]
        ear = analyzer.calculate_ear(eye)
        assert ear < 0.2  # closed eye

    def test_blink_detection(self):
        from app.camera.gaze_analyzer import GazeAnalyzer
        analyzer = GazeAnalyzer()
        # Simulate: 3 frames below threshold, then open
        for _ in range(3):
            analyzer.update_blink(0.15)
        blinked = analyzer.update_blink(0.30)
        assert blinked is True

    def test_eye_contact_starts_zero(self):
        from app.camera.gaze_analyzer import GazeAnalyzer
        analyzer = GazeAnalyzer()
        assert analyzer.eye_contact_percentage() == 0.0


# ===================================================================
# Posture Analyzer Tests
# ===================================================================
class TestPostureAnalyzer:
    """Tests for PostureAnalyzer."""

    def _make_pose(self, slouch: bool = False) -> list[MockLandmark]:
        """Create 33 mock pose landmarks."""
        landmarks = [MockLandmark() for _ in range(33)]
        # Shoulders
        landmarks[11] = MockLandmark(x=0.4, y=0.4, z=0.0)  # left shoulder
        landmarks[12] = MockLandmark(x=0.6, y=0.4, z=0.0)  # right shoulder
        # Hips
        landmarks[23] = MockLandmark(x=0.4, y=0.7, z=0.0)
        landmarks[24] = MockLandmark(x=0.6, y=0.7, z=0.0)
        # Nose
        landmarks[0] = MockLandmark(x=0.5, y=0.2, z=0.0)
        # Wrists (open posture)
        landmarks[15] = MockLandmark(x=0.2, y=0.5, z=0.0)
        landmarks[16] = MockLandmark(x=0.8, y=0.5, z=0.0)

        if slouch:
            # Move shoulders forward (increase z) to create slouch
            landmarks[11] = MockLandmark(x=0.4, y=0.5, z=0.3)
            landmarks[12] = MockLandmark(x=0.6, y=0.5, z=0.3)

        return landmarks

    def test_level_shoulders(self):
        from app.camera.posture_analyzer import PostureAnalyzer
        analyzer = PostureAnalyzer()
        pose = self._make_pose()
        slope = analyzer.calculate_shoulder_slope(pose)
        assert abs(slope) < 5  # nearly level

    def test_body_openness_open(self):
        from app.camera.posture_analyzer import PostureAnalyzer
        analyzer = PostureAnalyzer()
        pose = self._make_pose()
        openness = analyzer.calculate_body_openness(pose)
        assert openness > 0.5  # wrists spread wide

    def test_body_openness_crossed(self):
        from app.camera.posture_analyzer import PostureAnalyzer
        analyzer = PostureAnalyzer()
        pose = self._make_pose()
        # Cross wrists
        pose[15] = MockLandmark(x=0.6, y=0.5)  # left wrist on right side
        pose[16] = MockLandmark(x=0.4, y=0.5)  # right wrist on left side
        openness = analyzer.calculate_body_openness(pose)
        assert openness == 0.0

    def test_full_analysis_returns_result(self):
        from app.camera.posture_analyzer import PostureAnalyzer
        analyzer = PostureAnalyzer()
        pose = self._make_pose()
        result = analyzer.analyze(pose)
        assert hasattr(result, "shoulder_slope")
        assert hasattr(result, "slouch")
        assert hasattr(result, "body_openness")


# ===================================================================
# Hand Gesture Analyzer Tests
# ===================================================================
class TestHandGestureAnalyzer:
    """Tests for HandGestureAnalyzer."""

    def _make_open_hand(self) -> list[MockLandmark]:
        """21 landmarks for an open hand."""
        landmarks = [MockLandmark(x=0.5, y=0.5) for _ in range(21)]
        # Wrist
        landmarks[0] = MockLandmark(x=0.5, y=0.7)
        # Fingertips far from wrist
        landmarks[8] = MockLandmark(x=0.5, y=0.2)   # index tip
        landmarks[12] = MockLandmark(x=0.5, y=0.2)  # middle tip
        landmarks[16] = MockLandmark(x=0.5, y=0.2)  # ring tip
        landmarks[20] = MockLandmark(x=0.5, y=0.2)  # pinky tip
        # MCP joints between wrist and tips
        landmarks[5] = MockLandmark(x=0.5, y=0.45)
        landmarks[9] = MockLandmark(x=0.5, y=0.45)
        landmarks[13] = MockLandmark(x=0.5, y=0.45)
        landmarks[17] = MockLandmark(x=0.5, y=0.45)
        return landmarks

    def _make_clenched_fist(self) -> list[MockLandmark]:
        """21 landmarks for a clenched fist."""
        landmarks = [MockLandmark(x=0.5, y=0.5) for _ in range(21)]
        landmarks[0] = MockLandmark(x=0.5, y=0.7)  # wrist
        # Tips closer to wrist than MCPs
        landmarks[8] = MockLandmark(x=0.5, y=0.6)   # tips near wrist
        landmarks[12] = MockLandmark(x=0.5, y=0.6)
        landmarks[16] = MockLandmark(x=0.5, y=0.6)
        landmarks[20] = MockLandmark(x=0.5, y=0.6)
        # MCPs further from wrist
        landmarks[5] = MockLandmark(x=0.5, y=0.45)
        landmarks[9] = MockLandmark(x=0.5, y=0.45)
        landmarks[13] = MockLandmark(x=0.5, y=0.45)
        landmarks[17] = MockLandmark(x=0.5, y=0.45)
        return landmarks

    def test_open_hand_not_clenched(self):
        from app.camera.gesture_analyzer import HandGestureAnalyzer
        analyzer = HandGestureAnalyzer()
        hand = self._make_open_hand()
        assert analyzer.detect_hand_tension(hand) is False

    def test_clenched_fist_detected(self):
        from app.camera.gesture_analyzer import HandGestureAnalyzer
        analyzer = HandGestureAnalyzer()
        hand = self._make_clenched_fist()
        assert analyzer.detect_hand_tension(hand) is True

    def test_fidget_score_still_hands(self):
        from app.camera.gesture_analyzer import HandGestureAnalyzer
        analyzer = HandGestureAnalyzer()
        hand = self._make_open_hand()
        # Same position across 10 frames
        sequence = [hand] * 10
        score = analyzer.calculate_fidget_score(sequence)
        assert score < 0.1

    def test_self_touch_no_bbox(self):
        from app.camera.gesture_analyzer import HandGestureAnalyzer
        analyzer = HandGestureAnalyzer()
        hand = self._make_open_hand()
        result = analyzer.detect_self_touching(hand, face_bbox=None)
        assert result.is_touching_face is False


# ===================================================================
# Cross-Modal Conflict Detector Tests
# ===================================================================
class TestCameraConflictDetector:
    """Tests for CameraConflictDetector."""

    def test_no_conflict_when_all_none(self):
        from app.camera.pipeline import CameraConflictDetector
        detector = CameraConflictDetector()
        report = detector.detect_conflict(None, None, None)
        assert report.has_conflict is False

    def test_emotional_blunting_all_low(self):
        from app.camera.face_analyzer import FaceAnalysisResult, SmileAnalysis
        from app.camera.pipeline import CameraConflictDetector
        detector = CameraConflictDetector()
        face = FaceAnalysisResult(
            action_units={}, smile_analysis=SmileAnalysis("none", 0.0, 0.9),
            micro_expressions=[], deepface_emotions={},
            combined_emotions={"neutral": 0.9, "happy": 0.02, "sad": 0.02},
            processing_time_ms=10,
        )
        voice = {"energy": {"rms_mean": 0.01}, "pitch": {"std_f0": 3.0}}
        text = {"model_confidence": 0.3, "raw_scores": {"neutral": 0.9}}
        report = detector.detect_conflict(face, voice, text)
        assert report.has_conflict is True
        assert report.conflict_type == "emotional_blunting"
