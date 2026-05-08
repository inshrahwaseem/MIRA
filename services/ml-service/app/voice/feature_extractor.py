"""
MIRA Voice Feature Extractor.

Extracts clinically-validated vocal biomarkers from raw audio using
librosa, scipy, and openai-whisper (local, free, no API key).

Feature groups:
  - Prosodic: F0 pitch, energy/RMS, temporal/pauses
  - Spectral: MFCC (120 features), centroid, bandwidth, rolloff, ZCR, chroma
  - Voice quality: jitter, shimmer, HNR (stress biomarkers)
  - Transcription: Whisper local STT with word timestamps

All features are mapped to a Plutchik EmotionVector via rule-based
clinical thresholds (no ML training needed for voice → emotion).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from app.core.emotion_taxonomy import EmotionVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionResult:
    """Output of Whisper local STT."""

    text: str
    language: str
    confidence: float
    word_timestamps: list[dict[str, Any]]


@dataclass
class VoiceAnalysisResult:
    """Complete voice analysis output."""

    emotion_vector: EmotionVector
    features: dict[str, Any]
    transcription: Optional[TranscriptionResult]
    estimated_arousal: float
    clinical_flags: list[str]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# SpeechTranscriber — Whisper local (FREE, no API key)
# ---------------------------------------------------------------------------
class SpeechTranscriber:
    """Local Whisper model for speech-to-text."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model: Any = None

    @cached_property
    def model(self) -> Any:
        """Lazy-load Whisper model on first call."""
        import whisper

        logger.info(f"Loading Whisper model: {self._model_size}")
        return whisper.load_model(self._model_size)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio array to text with word-level timestamps.

        Args:
            audio: mono float32 waveform
            sample_rate: audio sample rate (resampled to 16kHz internally)
        """
        # Whisper expects 16kHz float32
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        try:
            result = self.model.transcribe(
                audio.astype(np.float32),
                word_timestamps=True,
                fp16=False,
            )

            # Extract word-level timestamps from segments
            word_timestamps: list[dict[str, Any]] = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_timestamps.append({
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                    })

            return TranscriptionResult(
                text=result.get("text", "").strip(),
                language=result.get("language", "en"),
                confidence=1.0 - result.get("no_speech_prob", 0.0) if "no_speech_prob" in result else 0.9,
                word_timestamps=word_timestamps,
            )
        except Exception as whisper_error:
            logger.error(f"Whisper transcription failed: {whisper_error}")
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                word_timestamps=[],
            )


# ---------------------------------------------------------------------------
# VoiceFeatureExtractor — all clinically-validated features
# ---------------------------------------------------------------------------
class VoiceFeatureExtractor:
    """
    Extracts prosodic, spectral, and voice-quality features from audio.

    All methods accept raw audio (np.ndarray) + sample rate (int).
    """

    # ── Prosodic: Pitch (F0) ──

    def extract_pitch_features(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Extract fundamental frequency (F0) features using librosa pyin.

        Clinical significance:
          - High F0 + high std → anxiety / excitement
          - Low F0 + narrow range → depression / sadness
        """
        f0_array, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )

        voiced_f0 = f0_array[~np.isnan(f0_array)]
        if len(voiced_f0) < 3:
            return {
                "mean_f0": 0.0, "std_f0": 0.0, "f0_range": 0.0,
                "f0_slope": 0.0, "voiced_fraction": 0.0,
            }

        # Linear trend slope (falling = sadness, rising = anxiety)
        time_indices = np.arange(len(voiced_f0))
        slope_coeffs = np.polyfit(time_indices, voiced_f0, deg=1)

        total_frames = len(f0_array)
        voiced_count = np.sum(~np.isnan(f0_array))

        return {
            "mean_f0": float(np.mean(voiced_f0)),
            "std_f0": float(np.std(voiced_f0)),
            "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
            "f0_slope": float(slope_coeffs[0]),
            "voiced_fraction": float(voiced_count / total_frames) if total_frames > 0 else 0.0,
        }

    # ── Prosodic: Energy / RMS ──

    def extract_energy_features(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Extract RMS energy features.

        Clinical significance:
          - Consistently low energy → depression / fatigue biomarker
          - Declining RMS slope → fatigue within session
        """
        rms = librosa.feature.rms(y=audio)[0]
        if len(rms) < 2:
            return {"rms_mean": 0.0, "rms_std": 0.0, "rms_slope": 0.0, "loudness_db": -80.0}

        time_indices = np.arange(len(rms))
        slope_coeffs = np.polyfit(time_indices, rms, deg=1)
        loudness = librosa.power_to_db(np.array([np.mean(rms ** 2)]))[0]

        return {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_slope": float(slope_coeffs[0]),
            "loudness_db": float(loudness),
        }

    # ── Prosodic: Temporal / Pauses ──

    def extract_temporal_features(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """
        Extract speech rate and pause features.

        Clinical significance:
          - Slow speech + long pauses = clinically validated depression marker
          - Fast speech + few pauses = mania / anxiety
        """
        rms = librosa.feature.rms(y=audio)[0]
        hop_length = 512
        frame_duration = hop_length / sample_rate

        # Silence threshold: frames below 15% of mean RMS
        silence_threshold = np.mean(rms) * 0.15
        is_silence = rms < silence_threshold

        # Detect pauses (contiguous silence > 300ms)
        min_pause_frames = int(0.3 / frame_duration)
        pause_count = 0
        total_pause_duration = 0.0
        longest_pause = 0.0
        current_silence_length = 0

        for frame_is_silent in is_silence:
            if frame_is_silent:
                current_silence_length += 1
            else:
                if current_silence_length >= min_pause_frames:
                    pause_duration = current_silence_length * frame_duration
                    pause_count += 1
                    total_pause_duration += pause_duration
                    longest_pause = max(longest_pause, pause_duration)
                current_silence_length = 0

        # Check trailing silence
        if current_silence_length >= min_pause_frames:
            pause_duration = current_silence_length * frame_duration
            pause_count += 1
            total_pause_duration += pause_duration
            longest_pause = max(longest_pause, pause_duration)

        # Speech rate estimate: count energy peaks as pseudo-syllables
        speech_frames = rms[~is_silence]
        total_duration = len(audio) / sample_rate
        speech_duration = max(total_duration - total_pause_duration, 0.1)

        # Peak counting for syllable estimation
        if len(speech_frames) > 10:
            peaks, _ = find_peaks(speech_frames, distance=int(0.15 / frame_duration))
            syllable_count = len(peaks)
        else:
            syllable_count = 0

        speech_rate = syllable_count / total_duration if total_duration > 0 else 0.0
        articulation_rate = syllable_count / speech_duration if speech_duration > 0 else 0.0

        return {
            "speech_rate_estimate": float(speech_rate),
            "pause_count": pause_count,
            "total_pause_duration": float(total_pause_duration),
            "longest_pause": float(longest_pause),
            "articulation_rate": float(articulation_rate),
        }

    # ── Spectral: MFCC (120 features total) ──

    def extract_mfcc(self, audio: np.ndarray, sample_rate: int, n_mfcc: int = 40) -> dict[str, list[float]]:
        """
        Extract 40 MFCCs + 40 deltas + 40 delta-deltas = 120 features.
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        return {
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "mfcc_delta_mean": np.mean(mfcc_delta, axis=1).tolist(),
            "mfcc_delta2_mean": np.mean(mfcc_delta2, axis=1).tolist(),
        }

    # ── Spectral: Centroid, Bandwidth, Rolloff, ZCR, Chroma ──

    def extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """Extract standard spectral descriptors."""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
        }

    # ── Voice Quality: Jitter ──

    def extract_jitter(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Pitch period perturbation (relative jitter).

        High jitter → vocal stress, emotion dysregulation.
        """
        f0_array, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        voiced_f0 = f0_array[~np.isnan(f0_array)]
        if len(voiced_f0) < 3:
            return 0.0

        # Convert F0 to periods
        periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(period_diffs) / np.mean(periods))
        return jitter

    # ── Voice Quality: Shimmer ──

    def extract_shimmer(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Amplitude perturbation (relative shimmer).

        High shimmer → vocal effort, stress.
        """
        rms = librosa.feature.rms(y=audio)[0]
        if len(rms) < 3:
            return 0.0

        amplitude_diffs = np.abs(np.diff(rms))
        shimmer = float(np.mean(amplitude_diffs) / (np.mean(rms) + 1e-10))
        return shimmer

    # ── Voice Quality: HNR ──

    def extract_hnr(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Harmonics-to-Noise Ratio via autocorrelation.

        Low HNR → breathy / strained voice → stress / sadness marker.
        """
        # Autocorrelation-based HNR estimation
        frame_length = int(0.04 * sample_rate)  # 40ms frames
        hop_length = int(0.01 * sample_rate)     # 10ms hop
        hnr_values: list[float] = []

        for start in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start : start + frame_length]
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Find peak in valid pitch range (80-500 Hz)
            min_lag = int(sample_rate / 500)
            max_lag = int(sample_rate / 80)
            if max_lag >= len(autocorr):
                max_lag = len(autocorr) - 1
            if min_lag >= max_lag:
                continue

            search_region = autocorr[min_lag:max_lag]
            if len(search_region) == 0:
                continue

            peak_val = float(np.max(search_region))
            zero_lag = float(autocorr[0])
            if zero_lag <= 0:
                continue

            ratio = peak_val / zero_lag
            if 0 < ratio < 1:
                hnr_db = 10 * np.log10(ratio / (1 - ratio + 1e-10))
                hnr_values.append(float(hnr_db))

        return float(np.mean(hnr_values)) if hnr_values else 0.0

    # ── Full extraction ──

    def extract_all(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """Extract every feature group and return a flat dict."""
        pitch = self.extract_pitch_features(audio, sample_rate)
        energy = self.extract_energy_features(audio, sample_rate)
        temporal = self.extract_temporal_features(audio, sample_rate)
        mfcc = self.extract_mfcc(audio, sample_rate)
        spectral = self.extract_spectral_features(audio, sample_rate)
        jitter = self.extract_jitter(audio, sample_rate)
        shimmer = self.extract_shimmer(audio, sample_rate)
        hnr = self.extract_hnr(audio, sample_rate)

        return {
            "pitch": pitch,
            "energy": energy,
            "temporal": temporal,
            "mfcc": mfcc,
            "spectral": spectral,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
        }


# ---------------------------------------------------------------------------
# VoiceEmotionMapper — rule-based clinical thresholds
# ---------------------------------------------------------------------------

# Dimension indices in 64-dim vector (from emotion_taxonomy)
_DIM = {
    "Joy": 3, "Trust": 4, "Fear": 5, "Surprise": 6,
    "Sadness": 7, "Disgust": 8, "Anger": 9, "Anticipation": 10,
}


class VoiceEmotionMapper:
    """
    Maps vocal features to a Plutchik EmotionVector using
    clinical threshold rules — no ML training required.
    """

    def map_to_emotion(self, features: dict[str, Any]) -> EmotionVector:
        """
        Rule-based mapping of voice features → 64-dim EmotionVector.

        Patterns (from psychoacoustics literature):
          High f0 + high energy + fast speech + high ZCR → anxiety/panic
          Low f0 + low energy + slow speech + long pauses → sadness/depression
          High energy + fast speech + high jitter → anger/fury
          Stable f0 + moderate energy + normal pace → calm/content
          All-low features → emotional blunting
        """
        pitch = features.get("pitch", {})
        energy = features.get("energy", {})
        temporal = features.get("temporal", {})
        spectral = features.get("spectral", {})
        jitter = features.get("jitter", 0.0)
        shimmer = features.get("shimmer", 0.0)
        hnr = features.get("hnr", 0.0)

        mean_f0 = pitch.get("mean_f0", 150.0)
        std_f0 = pitch.get("std_f0", 20.0)
        f0_slope = pitch.get("f0_slope", 0.0)
        rms_mean = energy.get("rms_mean", 0.05)
        speech_rate = temporal.get("speech_rate_estimate", 3.0)
        pause_duration = temporal.get("total_pause_duration", 0.0)
        zcr = spectral.get("zero_crossing_rate_mean", 0.05)

        vector = np.zeros(64, dtype=np.float32)

        # --- Anxiety / Panic ---
        anxiety_score = 0.0
        if mean_f0 > 200:
            anxiety_score += 0.3
        if std_f0 > 40:
            anxiety_score += 0.2
        if speech_rate > 5.0:
            anxiety_score += 0.2
        if zcr > 0.08:
            anxiety_score += 0.15
        if jitter > 0.03:
            anxiety_score += 0.15
        anxiety_score = min(anxiety_score, 1.0)

        # --- Sadness / Depression ---
        sadness_score = 0.0
        if mean_f0 < 120:
            sadness_score += 0.25
        if pitch.get("f0_range", 50) < 30:
            sadness_score += 0.2
        if rms_mean < 0.02:
            sadness_score += 0.2
        if speech_rate < 2.0:
            sadness_score += 0.2
        if pause_duration > 3.0:
            sadness_score += 0.15
        sadness_score = min(sadness_score, 1.0)

        # --- Anger / Fury ---
        anger_score = 0.0
        if rms_mean > 0.1:
            anger_score += 0.3
        if speech_rate > 4.5:
            anger_score += 0.2
        if jitter > 0.04:
            anger_score += 0.25
        if zcr > 0.1:
            anger_score += 0.15
        if shimmer > 0.1:
            anger_score += 0.1
        anger_score = min(anger_score, 1.0)

        # --- Calm / Content ---
        calm_score = 0.0
        if 130 < mean_f0 < 200:
            calm_score += 0.3
        if std_f0 < 25:
            calm_score += 0.2
        if 0.03 < rms_mean < 0.08:
            calm_score += 0.25
        if 2.5 < speech_rate < 4.5:
            calm_score += 0.25
        calm_score = min(calm_score, 1.0)

        # --- Joy (high energy + positive slope + high voiced fraction) ---
        joy_score = 0.0
        if mean_f0 > 180:
            joy_score += 0.2
        if f0_slope > 0.1:
            joy_score += 0.2
        if rms_mean > 0.06:
            joy_score += 0.2
        if pitch.get("voiced_fraction", 0.5) > 0.7:
            joy_score += 0.2
        if hnr > 15:
            joy_score += 0.2
        joy_score = min(joy_score, 1.0)

        # Fill primary emotion dims
        vector[_DIM["Fear"]] = anxiety_score
        vector[_DIM["Sadness"]] = sadness_score
        vector[_DIM["Anger"]] = anger_score
        vector[_DIM["Joy"]] = joy_score
        vector[_DIM["Trust"]] = calm_score * 0.5
        vector[_DIM["Anticipation"]] = max(anxiety_score * 0.3, joy_score * 0.3)

        # PAD values (dims 0-2)
        vector[0] = (joy_score + calm_score) - (sadness_score + anxiety_score * 0.5)  # valence
        vector[1] = max(anxiety_score, anger_score, joy_score * 0.7)                  # arousal
        vector[2] = max(anger_score * 0.8, calm_score * 0.6) - sadness_score * 0.4    # dominance

        # Clamp PAD
        vector[0] = float(np.clip(vector[0], -1.0, 1.0))
        vector[1] = float(np.clip(vector[1], 0.0, 1.0))
        vector[2] = float(np.clip(vector[2], 0.0, 1.0))

        return EmotionVector(vector)

    def detect_clinical_flags(self, features: dict[str, Any]) -> list[str]:
        """Flag clinically significant voice patterns."""
        flags: list[str] = []
        pitch = features.get("pitch", {})
        energy = features.get("energy", {})
        temporal = features.get("temporal", {})

        # Depression markers
        if (
            pitch.get("mean_f0", 999) < 110
            and energy.get("rms_mean", 999) < 0.02
            and temporal.get("speech_rate_estimate", 999) < 2.0
        ):
            flags.append("DEPRESSION_VOCAL_PATTERN: low_pitch + low_energy + slow_speech")

        # Long pauses
        if temporal.get("longest_pause", 0) > 5.0:
            flags.append("EXTENDED_SILENCE: pause > 5 seconds — potential dissociation")

        # Flat affect (very low variability across all features)
        if (
            pitch.get("std_f0", 999) < 8
            and energy.get("rms_std", 999) < 0.005
        ):
            flags.append("FLAT_AFFECT: minimal pitch and energy variation")

        # High stress
        if features.get("jitter", 0) > 0.05 and features.get("shimmer", 0) > 0.15:
            flags.append("HIGH_VOCAL_STRESS: elevated jitter + shimmer")

        return flags


# ---------------------------------------------------------------------------
# Full voice analysis pipeline
# ---------------------------------------------------------------------------
class VoiceAnalyzer:
    """Orchestrates transcription + feature extraction + emotion mapping."""

    def __init__(self) -> None:
        self._transcriber = SpeechTranscriber(model_size="base")
        self._extractor = VoiceFeatureExtractor()
        self._mapper = VoiceEmotionMapper()

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcribe: bool = True,
    ) -> VoiceAnalysisResult:
        """Full voice analysis pipeline."""
        start_time = time.perf_counter()

        # Feature extraction
        features = self._extractor.extract_all(audio, sample_rate)

        # Emotion mapping
        emotion_vector = self._mapper.map_to_emotion(features)

        # Clinical flags
        clinical_flags = self._mapper.detect_clinical_flags(features)

        # Transcription (optional, heavier compute)
        transcription: Optional[TranscriptionResult] = None
        if transcribe:
            transcription = self._transcriber.transcribe(audio, sample_rate)

        # Estimated arousal from features
        _, arousal, _ = emotion_vector.to_pad()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return VoiceAnalysisResult(
            emotion_vector=emotion_vector,
            features=features,
            transcription=transcription,
            estimated_arousal=arousal,
            clinical_flags=clinical_flags,
            processing_time_ms=elapsed_ms,
        )
