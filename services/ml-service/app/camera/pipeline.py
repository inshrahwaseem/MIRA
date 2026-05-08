"""
MIRA Camera Pipeline — Orchestrator + Cross-Modal Conflict Detector.

Runs FaceMesh, Pose, and Hands in parallel threads, then integrates
all camera sub-analyses into a single CameraAnalysisResult.

Also houses the cross-modal conflict detector that compares
face + voice + text for discrepancies (masked emotions).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import cv2
import numpy as np

from app.camera.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from app.camera.gaze_analyzer import EyeData, GazeAnalyzer
from app.camera.gesture_analyzer import HandAnalysisResult, HandGestureAnalyzer
from app.camera.posture_analyzer import PostureAnalysisResult, PostureAnalyzer
from app.core.emotion_taxonomy import EmotionVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraAnalysisResult:
    """Aggregated camera analysis from all sub-analyzers."""

    face: Optional[FaceAnalysisResult]
    gaze: Optional[EyeData]
    posture: Optional[PostureAnalysisResult]
    hands: Optional[HandAnalysisResult]
    emotion_vector: EmotionVector
    clinical_flags: list[str]
    processing_time_ms: float


@dataclass
class ConflictReport:
    """Cross-modal conflict detection result."""

    has_conflict: bool
    conflict_type: str = ""
    masked_emotion: str = ""
    genuine_emotion: str = ""
    confidence: float = 0.0
    explanation: str = ""
    llm_instruction_adjustment: str = ""


# ---------------------------------------------------------------------------
# CameraManager — async frame streaming
# ---------------------------------------------------------------------------
class CameraManager:
    """Async context manager for camera frame streaming at 10fps."""

    def __init__(self, camera_id: int = 0, target_fps: int = 10) -> None:
        self._camera_id = camera_id
        self._target_fps = target_fps
        self._capture: Optional[cv2.VideoCapture] = None

    async def __aenter__(self) -> CameraManager:
        """Open camera capture."""
        self._capture = cv2.VideoCapture(self._camera_id)
        if not self._capture.isOpened():
            logger.warning(f"Camera {self._camera_id} unavailable — graceful degradation.")
            self._capture = None
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release camera capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    async def stream_frames(self) -> AsyncGenerator[np.ndarray, None]:
        """Yield BGR frames at the target FPS."""
        if self._capture is None:
            return

        frame_interval = 1.0 / self._target_fps
        while True:
            start_time = time.perf_counter()
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

            # Throttle to target FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


# ---------------------------------------------------------------------------
# CameraPipeline — parallel analysis with threading
# ---------------------------------------------------------------------------
class CameraPipeline:
    """
    Orchestrates FaceMesh + Pose + Hands analysis in parallel threads.

    PRIVACY: Frames are processed in-memory only. Never written to disk.
    """

    def __init__(self) -> None:
        self._face_analyzer = FaceAnalyzer()
        self._gaze_analyzer = GazeAnalyzer()
        self._posture_analyzer = PostureAnalyzer()
        self._hand_analyzer = HandGestureAnalyzer()
        self._pose_processor: Any = None
        self._hands_processor: Any = None

    def _ensure_mediapipe(self) -> None:
        """Lazy-load MediaPipe Pose and Hands."""
        if self._pose_processor is None:
            import mediapipe as mp
            self._pose_processor = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands_processor = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def analyze_frame(
        self,
        frame: np.ndarray,
        hand_sequence: Optional[list[list[Any]]] = None,
        current_time_sec: float = 0.0,
    ) -> CameraAnalysisResult:
        """
        Run all camera sub-analyzers on a single frame.

        Uses threading to parallelise FaceMesh, Pose, and Hands
        with a 100ms join timeout.
        """
        start_time = time.perf_counter()
        self._ensure_mediapipe()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Shared results
        face_result: Optional[FaceAnalysisResult] = None
        gaze_result: Optional[EyeData] = None
        posture_result: Optional[PostureAnalysisResult] = None
        hand_result: Optional[HandAnalysisResult] = None

        # --- Thread: Face + Gaze ---
        def run_face() -> None:
            nonlocal face_result, gaze_result
            try:
                face_result = self._face_analyzer.analyze_frame(frame)
                # Gaze uses the same FaceMesh landmarks
                import mediapipe as mp
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, refine_landmarks=True,
                    max_num_faces=1, min_detection_confidence=0.5,
                )
                mesh_results = face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    gaze_result = self._gaze_analyzer.analyze_from_mediapipe(landmarks)
                face_mesh.close()
            except Exception as face_error:
                logger.warning(f"Face analysis thread failed: {face_error}")

        # --- Thread: Pose ---
        def run_pose() -> None:
            nonlocal posture_result
            try:
                pose_results = self._pose_processor.process(rgb_frame)
                if pose_results.pose_landmarks:
                    posture_result = self._posture_analyzer.analyze(
                        pose_results.pose_landmarks.landmark
                    )
            except Exception as pose_error:
                logger.warning(f"Pose analysis thread failed: {pose_error}")

        # --- Thread: Hands ---
        def run_hands() -> None:
            nonlocal hand_result
            try:
                hands_results = self._hands_processor.process(rgb_frame)
                if hands_results.multi_hand_landmarks:
                    hand_lms = hands_results.multi_hand_landmarks[0].landmark
                    # Face bounding box for self-touch detection
                    face_bbox = None
                    if face_result is not None:
                        # Approximate bbox from known face landmark positions
                        face_bbox = (0.25, 0.15, 0.75, 0.70)  # default face region
                    hand_result = self._hand_analyzer.analyze(
                        hand_landmarks=hand_lms,
                        hand_sequence=hand_sequence or [],
                        face_bbox=face_bbox,
                        current_time_sec=current_time_sec,
                    )
            except Exception as hand_error:
                logger.warning(f"Hand analysis thread failed: {hand_error}")

        # Run in parallel
        face_thread = threading.Thread(target=run_face, daemon=True)
        pose_thread = threading.Thread(target=run_pose, daemon=True)
        hand_thread = threading.Thread(target=run_hands, daemon=True)

        face_thread.start()
        pose_thread.start()
        hand_thread.start()

        face_thread.join(timeout=0.1)
        pose_thread.join(timeout=0.1)
        hand_thread.join(timeout=0.1)

        # Build emotion vector from camera data
        emotion_vector = self._build_emotion_vector(face_result, posture_result, hand_result)

        # Collect clinical flags
        clinical_flags: list[str] = []
        if posture_result:
            clinical_flags.extend(posture_result.emotion_signals)
        if hand_result:
            clinical_flags.extend(hand_result.emotion_signals)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return CameraAnalysisResult(
            face=face_result,
            gaze=gaze_result,
            posture=posture_result,
            hands=hand_result,
            emotion_vector=emotion_vector,
            clinical_flags=clinical_flags,
            processing_time_ms=elapsed_ms,
        )

    def _build_emotion_vector(
        self,
        face: Optional[FaceAnalysisResult],
        posture: Optional[PostureAnalysisResult],
        hands: Optional[HandAnalysisResult],
    ) -> EmotionVector:
        """Combine face/posture/hand signals into a 64-dim EmotionVector."""
        vector = np.zeros(64, dtype=np.float32)

        if face and face.combined_emotions:
            emotions = face.combined_emotions
            # Map DeepFace labels to vector dims
            vector[3] = emotions.get("happy", 0.0)     # Joy
            vector[5] = emotions.get("fear", 0.0)       # Fear
            vector[6] = emotions.get("surprise", 0.0)   # Surprise
            vector[7] = emotions.get("sad", 0.0)        # Sadness
            vector[8] = emotions.get("disgust", 0.0)    # Disgust
            vector[9] = emotions.get("angry", 0.0)      # Anger

            # PAD from face
            vector[0] = emotions.get("happy", 0) - emotions.get("sad", 0) - emotions.get("angry", 0) * 0.5
            vector[1] = max(emotions.get("fear", 0), emotions.get("angry", 0), emotions.get("surprise", 0))
            vector[2] = emotions.get("angry", 0) * 0.7 - emotions.get("fear", 0) * 0.5

        if posture:
            # Slouching depresses valence
            if posture.slouch.severity >= 2:
                vector[0] -= 0.15
            # Open body increases dominance
            vector[2] += posture.body_openness * 0.1

        if hands:
            # Fidgeting increases arousal
            vector[1] = max(vector[1], hands.fidget_score * 0.5)
            # Clenched fist increases anger dim
            if hands.is_clenched:
                vector[9] += 0.2

        # Clamp PAD
        vector[0] = float(np.clip(vector[0], -1.0, 1.0))
        vector[1] = float(np.clip(vector[1], 0.0, 1.0))
        vector[2] = float(np.clip(vector[2], 0.0, 1.0))

        return EmotionVector(vector)


# ---------------------------------------------------------------------------
# CrossModalConflictDetector (camera-aware version)
# ---------------------------------------------------------------------------
class CameraConflictDetector:
    """
    Detects cross-modal conflicts using detailed camera sub-signals.

    4 specific clinical patterns:
      1. Masked Positivity
      2. Suppressed Anxiety
      3. Controlled Anger
      4. Emotional Blunting
    """

    def detect_conflict(
        self,
        face_result: Optional[FaceAnalysisResult],
        voice_features: Optional[dict[str, Any]],
        text_scores: Optional[dict[str, Any]],
        posture: Optional[PostureAnalysisResult] = None,
        hands: Optional[HandAnalysisResult] = None,
    ) -> ConflictReport:
        """Run all 4 pattern detectors and return first match."""

        # Pattern 1: Masked Positivity
        if (
            face_result is not None
            and face_result.smile_analysis.smile_type in ("genuine", "polite")
            and voice_features is not None
            and voice_features.get("energy", {}).get("rms_mean", 1.0) < 0.025
            and text_scores is not None
            and text_scores.get("crisis_risk_score", 0) > 0.3
        ):
            return ConflictReport(
                has_conflict=True,
                conflict_type="masked_positivity",
                masked_emotion="depression",
                genuine_emotion="sadness",
                confidence=0.80,
                explanation=(
                    "Smiling face but voice energy is very low and text expresses distress. "
                    "Likely masking genuine sadness with a social smile."
                ),
                llm_instruction_adjustment=(
                    "Gently acknowledge the contrast. Do NOT say 'you seem fine'. "
                    "Use: 'I notice what you're saying feels heavy, even though you're smiling — I'm here for you.'"
                ),
            )

        # Pattern 2: Suppressed Anxiety
        if (
            text_scores is not None
            and text_scores.get("masking_probability", 0) > 0.6
            and voice_features is not None
            and voice_features.get("pitch", {}).get("mean_f0", 0) > 200
            and hands is not None
            and hands.fidget_score > 0.5
        ):
            return ConflictReport(
                has_conflict=True,
                conflict_type="suppressed_anxiety",
                masked_emotion="anxiety",
                genuine_emotion="apprehension",
                confidence=0.75,
                explanation=(
                    "Text says 'I'm fine' but voice pitch is elevated and hands are fidgeting. "
                    "Anxiety is being consciously suppressed."
                ),
                llm_instruction_adjustment=(
                    "Address possible anxiety without labelling. "
                    "Use: 'Your body seems a bit restless — how are you really feeling right now?'"
                ),
            )

        # Pattern 3: Controlled Anger
        if (
            text_scores is not None
            and text_scores.get("raw_scores", {}).get("anger", 0) > 0.5
            and face_result is not None
            and face_result.combined_emotions.get("neutral", 0) > 0.5
            and posture is not None
            and posture.body_openness < 0.3
        ):
            return ConflictReport(
                has_conflict=True,
                conflict_type="controlled_anger",
                masked_emotion="anger",
                genuine_emotion="anger",
                confidence=0.70,
                explanation="Text expresses anger but face appears neutral — anger is being controlled.",
                llm_instruction_adjustment="Validate the frustration. Offer anger-management reframing techniques.",
            )

        # Pattern 4: Emotional Blunting
        all_low = True
        if face_result and face_result.combined_emotions:
            max_face_score = max(face_result.combined_emotions.values(), default=0)
            if max_face_score > 0.3:
                all_low = False
        if voice_features:
            if voice_features.get("energy", {}).get("rms_mean", 0) > 0.03:
                all_low = False
            if voice_features.get("pitch", {}).get("std_f0", 0) > 15:
                all_low = False
        if text_scores:
            if text_scores.get("model_confidence", 0) > 0.5:
                raw = text_scores.get("raw_scores", {})
                if raw and max(raw.values(), default=0) > 0.5:
                    all_low = False

        if all_low and (face_result is not None or voice_features is not None):
            return ConflictReport(
                has_conflict=True,
                conflict_type="emotional_blunting",
                masked_emotion="numb",
                genuine_emotion="numb",
                confidence=0.65,
                explanation="Very low emotional signal across all modalities — possible emotional blunting or dissociation.",
                llm_instruction_adjustment=(
                    "This is clinically significant. Gently ask: "
                    "'Sometimes feeling nothing can be harder than feeling something. Has it been like this for a while?'"
                ),
            )

        return ConflictReport(has_conflict=False)
