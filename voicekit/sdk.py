"""VoiceKit SDK — the main entry point for AI integration.

Provides a clean, unified API that combines all voice detection
and analysis capabilities into a single interface.

Usage:
    from voicekit import VoiceKit

    kit = VoiceKit()

    # Full analysis
    result = kit.analyze("audio.wav")
    print(result.vad.segments)
    print(result.speaker.embedding)
    print(result.characteristics.emotion)
    print(result.quality.overall_score)

    # Individual modules
    vad_result = kit.detect_voice_activity("audio.wav")
    speaker_result = kit.identify_speaker("audio.wav")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np
import structlog

from voicekit.analyzers.characteristics import CharacteristicsAnalyzer
from voicekit.analyzers.quality import QualityAnalyzer
from voicekit.core.audio import AudioEngine
from voicekit.core.types import (
    AnalysisResult,
    AudioConfig,
    AudioSegment,
    CharacteristicsResult,
    QualityResult,
    SpeakerResult,
    VADResult,
)
from voicekit.detectors.speaker import SpeakerDetector
from voicekit.detectors.vad import VoiceActivityDetector

logger = structlog.get_logger(__name__)


class VoiceKit:
    """Production-grade voice detection and analysis SDK.

    This is the primary integration point for AI companies. It provides
    a unified API for voice activity detection, speaker identification,
    voice characteristics analysis, and audio quality assessment.

    Args:
        config: Audio processing configuration. Uses sensible defaults if not provided.

    Example:
        >>> kit = VoiceKit()
        >>> result = kit.analyze("meeting_recording.wav")
        >>> for segment in result.vad.segments:
        ...     print(f"Speech: {segment.start_seconds:.1f}s - {segment.end_seconds:.1f}s")
        >>> print(f"Speaker emotion: {result.characteristics.emotion.value}")
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        self._engine = AudioEngine(self.config)
        self._vad = VoiceActivityDetector(self.config, self._engine)
        self._speaker = SpeakerDetector(self.config, self._engine)
        self._characteristics = CharacteristicsAnalyzer(self.config, self._engine)
        self._quality = QualityAnalyzer(self.config, self._engine)

    def analyze(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        include_vad: bool = True,
        include_speaker: bool = True,
        include_characteristics: bool = True,
        include_quality: bool = True,
    ) -> AnalysisResult:
        """Run full voice analysis pipeline on audio.

        This is the primary method for comprehensive voice analysis.
        Each module can be individually toggled for performance.

        Args:
            source: Audio file path, WAV bytes, or numpy array.
            include_vad: Run voice activity detection.
            include_speaker: Run speaker embedding/identification.
            include_characteristics: Run voice characteristics analysis.
            include_quality: Run audio quality assessment.

        Returns:
            AnalysisResult with all requested analysis results.
        """
        start_time = time.monotonic()
        logger.info("analysis.start", source_type=type(source).__name__)

        segment = self._engine.load(source)

        vad_result = self._vad.detect(segment) if include_vad else VADResult()
        speaker_result = (
            self._speaker.detect(segment) if include_speaker else SpeakerResult()
        )
        char_result = (
            self._characteristics.analyze(segment)
            if include_characteristics
            else CharacteristicsResult()
        )
        quality_result = (
            self._quality.analyze(segment) if include_quality else QualityResult()
        )

        elapsed = time.monotonic() - start_time
        logger.info("analysis.complete", duration_seconds=round(elapsed, 3))

        return AnalysisResult(
            audio=segment,
            vad=vad_result,
            speaker=speaker_result,
            characteristics=char_result,
            quality=quality_result,
            processing_time_seconds=round(elapsed, 4),
        )

    def detect_voice_activity(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> VADResult:
        """Run only voice activity detection.

        Args:
            source: Audio file path, WAV bytes, or numpy array.

        Returns:
            VADResult with detected speech segments.
        """
        segment = self._engine.load(source)
        return self._vad.detect(segment)

    def identify_speaker(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> SpeakerResult:
        """Run only speaker identification.

        Args:
            source: Audio file path, WAV bytes, or numpy array.

        Returns:
            SpeakerResult with speaker embedding and identification.
        """
        segment = self._engine.load(source)
        return self._speaker.detect(segment)

    def analyze_characteristics(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> CharacteristicsResult:
        """Run only voice characteristics analysis.

        Args:
            source: Audio file path, WAV bytes, or numpy array.

        Returns:
            CharacteristicsResult with vocal property analysis.
        """
        segment = self._engine.load(source)
        return self._characteristics.analyze(segment)

    def assess_quality(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> QualityResult:
        """Run only audio quality assessment.

        Args:
            source: Audio file path, WAV bytes, or numpy array.

        Returns:
            QualityResult with quality metrics and issues.
        """
        segment = self._engine.load(source)
        return self._quality.analyze(segment)

    def enroll_speaker(
        self, speaker_id: str, source: Union[str, Path, bytes, np.ndarray]
    ) -> np.ndarray:
        """Enroll a speaker for identification.

        Args:
            speaker_id: Unique label for the speaker.
            source: Audio sample of the speaker.

        Returns:
            Speaker embedding vector.
        """
        segment = self._engine.load(source)
        return self._speaker.enroll(speaker_id, segment)

    def verify_speaker(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        claimed_id: str,
    ) -> tuple[bool, float]:
        """Verify a speaker's claimed identity.

        Args:
            source: Audio sample to verify.
            claimed_id: Speaker ID to verify against.

        Returns:
            Tuple of (is_match, similarity_score).
        """
        segment = self._engine.load(source)
        return self._speaker.verify(segment, claimed_id)

    def compare_speakers(
        self,
        source_a: Union[str, Path, bytes, np.ndarray],
        source_b: Union[str, Path, bytes, np.ndarray],
    ) -> float:
        """Compare two audio samples for speaker similarity.

        Args:
            source_a: First audio sample.
            source_b: Second audio sample.

        Returns:
            Cosine similarity score (-1.0 to 1.0).
        """
        seg_a = self._engine.load(source_a)
        seg_b = self._engine.load(source_b)
        return self._speaker.compare(seg_a, seg_b)

    def process_stream_frame(
        self, frame: np.ndarray, state: dict | None = None
    ) -> tuple[bool, float, dict]:
        """Process a single frame for real-time streaming VAD.

        Args:
            frame: Audio frame as numpy array.
            state: State dict from previous call, or None.

        Returns:
            Tuple of (is_speech, confidence, updated_state).
        """
        return self._vad.detect_streaming(frame, state)
