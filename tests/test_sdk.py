"""Tests for the main VoiceKit SDK interface."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit import VoiceKit
from voicekit.core.types import (
    AnalysisResult,
    AudioConfig,
    CharacteristicsResult,
    Emotion,
    QualityResult,
    SpeakerResult,
    VADResult,
)


class TestVoiceKit:
    """Test the main SDK interface."""

    def test_analyze_numpy_array(self) -> None:
        kit = VoiceKit()
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = (0.7 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        result = kit.analyze(signal)

        assert isinstance(result, AnalysisResult)
        assert isinstance(result.vad, VADResult)
        assert isinstance(result.speaker, SpeakerResult)
        assert isinstance(result.characteristics, CharacteristicsResult)
        assert isinstance(result.quality, QualityResult)
        assert result.processing_time_seconds > 0

    def test_analyze_wav_bytes(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        result = kit.analyze(wav_bytes)

        assert isinstance(result, AnalysisResult)
        assert result.audio.duration_seconds > 0.9
        assert len(result.speaker.embedding) == 192

    def test_analyze_selective_modules(self) -> None:
        kit = VoiceKit()
        sr = 16000
        signal = np.sin(np.linspace(0, 2 * np.pi * 200, sr)).astype(np.float32)

        result = kit.analyze(
            signal,
            include_vad=True,
            include_speaker=False,
            include_characteristics=False,
            include_quality=False,
        )

        assert result.vad.total_duration_seconds > 0
        # Non-included modules should have default values
        assert len(result.speaker.embedding) == 0
        assert result.characteristics.emotion == Emotion.NEUTRAL
        assert result.quality.overall_score == 0.0

    def test_detect_voice_activity(self, speech_wav_bytes: bytes) -> None:
        kit = VoiceKit()
        result = kit.detect_voice_activity(speech_wav_bytes)

        assert isinstance(result, VADResult)
        assert result.total_duration_seconds > 0

    def test_identify_speaker(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        result = kit.identify_speaker(wav_bytes)

        assert isinstance(result, SpeakerResult)
        assert len(result.embedding) == 192

    def test_analyze_characteristics(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        result = kit.analyze_characteristics(wav_bytes)

        assert isinstance(result, CharacteristicsResult)
        assert isinstance(result.emotion, Emotion)

    def test_assess_quality(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        result = kit.assess_quality(wav_bytes)

        assert isinstance(result, QualityResult)
        assert 0 <= result.overall_score <= 1.0

    def test_enroll_and_verify_speaker(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        kit.enroll_speaker("alice", wav_bytes)

        is_match, score = kit.verify_speaker(wav_bytes, "alice")
        assert is_match is True
        assert score > 0.7

    def test_compare_speakers(self, wav_bytes: bytes) -> None:
        kit = VoiceKit()
        score = kit.compare_speakers(wav_bytes, wav_bytes)
        assert score > 0.99

    def test_custom_config(self) -> None:
        config = AudioConfig(
            sample_rate=8000,
            frame_duration_ms=20,
            embedding_dim=128,
        )
        kit = VoiceKit(config)

        sr = 8000
        signal = np.sin(np.linspace(0, 2 * np.pi * 200, sr)).astype(np.float32)
        result = kit.analyze(signal)

        assert result.audio.sample_rate == 8000
        assert len(result.speaker.embedding) == 128

    def test_streaming_vad(self) -> None:
        kit = VoiceKit()
        frame = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 0.03, 480)
        ).astype(np.float32)

        is_speech, confidence, state = kit.process_stream_frame(frame)
        assert isinstance(is_speech, bool)
        assert 0 <= confidence <= 1.0
        assert state is not None

        # Process more frames
        for _ in range(5):
            is_speech, confidence, state = kit.process_stream_frame(frame, state)

        assert len(state["energy_history"]) == 6
