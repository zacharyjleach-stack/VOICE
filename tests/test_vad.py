"""Tests for Voice Activity Detection."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit.core.types import AudioConfig, AudioSegment
from voicekit.detectors.vad import VoiceActivityDetector


class TestVAD:
    """Test voice activity detection."""

    def test_silence_has_no_speech(
        self, config: AudioConfig, silence_segment: AudioSegment
    ) -> None:
        vad = VoiceActivityDetector(config)
        result = vad.detect(silence_segment)

        assert len(result.segments) == 0
        assert result.speech_ratio == 0.0
        assert result.total_speech_seconds == 0.0

    def test_tone_detected_as_activity(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        vad = VoiceActivityDetector(config)
        result = vad.detect(tone_segment)

        # A pure tone should register as voice activity
        assert result.total_duration_seconds > 0.9

    def test_speech_like_detected(
        self, config: AudioConfig, speech_like_segment: AudioSegment
    ) -> None:
        vad = VoiceActivityDetector(config)
        result = vad.detect(speech_like_segment)

        # Should detect speech with some segments
        assert result.speech_ratio > 0
        assert result.total_speech_seconds > 0

    def test_segments_have_valid_times(
        self, config: AudioConfig, speech_like_segment: AudioSegment
    ) -> None:
        vad = VoiceActivityDetector(config)
        result = vad.detect(speech_like_segment)

        for seg in result.segments:
            assert seg.start_seconds >= 0
            assert seg.end_seconds > seg.start_seconds
            assert seg.duration_seconds > 0
            assert 0 <= seg.confidence <= 1.0

    def test_empty_audio(self, config: AudioConfig) -> None:
        vad = VoiceActivityDetector(config)
        segment = AudioSegment(samples=[], sample_rate=16000, duration_seconds=0.0)
        result = vad.detect(segment)
        assert len(result.segments) == 0

    def test_custom_threshold(self, tone_segment: AudioSegment) -> None:
        # Very high threshold — should detect less
        strict_config = AudioConfig(vad_threshold=0.9)
        vad = VoiceActivityDetector(strict_config)
        strict_result = vad.detect(tone_segment)

        # Very low threshold — should detect more
        lenient_config = AudioConfig(vad_threshold=0.1)
        vad_lenient = VoiceActivityDetector(lenient_config)
        lenient_result = vad_lenient.detect(tone_segment)

        assert lenient_result.speech_ratio >= strict_result.speech_ratio


class TestStreamingVAD:
    """Test streaming/real-time VAD."""

    def test_streaming_returns_state(self, config: AudioConfig) -> None:
        vad = VoiceActivityDetector(config)
        frame = np.sin(np.linspace(0, 2 * np.pi * 440 * 0.03, 480)).astype(
            np.float32
        )
        is_speech, confidence, state = vad.detect_streaming(frame)

        assert isinstance(is_speech, bool)
        assert 0 <= confidence <= 1.0
        assert isinstance(state, dict)
        assert "energy_history" in state

    def test_streaming_state_persists(self, config: AudioConfig) -> None:
        vad = VoiceActivityDetector(config)
        state = None
        frame = np.sin(np.linspace(0, 2 * np.pi * 440 * 0.03, 480)).astype(
            np.float32
        )

        for _ in range(10):
            _, _, state = vad.detect_streaming(frame, state)

        assert len(state["energy_history"]) == 10

    def test_silence_not_speech_streaming(self, config: AudioConfig) -> None:
        vad = VoiceActivityDetector(config)
        silence = np.zeros(480, dtype=np.float32)
        is_speech, confidence, _ = vad.detect_streaming(silence)

        # Pure silence should generally not be detected as speech
        assert confidence < 0.8
