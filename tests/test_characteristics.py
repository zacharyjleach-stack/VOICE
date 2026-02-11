"""Tests for voice characteristics analysis."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit.analyzers.characteristics import CharacteristicsAnalyzer
from voicekit.core.types import AudioConfig, AudioSegment, Emotion, Gender


class TestCharacteristicsAnalyzer:
    """Test voice characteristics analysis."""

    def test_analyze_returns_all_fields(
        self, config: AudioConfig, speech_like_segment: AudioSegment
    ) -> None:
        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(speech_like_segment)

        assert isinstance(result.emotion, Emotion)
        assert len(result.emotion_scores) > 0
        assert isinstance(result.gender, Gender)
        assert 0 <= result.gender_confidence <= 1.0
        assert result.pitch_mean_hz >= 0
        assert result.pitch_std_hz >= 0
        assert result.speech_rate_sps >= 0

    def test_emotion_scores_sum_to_one(
        self, config: AudioConfig, speech_like_segment: AudioSegment
    ) -> None:
        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(speech_like_segment)

        total = sum(result.emotion_scores.values())
        assert abs(total - 1.0) < 0.01

    def test_all_emotions_have_scores(
        self, config: AudioConfig, speech_like_segment: AudioSegment
    ) -> None:
        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(speech_like_segment)

        for emotion in Emotion:
            assert emotion.value in result.emotion_scores

    def test_low_pitch_classified_male(self, config: AudioConfig) -> None:
        """A 120Hz tone should be classified as male."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 120 * t).astype(np.float32)
        signal *= 0.7
        segment = AudioSegment(
            samples=signal.tolist(),
            sample_rate=sr,
            duration_seconds=duration,
        )

        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(segment)
        assert result.gender == Gender.MALE

    def test_high_pitch_classified_female(self, config: AudioConfig) -> None:
        """A 250Hz tone should be classified as female."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 250 * t).astype(np.float32)
        signal *= 0.7
        segment = AudioSegment(
            samples=signal.tolist(),
            sample_rate=sr,
            duration_seconds=duration,
        )

        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(segment)
        assert result.gender == Gender.FEMALE

    def test_silence_analysis(
        self, config: AudioConfig, silence_segment: AudioSegment
    ) -> None:
        analyzer = CharacteristicsAnalyzer(config)
        result = analyzer.analyze(silence_segment)

        assert result.pitch_mean_hz == 0.0
        assert result.gender == Gender.UNKNOWN
