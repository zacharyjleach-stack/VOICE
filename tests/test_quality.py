"""Tests for audio quality assessment."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit.analyzers.quality import QualityAnalyzer
from voicekit.core.types import AudioConfig, AudioSegment


class TestQualityAnalyzer:
    """Test audio quality assessment."""

    def test_good_quality_tone(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(tone_segment)

        assert result.overall_score > 0.3
        assert result.is_usable is True
        assert result.clipping_ratio < 0.01

    def test_silence_low_quality(
        self, config: AudioConfig, silence_segment: AudioSegment
    ) -> None:
        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(silence_segment)

        assert result.silence_ratio > 0.9
        assert result.is_usable is False
        assert any("silence" in issue.lower() for issue in result.issues)

    def test_clipped_audio_detected(self, config: AudioConfig) -> None:
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # Clip the signal
        signal = np.clip(signal * 3, -1.0, 1.0)
        segment = AudioSegment(
            samples=signal.tolist(),
            sample_rate=sr,
            duration_seconds=1.0,
        )

        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(segment)

        assert result.clipping_ratio > 0.01
        assert any("clip" in issue.lower() for issue in result.issues)

    def test_low_signal_detected(self, config: AudioConfig) -> None:
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 0.005).astype(np.float32)
        segment = AudioSegment(
            samples=signal.tolist(),
            sample_rate=sr,
            duration_seconds=1.0,
        )

        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(segment)

        assert any("low signal" in issue.lower() for issue in result.issues)

    def test_empty_audio(self, config: AudioConfig) -> None:
        segment = AudioSegment(samples=[], sample_rate=16000)
        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(segment)

        assert result.overall_score == 0.0
        assert result.is_usable is False

    def test_noisy_audio_low_snr(
        self, config: AudioConfig, noisy_segment: AudioSegment
    ) -> None:
        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(noisy_segment)

        # Pure noise should have low SNR
        assert result.snr_db < 20

    def test_quality_score_bounded(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        analyzer = QualityAnalyzer(config)
        result = analyzer.analyze(tone_segment)

        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.silence_ratio <= 1.0
        assert 0.0 <= result.clipping_ratio <= 1.0
