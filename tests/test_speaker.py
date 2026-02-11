"""Tests for speaker detection and identification."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit.core.types import AudioConfig, AudioSegment
from voicekit.detectors.speaker import SpeakerDetector


class TestSpeakerDetector:
    """Test speaker embedding and identification."""

    def test_embedding_shape(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        result = detector.detect(tone_segment)

        assert len(result.embedding) == config.embedding_dim
        assert result.speaker_count == 1

    def test_embedding_normalized(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        embedding = detector.extract_embedding(tone_segment)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # L2 normalized

    def test_same_audio_similar_embeddings(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        emb1 = detector.extract_embedding(tone_segment)
        emb2 = detector.extract_embedding(tone_segment)

        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        # Same audio should produce identical embeddings
        assert similarity > 0.99

    def test_different_audio_different_embeddings(
        self,
        config: AudioConfig,
        tone_segment: AudioSegment,
        noisy_segment: AudioSegment,
    ) -> None:
        detector = SpeakerDetector(config)
        emb_tone = detector.extract_embedding(tone_segment)
        emb_noise = detector.extract_embedding(noisy_segment)

        similarity = np.dot(emb_tone, emb_noise) / (
            np.linalg.norm(emb_tone) * np.linalg.norm(emb_noise)
        )
        # Very different audio should have lower similarity
        assert similarity < 0.95

    def test_enroll_and_identify(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        detector.enroll("speaker_a", tone_segment)

        result = detector.detect(tone_segment)
        # Should identify the enrolled speaker
        assert result.speaker_id == "speaker_a"
        assert result.confidence > 0.7

    def test_verify_speaker(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        detector.enroll("test_speaker", tone_segment)

        is_match, similarity = detector.verify(tone_segment, "test_speaker")
        assert is_match is True
        assert similarity > 0.7

    def test_verify_unknown_speaker(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        is_match, similarity = detector.verify(tone_segment, "unknown")
        assert is_match is False
        assert similarity == 0.0

    def test_compare_same_audio(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        detector = SpeakerDetector(config)
        score = detector.compare(tone_segment, tone_segment)
        assert score > 0.99

    def test_empty_audio_embedding(self, config: AudioConfig) -> None:
        detector = SpeakerDetector(config)
        empty = AudioSegment(
            samples=[0.0] * 160,  # Tiny segment
            sample_rate=16000,
            duration_seconds=0.01,
        )
        result = detector.detect(empty)
        assert len(result.embedding) == config.embedding_dim
