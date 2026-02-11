"""Tests for the core AudioEngine."""

from __future__ import annotations

import numpy as np
import pytest

from voicekit.core.audio import AudioEngine
from voicekit.core.types import AudioConfig, AudioSegment


class TestAudioEngineLoad:
    """Test audio loading from various sources."""

    def test_load_numpy_array(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        samples = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        segment = engine.load(samples)

        assert segment.sample_rate == 16000
        assert segment.channels == 1
        assert abs(segment.duration_seconds - 1.0) < 0.01
        assert len(segment.samples) == 16000

    def test_load_wav_bytes(self, config: AudioConfig, wav_bytes: bytes) -> None:
        engine = AudioEngine(config)
        segment = engine.load(wav_bytes)

        assert segment.sample_rate == 16000
        assert segment.channels == 1
        assert segment.duration_seconds > 0.9

    def test_load_stereo_numpy(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        # Stereo signal
        left = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        right = np.cos(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        stereo = np.column_stack([left, right])
        segment = engine.load(stereo)

        assert segment.channels == 1  # Should be mono
        assert len(segment.samples) == 16000

    def test_load_nonexistent_file(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        with pytest.raises(FileNotFoundError):
            engine.load("/nonexistent/audio.wav")

    def test_load_normalizes_loud_audio(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        loud = np.ones(16000, dtype=np.float32) * 5.0
        segment = engine.load(loud)
        max_val = max(abs(s) for s in segment.samples)
        assert max_val <= 1.0


class TestAudioEngineFeatures:
    """Test feature extraction methods."""

    def test_extract_frames(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        engine = AudioEngine(config)
        frames = engine.extract_frames(tone_segment)

        expected_frame_size = int(16000 * 30 / 1000)  # 480 samples
        assert len(frames) > 0
        assert all(len(f) == expected_frame_size for f in frames)

    def test_compute_energy_silence(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        silence = np.zeros(480, dtype=np.float32)
        energy = engine.compute_energy(silence)
        assert energy == -100.0

    def test_compute_energy_tone(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        t = np.linspace(0, 0.03, 480, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        energy = engine.compute_energy(tone)
        assert energy > -20

    def test_compute_zcr(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        t = np.linspace(0, 0.03, 480, endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        zcr = engine.compute_zcr(tone)
        # 440Hz in 30ms should have about 13 zero crossings
        assert 0.01 < zcr < 0.5

    def test_compute_pitch(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        sr = 16000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        pitch = engine.compute_pitch(tone, sr)
        # Should be close to 200Hz
        assert 180 < pitch < 220

    def test_extract_mfcc(
        self, config: AudioConfig, tone_segment: AudioSegment
    ) -> None:
        engine = AudioEngine(config)
        mfccs = engine.extract_mfcc(tone_segment, n_mfcc=13)
        assert mfccs.shape[0] == 13
        assert mfccs.shape[1] > 0

    def test_resample(self, config: AudioConfig) -> None:
        engine = AudioEngine(config)
        original = np.sin(np.linspace(0, 2 * np.pi, 8000)).astype(np.float32)
        resampled = engine._resample(original, 8000, 16000)
        assert len(resampled) == 16000
