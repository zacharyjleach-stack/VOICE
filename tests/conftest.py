"""Shared test fixtures for VoiceKit tests."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from voicekit.core.types import AudioConfig, AudioSegment


@pytest.fixture
def config() -> AudioConfig:
    """Default test configuration."""
    return AudioConfig(sample_rate=16000, frame_duration_ms=30)


@pytest.fixture
def silence_segment() -> AudioSegment:
    """1 second of silence at 16kHz."""
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    return AudioSegment(
        samples=samples.tolist(),
        sample_rate=sr,
        channels=1,
        duration_seconds=1.0,
    )


@pytest.fixture
def tone_segment() -> AudioSegment:
    """1 second of a 440Hz sine tone at 16kHz."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioSegment(
        samples=samples.tolist(),
        sample_rate=sr,
        channels=1,
        duration_seconds=duration,
    )


@pytest.fixture
def speech_like_segment() -> AudioSegment:
    """2 seconds of speech-like audio (modulated tone with pauses).

    Simulates speech with:
    - A modulated tone (varying pitch around 150Hz typical male speech)
    - Amplitude modulation to simulate syllables
    - A silence gap in the middle to test VAD segmentation
    """
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Fundamental frequency with slight variation
    f0 = 150 + 20 * np.sin(2 * np.pi * 3 * t)

    # Generate voice-like signal
    phase = np.cumsum(2 * np.pi * f0 / sr)
    signal = np.sin(phase)

    # Add harmonics for more voice-like quality
    signal += 0.5 * np.sin(2 * phase)
    signal += 0.25 * np.sin(3 * phase)

    # Amplitude modulation (syllable-like envelope at ~4Hz)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal *= envelope

    # Insert silence gap (0.8s to 1.2s)
    gap_start = int(0.8 * sr)
    gap_end = int(1.2 * sr)
    signal[gap_start:gap_end] = 0.0

    # Normalize
    signal = (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)

    return AudioSegment(
        samples=signal.tolist(),
        sample_rate=sr,
        channels=1,
        duration_seconds=duration,
    )


@pytest.fixture
def noisy_segment() -> AudioSegment:
    """1 second of white noise at 16kHz."""
    sr = 16000
    rng = np.random.RandomState(42)
    samples = (rng.randn(sr) * 0.3).astype(np.float32)
    return AudioSegment(
        samples=samples.tolist(),
        sample_rate=sr,
        channels=1,
        duration_seconds=1.0,
    )


@pytest.fixture
def wav_bytes() -> bytes:
    """Generate valid WAV bytes (1 second of 440Hz tone)."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.8 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def speech_wav_bytes() -> bytes:
    """Generate WAV bytes with speech-like content."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    f0 = 150 + 20 * np.sin(2 * np.pi * 3 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase)

    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal *= envelope
    signal = signal / np.max(np.abs(signal)) * 0.7

    int_samples = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()
