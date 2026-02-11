"""Voice Activity Detection (VAD) engine — Silero Neural Network backend.

Uses the Silero VAD model (via torch.hub) for state-of-the-art speech
detection. Replaces the legacy energy/ZCR heuristic approach with a
compact neural network that runs locally on CPU, CUDA, or MPS.

The Silero model is downloaded automatically on first run (~1MB).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from voicekit.core.types import AudioConfig, AudioSegment, VADResult, VoiceSegment


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SileroVoiceDetector:
    """Neural VAD powered by Silero VAD v5.

    Loads the Silero model once on init and provides both single-frame
    and full-segment speech detection.

    Args:
        config: Audio processing configuration (sample_rate should be 16000).
        device: Compute device ("cpu", "cuda", "mps", or None for auto-detect).
        threshold: Speech probability threshold (0.0-1.0). Default 0.5.
    """

    SUPPORTED_SAMPLE_RATES = (8000, 16000)

    def __init__(
        self,
        config: AudioConfig | None = None,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.config = config or AudioConfig()
        self.device = device or _detect_device()
        self.threshold = threshold

        # Load Silero VAD model
        try:
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Silero VAD model. Ensure you have internet "
                f"access for the initial download. Error: {e}"
            ) from e

        self._model = self._model.to(self.device)
        self._model.eval()

        # Extract utility functions from Silero's bundle
        (
            self._get_speech_timestamps,
            _,  # save_audio
            _,  # read_audio
            _,  # VADIterator
            _,  # collect_chunks
        ) = self._utils

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Classify a single audio frame as speech or non-speech.

        Args:
            audio_frame: 1-D numpy array of audio samples (16kHz, mono).
                         Recommended frame sizes: 512, 1024, or 1536 samples.

        Returns:
            True if the frame contains speech.
        """
        tensor = torch.from_numpy(audio_frame).float().to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prob = self._model(tensor, self.config.sample_rate).item()

        return prob >= self.threshold

    def speech_probability(self, audio_frame: np.ndarray) -> float:
        """Return the raw speech probability for a single frame.

        Args:
            audio_frame: 1-D numpy array of audio samples (16kHz, mono).

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        tensor = torch.from_numpy(audio_frame).float().to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prob = self._model(tensor, self.config.sample_rate).item()

        return float(prob)

    def detect(self, segment: AudioSegment) -> VADResult:
        """Run full voice activity detection on an audio segment.

        Uses Silero's get_speech_timestamps for optimal segment-level
        detection with built-in smoothing and merging.

        Args:
            segment: Audio segment to analyze.

        Returns:
            VADResult with detected speech segments and statistics.
        """
        samples = np.array(segment.samples, dtype=np.float32)
        if len(samples) == 0:
            return VADResult(total_duration_seconds=segment.duration_seconds)

        audio_tensor = torch.from_numpy(samples).float().to(self.device)

        # Reset model state for clean segment processing
        self._model.reset_states()

        try:
            speech_timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=self.config.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.max_silence_duration_ms,
                return_seconds=False,
            )
        except Exception:
            # Fallback: process frame-by-frame if timestamps API fails
            return self._detect_frame_by_frame(segment)

        sr = self.config.sample_rate
        voice_segments: list[VoiceSegment] = []

        for ts in speech_timestamps:
            start_s = ts["start"] / sr
            end_s = ts["end"] / sr

            # Compute confidence from the frames within this segment
            chunk = samples[ts["start"] : ts["end"]]
            confidence = self._estimate_segment_confidence(chunk)

            voice_segments.append(
                VoiceSegment(
                    start_seconds=round(start_s, 4),
                    end_seconds=round(end_s, 4),
                    confidence=confidence,
                )
            )

        total_speech = sum(s.duration_seconds for s in voice_segments)
        speech_ratio = (
            total_speech / segment.duration_seconds
            if segment.duration_seconds > 0
            else 0.0
        )

        return VADResult(
            segments=voice_segments,
            speech_ratio=round(speech_ratio, 4),
            total_speech_seconds=round(total_speech, 4),
            total_duration_seconds=segment.duration_seconds,
        )

    def detect_streaming(
        self, frame: np.ndarray, state: dict | None = None
    ) -> tuple[bool, float, dict]:
        """Process a single frame for real-time streaming VAD.

        Maintains internal state across calls for smooth detection.

        Args:
            frame: Single audio frame as numpy array (16kHz, mono).
            state: Persistent state dict from previous call (or None).

        Returns:
            Tuple of (is_speech, confidence, updated_state).
        """
        if state is None:
            self._model.reset_states()
            state = {"frame_count": 0, "speech_active": False}

        prob = self.speech_probability(frame)
        is_speech = prob >= self.threshold

        state["frame_count"] += 1
        state["speech_active"] = is_speech

        return is_speech, prob, state

    def _detect_frame_by_frame(self, segment: AudioSegment) -> VADResult:
        """Fallback frame-by-frame detection."""
        samples = np.array(segment.samples, dtype=np.float32)
        sr = self.config.sample_rate
        frame_size = 512  # Silero-recommended frame size for 16kHz
        frame_dur = frame_size / sr

        self._model.reset_states()

        labels: list[tuple[float, bool, float]] = []
        for start in range(0, len(samples) - frame_size, frame_size):
            frame = samples[start : start + frame_size]
            prob = self.speech_probability(frame)
            time_s = start / sr
            labels.append((time_s, prob >= self.threshold, prob))

        # Merge consecutive speech frames into segments
        voice_segments: list[VoiceSegment] = []
        seg_start: float | None = None
        seg_probs: list[float] = []

        for time_s, is_speech, prob in labels:
            if is_speech and seg_start is None:
                seg_start = time_s
                seg_probs = [prob]
            elif is_speech and seg_start is not None:
                seg_probs.append(prob)
            elif not is_speech and seg_start is not None:
                end_s = time_s + frame_dur
                dur = end_s - seg_start
                if dur >= self.config.min_speech_duration_ms / 1000:
                    voice_segments.append(
                        VoiceSegment(
                            start_seconds=round(seg_start, 4),
                            end_seconds=round(end_s, 4),
                            confidence=round(float(np.mean(seg_probs)), 4),
                        )
                    )
                seg_start = None
                seg_probs = []

        if seg_start is not None:
            end_s = len(samples) / sr
            dur = end_s - seg_start
            if dur >= self.config.min_speech_duration_ms / 1000:
                voice_segments.append(
                    VoiceSegment(
                        start_seconds=round(seg_start, 4),
                        end_seconds=round(end_s, 4),
                        confidence=round(float(np.mean(seg_probs)), 4),
                    )
                )

        total_speech = sum(s.duration_seconds for s in voice_segments)
        speech_ratio = (
            total_speech / segment.duration_seconds
            if segment.duration_seconds > 0
            else 0.0
        )

        return VADResult(
            segments=voice_segments,
            speech_ratio=round(speech_ratio, 4),
            total_speech_seconds=round(total_speech, 4),
            total_duration_seconds=segment.duration_seconds,
        )

    def _estimate_segment_confidence(self, chunk: np.ndarray) -> float:
        """Estimate confidence for a speech segment by sampling frames."""
        frame_size = 512
        if len(chunk) < frame_size:
            return self.speech_probability(chunk) if len(chunk) > 0 else 0.0

        # Sample up to 5 evenly-spaced frames for efficiency
        n_frames = min(5, len(chunk) // frame_size)
        step = len(chunk) // n_frames
        probs = []
        for i in range(n_frames):
            start = i * step
            frame = chunk[start : start + frame_size]
            if len(frame) == frame_size:
                probs.append(self.speech_probability(frame))

        return round(float(np.mean(probs)), 4) if probs else 0.0


# Backward-compatible alias so existing imports keep working
VoiceActivityDetector = SileroVoiceDetector
