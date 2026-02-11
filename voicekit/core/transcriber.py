"""Local speech transcription engine — Faster-Whisper backend.

Uses CTranslate2-optimized Whisper models for fast, accurate,
fully offline speech-to-text. Auto-detects CUDA/MPS/CPU.

The model is downloaded automatically on first run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _detect_compute_type() -> tuple[str, str]:
    """Auto-detect the best device and compute type.

    Returns:
        Tuple of (device, compute_type) for Faster-Whisper.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Faster-Whisper doesn't natively support MPS yet,
            # fall back to CPU with int8 for Apple Silicon efficiency
            return "cpu", "int8"
    except ImportError:
        pass

    return "cpu", "int8"


class LocalTranscriber:
    """Offline speech-to-text using Faster-Whisper (CTranslate2).

    Provides fast, accurate transcription running entirely on-device.
    Models are auto-downloaded from Hugging Face on first use.

    Args:
        model_size: Whisper model variant. Options:
            "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
        device: Compute device ("cpu", "cuda", or "auto").
        compute_type: Quantization ("float16", "int8", "int8_float16", "auto").
        beam_size: Beam search width. Higher = more accurate but slower.
        download_root: Directory to cache downloaded models.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int = 5,
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.beam_size = beam_size

        # Auto-detect device and precision
        auto_device, auto_compute = _detect_compute_type()
        self.device = device or auto_device
        self.compute_type = compute_type or auto_compute

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for transcription. "
                "Install with: pip install faster-whisper"
            )

        try:
            self._model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{model_size}'. "
                f"The model will be downloaded automatically on first run. "
                f"Ensure you have internet access and sufficient disk space. "
                f"Error: {e}"
            ) from e

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, OGG, etc.).

        Returns:
            Full transcription as a single string.
        """
        audio_path = str(audio_path)

        try:
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                vad_filter=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Transcription failed for '{audio_path}': {e}"
            ) from e

        # Collect all segment texts
        text_parts = [segment.text for segment in segments]
        return " ".join(text_parts).strip()

    def transcribe_with_timestamps(
        self, audio_path: str | Path
    ) -> list[dict]:
        """Transcribe audio with per-segment timestamps.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of dicts with keys: "start", "end", "text", "confidence".
        """
        audio_path = str(audio_path)

        try:
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                vad_filter=True,
                word_timestamps=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Transcription failed for '{audio_path}': {e}"
            ) from e

        results = []
        for segment in segments:
            results.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                    "confidence": round(
                        np.exp(segment.avg_log_prob), 4
                    ),
                }
            )

        return results

    def transcribe_numpy(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio directly from a numpy array.

        Args:
            audio: 1-D numpy array of float32 samples.
            sample_rate: Sample rate in Hz (16000 recommended).

        Returns:
            Full transcription as a single string.
        """
        audio = audio.astype(np.float32)

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.beam_size,
                vad_filter=True,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription from numpy failed: {e}") from e

        text_parts = [segment.text for segment in segments]
        return " ".join(text_parts).strip()

    @property
    def detected_language(self) -> str | None:
        """Return the language detected during the last transcription.

        Only available after a transcribe() call. Returns None if
        using an English-only model.
        """
        return getattr(self, "_last_language", None)
