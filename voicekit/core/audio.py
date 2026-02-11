"""Core audio loading, preprocessing, and feature extraction engine."""

from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path
from typing import Union

import numpy as np

from voicekit.core.types import AudioConfig, AudioSegment


class AudioEngine:
    """Handles audio loading, resampling, and feature extraction.

    This is the foundation that all detectors and analyzers build on.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

    def load(self, source: Union[str, Path, bytes, np.ndarray]) -> AudioSegment:
        """Load audio from a file path, raw bytes, or numpy array.

        Args:
            source: File path (str/Path), WAV bytes, or numpy array of samples.

        Returns:
            AudioSegment with normalized mono audio at the configured sample rate.

        Raises:
            ValueError: If the source format is unsupported.
            FileNotFoundError: If the file path does not exist.
        """
        if isinstance(source, np.ndarray):
            samples = self._normalize_array(source)
            return self._make_segment(samples, self.config.sample_rate)

        if isinstance(source, bytes):
            return self._load_wav_bytes(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        return self._load_file(path)

    def _load_file(self, path: Path) -> AudioSegment:
        """Load audio from a file path."""
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            samples = self._to_mono(data)
            if sr != self.config.sample_rate:
                samples = self._resample(samples, sr, self.config.sample_rate)
            return self._make_segment(samples, self.config.sample_rate)
        except ImportError:
            # Fallback to wave module for WAV files
            if path.suffix.lower() in (".wav", ".wave"):
                with open(path, "rb") as f:
                    return self._load_wav_bytes(f.read())
            raise ValueError(
                f"Install soundfile for non-WAV formats: {path.suffix}"
            )

    def _load_wav_bytes(self, data: bytes) -> AudioSegment:
        """Load audio from WAV bytes using the stdlib wave module."""
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert raw bytes to float samples
        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            int_samples = struct.unpack(fmt, raw)
            samples = np.array(int_samples, dtype=np.float32) / 32768.0
        elif sampwidth == 4:
            fmt = f"<{n_frames * n_channels}i"
            int_samples = struct.unpack(fmt, raw)
            samples = np.array(int_samples, dtype=np.float32) / 2147483648.0
        elif sampwidth == 1:
            int_samples = np.frombuffer(raw, dtype=np.uint8)
            samples = (int_samples.astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)
            samples = self._to_mono(samples)

        if sr != self.config.sample_rate:
            samples = self._resample(samples, sr, self.config.sample_rate)

        return self._make_segment(samples, self.config.sample_rate)

    def _to_mono(self, samples: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if samples.ndim == 1:
            return samples
        return np.mean(samples, axis=1).astype(np.float32)

    def _normalize_array(self, samples: np.ndarray) -> np.ndarray:
        """Normalize a numpy array to float32 in [-1.0, 1.0]."""
        samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = self._to_mono(samples)
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples = samples / peak
        return samples

    def _resample(
        self, samples: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to the target sample rate using linear interpolation."""
        if orig_sr == target_sr:
            return samples
        try:
            from scipy.signal import resample as scipy_resample

            target_len = int(len(samples) * target_sr / orig_sr)
            return scipy_resample(samples, target_len).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            target_len = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, target_len)
            return np.interp(indices, np.arange(len(samples)), samples).astype(
                np.float32
            )

    def _make_segment(self, samples: np.ndarray, sr: int) -> AudioSegment:
        """Create an AudioSegment from numpy samples."""
        return AudioSegment(
            samples=samples.tolist(),
            sample_rate=sr,
            channels=1,
            duration_seconds=len(samples) / sr,
        )

    def extract_frames(
        self, segment: AudioSegment
    ) -> list[np.ndarray]:
        """Split audio into fixed-duration frames for analysis.

        Returns:
            List of numpy arrays, each containing one frame of audio.
        """
        samples = np.array(segment.samples, dtype=np.float32)
        frame_size = int(
            segment.sample_rate * self.config.frame_duration_ms / 1000
        )
        frames = []
        for start in range(0, len(samples), frame_size):
            frame = samples[start : start + frame_size]
            if len(frame) == frame_size:
                frames.append(frame)
        return frames

    def extract_mfcc(
        self,
        segment: AudioSegment,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> np.ndarray:
        """Extract MFCC features from an audio segment.

        Args:
            segment: Input audio segment.
            n_mfcc: Number of MFCC coefficients.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.

        Returns:
            MFCC feature matrix of shape (n_mfcc, n_frames).
        """
        samples = np.array(segment.samples, dtype=np.float32)
        try:
            import librosa

            mfccs = librosa.feature.mfcc(
                y=samples,
                sr=segment.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            return mfccs
        except ImportError:
            return self._mfcc_fallback(
                samples, segment.sample_rate, n_mfcc, n_fft, hop_length
            )

    def _mfcc_fallback(
        self,
        samples: np.ndarray,
        sr: int,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
    ) -> np.ndarray:
        """Compute MFCCs without librosa using basic DSP."""
        from scipy.fftpack import dct

        # Pre-emphasis
        emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])

        # Frame the signal
        frame_length = n_fft
        n_frames = 1 + (len(emphasized) - frame_length) // hop_length
        if n_frames <= 0:
            return np.zeros((n_mfcc, 1), dtype=np.float32)

        indices = (
            np.tile(np.arange(frame_length), (n_frames, 1))
            + np.tile(np.arange(n_frames) * hop_length, (frame_length, 1)).T
        )
        frames = emphasized[indices]

        # Apply Hamming window
        frames *= np.hamming(frame_length)

        # FFT and power spectrum
        mag = np.abs(np.fft.rfft(frames, n=n_fft))
        power = (mag ** 2) / n_fft

        # Mel filter bank
        n_mels = 40
        mel_filters = self._mel_filterbank(n_mels, n_fft, sr)
        mel_spec = np.dot(power, mel_filters.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
        log_mel = np.log(mel_spec)

        # DCT to get MFCCs
        mfccs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
        return mfccs.T.astype(np.float32)

    def _mel_filterbank(
        self, n_mels: int, n_fft: int, sr: int
    ) -> np.ndarray:
        """Create a Mel-scale filter bank."""
        low_freq = 0
        high_freq = sr / 2
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)

        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filters = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                filters[i, j] = (j - bin_points[i]) / max(
                    bin_points[i + 1] - bin_points[i], 1
                )
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                filters[i, j] = (bin_points[i + 2] - j) / max(
                    bin_points[i + 2] - bin_points[i + 1], 1
                )
        return filters

    @staticmethod
    def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def compute_energy(self, frame: np.ndarray) -> float:
        """Compute frame energy in decibels."""
        energy = np.sum(frame ** 2) / len(frame)
        if energy < 1e-10:
            return -100.0
        return float(10 * math.log10(energy))

    def compute_zcr(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate of a frame."""
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / (2 * len(frame)))

    def compute_pitch(
        self, samples: np.ndarray, sr: int, fmin: float = 50, fmax: float = 500
    ) -> float:
        """Estimate fundamental frequency using autocorrelation.

        Returns:
            Estimated F0 in Hz, or 0.0 if unvoiced.
        """
        if len(samples) < sr // int(fmin):
            return 0.0

        # Autocorrelation method
        corr = np.correlate(samples, samples, mode="full")
        corr = corr[len(corr) // 2 :]

        # Search within expected pitch range
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        max_lag = min(max_lag, len(corr) - 1)

        if min_lag >= max_lag:
            return 0.0

        search = corr[min_lag : max_lag + 1]
        if len(search) == 0 or np.max(search) <= 0:
            return 0.0

        peak_idx = np.argmax(search) + min_lag
        if corr[0] == 0:
            return 0.0

        # Voicing check
        if corr[peak_idx] / corr[0] < 0.3:
            return 0.0

        return float(sr / peak_idx)
