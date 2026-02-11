"""Voice Activity Detection (VAD) engine.

Implements a multi-feature energy-based VAD with adaptive thresholding.
Designed for production use with real-time streaming and batch modes.
"""

from __future__ import annotations

import numpy as np

from voicekit.core.audio import AudioEngine
from voicekit.core.types import AudioConfig, AudioSegment, VADResult, VoiceSegment


class VoiceActivityDetector:
    """Detects speech segments in audio using multi-feature analysis.

    Combines frame energy, zero-crossing rate, and spectral features
    to determine voiced regions with adaptive thresholding.
    """

    def __init__(
        self, config: AudioConfig | None = None, engine: AudioEngine | None = None
    ) -> None:
        self.config = config or AudioConfig()
        self.engine = engine or AudioEngine(self.config)

    def detect(self, segment: AudioSegment) -> VADResult:
        """Run voice activity detection on an audio segment.

        Args:
            segment: Audio segment to analyze.

        Returns:
            VADResult with detected speech segments and statistics.
        """
        frames = self.engine.extract_frames(segment)
        if not frames:
            return VADResult(total_duration_seconds=segment.duration_seconds)

        # Compute per-frame features
        energies = np.array([self.engine.compute_energy(f) for f in frames])
        zcrs = np.array([self.engine.compute_zcr(f) for f in frames])
        spectral_flatness = np.array(
            [self._spectral_flatness(f) for f in frames]
        )

        # Adaptive threshold based on energy distribution
        energy_threshold = self._compute_adaptive_threshold(energies)

        # ZCR threshold — speech typically has moderate ZCR
        zcr_low, zcr_high = 0.02, 0.35

        # Classify each frame
        frame_labels = np.zeros(len(frames), dtype=np.int32)
        for i in range(len(frames)):
            is_energetic = energies[i] > energy_threshold
            is_speech_zcr = zcr_low <= zcrs[i] <= zcr_high
            is_not_noise = spectral_flatness[i] < 0.85

            score = (
                0.5 * float(is_energetic)
                + 0.25 * float(is_speech_zcr)
                + 0.25 * float(is_not_noise)
            )
            if score >= self.config.vad_threshold:
                frame_labels[i] = 1

        # Apply hangover scheme to smooth detections
        frame_labels = self._apply_hangover(frame_labels)

        # Convert frame labels to time segments
        segments = self._labels_to_segments(
            frame_labels, segment.sample_rate, energies
        )

        # Filter short segments
        min_dur = self.config.min_speech_duration_ms / 1000
        segments = [s for s in segments if s.duration_seconds >= min_dur]

        total_speech = sum(s.duration_seconds for s in segments)
        speech_ratio = (
            total_speech / segment.duration_seconds
            if segment.duration_seconds > 0
            else 0.0
        )

        return VADResult(
            segments=segments,
            speech_ratio=speech_ratio,
            total_speech_seconds=total_speech,
            total_duration_seconds=segment.duration_seconds,
        )

    def detect_streaming(
        self, frame: np.ndarray, state: dict | None = None
    ) -> tuple[bool, float, dict]:
        """Process a single frame for streaming/real-time VAD.

        Args:
            frame: Single audio frame as numpy array.
            state: Persistent state dict from previous call (or None for first).

        Returns:
            Tuple of (is_speech, confidence, updated_state).
        """
        if state is None:
            state = {
                "energy_history": [],
                "hangover_counter": 0,
                "speech_active": False,
            }

        energy = self.engine.compute_energy(frame)
        zcr = self.engine.compute_zcr(frame)
        flatness = self._spectral_flatness(frame)

        state["energy_history"].append(energy)
        # Keep rolling window
        if len(state["energy_history"]) > 100:
            state["energy_history"] = state["energy_history"][-100:]

        threshold = self._compute_adaptive_threshold(
            np.array(state["energy_history"])
        )

        is_energetic = energy > threshold
        is_speech_zcr = 0.02 <= zcr <= 0.35
        is_not_noise = flatness < 0.85

        confidence = (
            0.5 * float(is_energetic)
            + 0.25 * float(is_speech_zcr)
            + 0.25 * float(is_not_noise)
        )

        is_speech = confidence >= self.config.vad_threshold

        # Hangover logic
        hangover_frames = int(
            self.config.max_silence_duration_ms
            / self.config.frame_duration_ms
        )
        if is_speech:
            state["hangover_counter"] = hangover_frames
            state["speech_active"] = True
        elif state["hangover_counter"] > 0:
            state["hangover_counter"] -= 1
            is_speech = True
        else:
            state["speech_active"] = False

        return is_speech, confidence, state

    def _compute_adaptive_threshold(self, energies: np.ndarray) -> float:
        """Compute adaptive energy threshold using noise floor estimation."""
        if len(energies) == 0:
            return -30.0

        sorted_e = np.sort(energies)
        # Estimate noise floor from quietest 20%
        noise_count = max(1, len(sorted_e) // 5)
        noise_floor = np.mean(sorted_e[:noise_count])

        # Threshold between noise floor and signal
        signal_level = np.mean(sorted_e[-noise_count:])
        threshold = noise_floor + 0.4 * (signal_level - noise_floor)

        return float(threshold)

    def _spectral_flatness(self, frame: np.ndarray) -> float:
        """Compute spectral flatness (Wiener entropy). 1.0 = noise, 0.0 = tonal."""
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum[1:]  # Skip DC
        if len(spectrum) == 0 or np.max(spectrum) == 0:
            return 1.0

        spectrum = np.maximum(spectrum, 1e-10)
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)

        if arith_mean == 0:
            return 1.0
        return float(geo_mean / arith_mean)

    def _apply_hangover(self, labels: np.ndarray) -> np.ndarray:
        """Apply hangover scheme to smooth frame classifications."""
        hangover_frames = int(
            self.config.max_silence_duration_ms
            / self.config.frame_duration_ms
        )
        result = labels.copy()
        counter = 0

        for i in range(len(result)):
            if labels[i] == 1:
                counter = hangover_frames
            elif counter > 0:
                result[i] = 1
                counter -= 1
        return result

    def _labels_to_segments(
        self,
        labels: np.ndarray,
        sample_rate: int,
        energies: np.ndarray,
    ) -> list[VoiceSegment]:
        """Convert binary frame labels to VoiceSegment list."""
        frame_dur = self.config.frame_duration_ms / 1000.0
        segments: list[VoiceSegment] = []
        start_idx: int | None = None

        for i in range(len(labels)):
            if labels[i] == 1 and start_idx is None:
                start_idx = i
            elif labels[i] == 0 and start_idx is not None:
                seg_energies = energies[start_idx:i]
                confidence = self._segment_confidence(seg_energies, energies)
                segments.append(
                    VoiceSegment(
                        start_seconds=start_idx * frame_dur,
                        end_seconds=i * frame_dur,
                        confidence=confidence,
                    )
                )
                start_idx = None

        # Handle segment ending at the last frame
        if start_idx is not None:
            seg_energies = energies[start_idx:]
            confidence = self._segment_confidence(seg_energies, energies)
            segments.append(
                VoiceSegment(
                    start_seconds=start_idx * frame_dur,
                    end_seconds=len(labels) * frame_dur,
                    confidence=confidence,
                )
            )

        return segments

    def _segment_confidence(
        self, seg_energies: np.ndarray, all_energies: np.ndarray
    ) -> float:
        """Estimate confidence for a speech segment."""
        if len(seg_energies) == 0:
            return 0.0
        sorted_all = np.sort(all_energies)
        noise_count = max(1, len(sorted_all) // 5)
        noise_floor = np.mean(sorted_all[:noise_count])
        seg_mean = np.mean(seg_energies)
        diff = seg_mean - noise_floor
        # Normalize to [0, 1] — 20dB above noise = max confidence
        confidence = min(max(diff / 20.0, 0.0), 1.0)
        return float(confidence)
