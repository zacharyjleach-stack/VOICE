"""Audio quality assessment module.

Evaluates audio quality for downstream processing, detecting issues
like noise, clipping, silence, and low signal levels.
"""

from __future__ import annotations

import numpy as np

from voicekit.core.audio import AudioEngine
from voicekit.core.types import AudioConfig, AudioSegment, QualityResult


class QualityAnalyzer:
    """Assesses audio quality and usability for voice processing.

    Checks for common audio issues that can degrade the performance
    of voice detection and analysis systems.
    """

    def __init__(
        self, config: AudioConfig | None = None, engine: AudioEngine | None = None
    ) -> None:
        self.config = config or AudioConfig()
        self.engine = engine or AudioEngine(self.config)

    def analyze(self, segment: AudioSegment) -> QualityResult:
        """Assess audio quality of a segment.

        Args:
            segment: Audio segment to evaluate.

        Returns:
            QualityResult with quality metrics and detected issues.
        """
        samples = np.array(segment.samples, dtype=np.float32)
        issues: list[str] = []

        if len(samples) == 0:
            return QualityResult(
                overall_score=0.0,
                is_usable=False,
                issues=["Empty audio segment"],
            )

        # Signal-to-noise ratio estimation
        snr_db = self._estimate_snr(samples, segment.sample_rate)

        # Clipping detection
        clipping_ratio = self._detect_clipping(samples)

        # Silence ratio
        silence_ratio = self._compute_silence_ratio(samples, segment.sample_rate)

        # Issue detection
        if snr_db < 5:
            issues.append(f"Very low SNR ({snr_db:.1f} dB) — high noise level")
        elif snr_db < 15:
            issues.append(f"Low SNR ({snr_db:.1f} dB) — noticeable noise")

        if clipping_ratio > 0.01:
            issues.append(
                f"Audio clipping detected ({clipping_ratio * 100:.1f}% of samples)"
            )

        if silence_ratio > 0.9:
            issues.append(
                f"Mostly silence ({silence_ratio * 100:.0f}%) — insufficient speech"
            )

        peak = float(np.max(np.abs(samples)))
        if peak < 0.01:
            issues.append("Very low signal level — audio may be inaudible")
        elif peak < 0.1:
            issues.append("Low signal level — consider amplification")

        # DC offset check
        dc_offset = abs(float(np.mean(samples)))
        if dc_offset > 0.05:
            issues.append(f"Significant DC offset ({dc_offset:.3f})")

        # Duration check
        if segment.duration_seconds < 0.5:
            issues.append("Very short audio segment (< 0.5s)")

        # Overall score computation
        overall_score = self._compute_overall_score(
            snr_db, clipping_ratio, silence_ratio, peak, dc_offset
        )

        is_usable = overall_score >= 0.3 and silence_ratio < 0.95

        return QualityResult(
            overall_score=round(overall_score, 3),
            snr_db=round(snr_db, 2),
            clipping_ratio=round(clipping_ratio, 5),
            silence_ratio=round(silence_ratio, 3),
            is_usable=is_usable,
            issues=issues,
        )

    def _estimate_snr(self, samples: np.ndarray, sr: int) -> float:
        """Estimate signal-to-noise ratio using energy-based VAD.

        Separates frames into signal and noise based on energy,
        then computes SNR from the ratio.
        """
        frame_size = int(sr * 0.03)
        energies = []

        for start in range(0, len(samples) - frame_size, frame_size):
            frame = samples[start : start + frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)

        if len(energies) < 2:
            return 0.0

        energies = np.array(energies)
        sorted_e = np.sort(energies)

        # Bottom 20% as noise estimate
        noise_count = max(1, len(sorted_e) // 5)
        noise_energy = np.mean(sorted_e[:noise_count])

        # Top 30% as signal estimate
        signal_count = max(1, int(len(sorted_e) * 0.3))
        signal_energy = np.mean(sorted_e[-signal_count:])

        if noise_energy < 1e-10:
            noise_energy = 1e-10

        snr = 10 * np.log10(signal_energy / noise_energy)
        return float(max(snr, 0.0))

    def _detect_clipping(
        self, samples: np.ndarray, threshold: float = 0.99
    ) -> float:
        """Detect clipping by counting samples near max amplitude."""
        clipped = np.sum(np.abs(samples) >= threshold)
        return float(clipped / len(samples))

    def _compute_silence_ratio(
        self, samples: np.ndarray, sr: int
    ) -> float:
        """Compute the ratio of silent frames."""
        frame_size = int(sr * 0.03)
        silence_threshold = -40  # dB
        total_frames = 0
        silent_frames = 0

        for start in range(0, len(samples) - frame_size, frame_size):
            frame = samples[start : start + frame_size]
            energy = np.mean(frame ** 2)
            energy_db = (
                10 * np.log10(energy) if energy > 1e-10 else -100
            )
            total_frames += 1
            if energy_db < silence_threshold:
                silent_frames += 1

        if total_frames == 0:
            return 1.0
        return float(silent_frames / total_frames)

    def _compute_overall_score(
        self,
        snr_db: float,
        clipping_ratio: float,
        silence_ratio: float,
        peak: float,
        dc_offset: float,
    ) -> float:
        """Compute overall quality score (0.0-1.0)."""
        # SNR component (0-1, 30dB+ = perfect)
        snr_score = min(snr_db / 30.0, 1.0)

        # Clipping component (penalize clipping)
        clip_score = max(1.0 - clipping_ratio * 20, 0.0)

        # Silence component (penalize too much silence)
        silence_score = max(1.0 - silence_ratio, 0.0)

        # Level component
        level_score = min(peak / 0.5, 1.0)

        # DC offset component
        dc_score = max(1.0 - dc_offset * 10, 0.0)

        # Weighted average
        overall = (
            0.35 * snr_score
            + 0.20 * clip_score
            + 0.20 * silence_score
            + 0.15 * level_score
            + 0.10 * dc_score
        )
        return min(max(overall, 0.0), 1.0)
