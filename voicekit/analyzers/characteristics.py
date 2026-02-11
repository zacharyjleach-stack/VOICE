"""Voice characteristics analyzer.

Analyzes vocal traits including emotion, gender, age group,
pitch, energy, and speech rate from audio segments.
"""

from __future__ import annotations

import numpy as np

from voicekit.core.audio import AudioEngine
from voicekit.core.types import (
    AgeGroup,
    AudioConfig,
    AudioSegment,
    CharacteristicsResult,
    Emotion,
    Gender,
)


class CharacteristicsAnalyzer:
    """Analyzes voice characteristics from audio.

    Extracts prosodic and spectral features to estimate emotion,
    gender, age group, and other vocal properties.
    """

    def __init__(
        self, config: AudioConfig | None = None, engine: AudioEngine | None = None
    ) -> None:
        self.config = config or AudioConfig()
        self.engine = engine or AudioEngine(self.config)

    def analyze(self, segment: AudioSegment) -> CharacteristicsResult:
        """Analyze voice characteristics from an audio segment.

        Args:
            segment: Audio segment to analyze.

        Returns:
            CharacteristicsResult with detected vocal properties.
        """
        samples = np.array(segment.samples, dtype=np.float32)
        sr = segment.sample_rate

        # Extract fundamental features
        pitch_values = self._extract_pitch_contour(samples, sr)
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        energy_db = self._compute_segment_energy(samples)
        speech_rate = self._estimate_speech_rate(samples, sr)

        # Extract MFCCs for classification tasks
        mfccs = self.engine.extract_mfcc(segment, n_mfcc=13)

        # Classify characteristics
        emotion, emotion_scores = self._classify_emotion(
            mfccs, pitch_mean, pitch_std, energy_db, speech_rate
        )
        gender, gender_conf = self._classify_gender(pitch_mean, mfccs)
        age_group, age_conf = self._classify_age_group(
            pitch_mean, speech_rate, mfccs
        )

        return CharacteristicsResult(
            emotion=emotion,
            emotion_scores=emotion_scores,
            gender=gender,
            gender_confidence=gender_conf,
            age_group=age_group,
            age_group_confidence=age_conf,
            pitch_mean_hz=pitch_mean,
            pitch_std_hz=pitch_std,
            energy_mean_db=energy_db,
            speech_rate_sps=speech_rate,
        )

    def _extract_pitch_contour(
        self, samples: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract pitch contour from audio."""
        frame_size = int(sr * 0.03)  # 30ms frames
        hop_size = int(sr * 0.01)  # 10ms hop
        pitches = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start : start + frame_size]
            f0 = self.engine.compute_pitch(frame, sr)
            if f0 > 0:
                pitches.append(f0)

        return np.array(pitches)

    def _compute_segment_energy(self, samples: np.ndarray) -> float:
        """Compute overall segment energy in dB."""
        energy = np.mean(samples ** 2)
        if energy < 1e-10:
            return -100.0
        return float(10 * np.log10(energy))

    def _estimate_speech_rate(
        self, samples: np.ndarray, sr: int
    ) -> float:
        """Estimate speech rate in syllables per second.

        Uses energy envelope peak counting as a proxy for syllable rate.
        """
        # Compute energy envelope
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        envelope = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start : start + frame_size]
            envelope.append(np.sqrt(np.mean(frame ** 2)))

        if len(envelope) < 3:
            return 0.0

        envelope = np.array(envelope)

        # Smooth envelope
        kernel_size = 5
        if len(envelope) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            envelope = np.convolve(envelope, kernel, mode="same")

        # Count peaks above threshold
        threshold = np.mean(envelope) * 0.5
        peaks = 0
        above = False
        for val in envelope:
            if val > threshold and not above:
                peaks += 1
                above = True
            elif val <= threshold:
                above = False

        duration = len(samples) / sr
        if duration == 0:
            return 0.0

        return float(peaks / duration)

    def _classify_emotion(
        self,
        mfccs: np.ndarray,
        pitch_mean: float,
        pitch_std: float,
        energy_db: float,
        speech_rate: float,
    ) -> tuple[Emotion, dict[str, float]]:
        """Classify emotion from prosodic and spectral features.

        Uses a rule-based approach with feature-derived scoring.
        """
        scores: dict[str, float] = {}

        # Feature normalization (approximate ranges)
        pitch_norm = min(pitch_mean / 300.0, 1.0) if pitch_mean > 0 else 0.0
        pitch_var = min(pitch_std / 80.0, 1.0)
        energy_norm = min(max((energy_db + 40) / 40, 0), 1.0)
        rate_norm = min(speech_rate / 8.0, 1.0)

        # MFCC-derived features
        mfcc_mean = np.mean(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)
        spectral_tilt = float(mfcc_mean[1]) if len(mfcc_mean) > 1 else 0.0
        spectral_tilt_norm = (spectral_tilt + 30) / 60  # normalize roughly

        # Scoring heuristics based on speech prosody research
        scores["neutral"] = 0.3 + 0.2 * (1 - pitch_var) + 0.1 * (1 - abs(energy_norm - 0.5))
        scores["happy"] = 0.1 + 0.3 * pitch_norm + 0.2 * pitch_var + 0.2 * energy_norm + 0.1 * rate_norm
        scores["sad"] = 0.1 + 0.3 * (1 - energy_norm) + 0.2 * (1 - rate_norm) + 0.2 * (1 - pitch_var)
        scores["angry"] = 0.1 + 0.3 * energy_norm + 0.2 * pitch_var + 0.1 * rate_norm + 0.1 * max(spectral_tilt_norm, 0)
        scores["fearful"] = 0.05 + 0.3 * pitch_var + 0.2 * rate_norm + 0.1 * (1 - energy_norm)
        scores["surprised"] = 0.05 + 0.3 * pitch_norm + 0.3 * pitch_var + 0.1 * energy_norm
        scores["disgusted"] = 0.05 + 0.2 * (1 - rate_norm) + 0.2 * (1 - pitch_norm) + 0.1 * energy_norm

        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 4) for k, v in scores.items()}

        best_emotion = max(scores, key=lambda k: scores[k])
        return Emotion(best_emotion), scores

    def _classify_gender(
        self, pitch_mean: float, mfccs: np.ndarray
    ) -> tuple[Gender, float]:
        """Classify speaker gender from pitch and spectral features."""
        if pitch_mean <= 0:
            return Gender.UNKNOWN, 0.0

        # Pitch-based classification with soft boundaries
        # Male: ~85-180 Hz, Female: ~165-300 Hz
        if pitch_mean < 150:
            confidence = min((150 - pitch_mean) / 65, 1.0)
            return Gender.MALE, round(0.6 + 0.4 * confidence, 3)
        elif pitch_mean > 185:
            confidence = min((pitch_mean - 185) / 115, 1.0)
            return Gender.FEMALE, round(0.6 + 0.4 * confidence, 3)
        else:
            # Ambiguous range — use spectral features to help
            mfcc_mean = np.mean(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)
            # Lower formants suggest male
            if len(mfcc_mean) > 3 and mfcc_mean[2] < 0:
                return Gender.MALE, 0.55
            else:
                return Gender.FEMALE, 0.55

    def _classify_age_group(
        self, pitch_mean: float, speech_rate: float, mfccs: np.ndarray
    ) -> tuple[AgeGroup, float]:
        """Classify speaker age group from vocal features."""
        if pitch_mean <= 0:
            return AgeGroup.ADULT, 0.3

        mfcc_mean = np.mean(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)
        mfcc_var = np.var(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)

        # Children typically have higher pitch (>250 Hz) and more variability
        if pitch_mean > 250:
            return AgeGroup.CHILD, 0.7

        # Seniors often have pitch instability and slower rate
        pitch_instability = float(np.mean(mfcc_var[:5])) if len(mfcc_var) >= 5 else 0
        if speech_rate < 2.5 and pitch_instability > 15:
            return AgeGroup.SENIOR, 0.55

        # Young adults tend to have higher pitch than older adults
        if pitch_mean > 180:
            return AgeGroup.YOUNG_ADULT, 0.5
        else:
            return AgeGroup.ADULT, 0.5
