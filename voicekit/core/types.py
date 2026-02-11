"""Core type definitions for the VoiceKit SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Emotion(str, Enum):
    """Detected emotion in voice."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"


class Gender(str, Enum):
    """Detected speaker gender."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    """Detected speaker age group."""

    CHILD = "child"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    SENIOR = "senior"


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio processing.

    Attributes:
        sample_rate: Target sample rate in Hz. Audio will be resampled if needed.
        frame_duration_ms: Duration of each analysis frame in milliseconds.
        min_speech_duration_ms: Minimum speech segment duration to keep.
        max_silence_duration_ms: Maximum silence within a speech segment before splitting.
        vad_threshold: Voice activity detection confidence threshold (0.0-1.0).
        embedding_dim: Dimensionality of speaker embeddings.
    """

    sample_rate: int = 16000
    frame_duration_ms: int = 30
    min_speech_duration_ms: int = 250
    max_silence_duration_ms: int = 300
    vad_threshold: float = 0.5
    embedding_dim: int = 192


@dataclass(frozen=True)
class AudioSegment:
    """A segment of audio data.

    Attributes:
        samples: Raw audio samples as a list of floats (-1.0 to 1.0).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        duration_seconds: Duration of the segment in seconds.
    """

    samples: list[float]
    sample_rate: int
    channels: int = 1
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_seconds == 0.0 and self.samples:
            object.__setattr__(
                self, "duration_seconds", len(self.samples) / self.sample_rate
            )


@dataclass(frozen=True)
class VoiceSegment:
    """A detected voice activity segment.

    Attributes:
        start_seconds: Start time of the segment.
        end_seconds: End time of the segment.
        confidence: Detection confidence (0.0-1.0).
    """

    start_seconds: float
    end_seconds: float
    confidence: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class VADResult:
    """Voice Activity Detection result.

    Attributes:
        segments: List of detected voice segments.
        speech_ratio: Ratio of speech to total audio duration.
        total_speech_seconds: Total seconds of detected speech.
        total_duration_seconds: Total audio duration in seconds.
    """

    segments: list[VoiceSegment] = field(default_factory=list)
    speech_ratio: float = 0.0
    total_speech_seconds: float = 0.0
    total_duration_seconds: float = 0.0


@dataclass(frozen=True)
class SpeakerResult:
    """Speaker identification result.

    Attributes:
        embedding: Speaker embedding vector for comparison/clustering.
        speaker_count: Estimated number of distinct speakers.
        speaker_id: Assigned speaker label (when using enrollment).
        confidence: Identification confidence (0.0-1.0).
    """

    embedding: list[float] = field(default_factory=list)
    speaker_count: int = 1
    speaker_id: Optional[str] = None
    confidence: float = 0.0


@dataclass(frozen=True)
class CharacteristicsResult:
    """Voice characteristics analysis result.

    Attributes:
        emotion: Detected primary emotion.
        emotion_scores: Confidence scores for each emotion.
        gender: Detected gender.
        gender_confidence: Gender detection confidence.
        age_group: Detected age group.
        age_group_confidence: Age group detection confidence.
        pitch_mean_hz: Mean fundamental frequency in Hz.
        pitch_std_hz: Standard deviation of fundamental frequency.
        energy_mean_db: Mean energy in decibels.
        speech_rate_sps: Estimated speech rate (syllables per second).
    """

    emotion: Emotion = Emotion.NEUTRAL
    emotion_scores: dict[str, float] = field(default_factory=dict)
    gender: Gender = Gender.UNKNOWN
    gender_confidence: float = 0.0
    age_group: AgeGroup = AgeGroup.ADULT
    age_group_confidence: float = 0.0
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    energy_mean_db: float = 0.0
    speech_rate_sps: float = 0.0


@dataclass(frozen=True)
class QualityResult:
    """Audio quality assessment result.

    Attributes:
        overall_score: Overall quality score (0.0-1.0).
        snr_db: Estimated signal-to-noise ratio in dB.
        clipping_ratio: Ratio of clipped samples.
        silence_ratio: Ratio of silent frames.
        is_usable: Whether the audio quality is sufficient for analysis.
        issues: List of detected quality issues.
    """

    overall_score: float = 0.0
    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    is_usable: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisResult:
    """Complete voice analysis result combining all detectors.

    Attributes:
        audio: The processed audio segment.
        vad: Voice activity detection result.
        speaker: Speaker identification result.
        characteristics: Voice characteristics analysis result.
        quality: Audio quality assessment result.
        processing_time_seconds: Total processing time.
    """

    audio: AudioSegment
    vad: VADResult
    speaker: SpeakerResult
    characteristics: CharacteristicsResult
    quality: QualityResult
    processing_time_seconds: float = 0.0
