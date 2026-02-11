"""
VoiceKit — Production-grade voice detection and analysis SDK.

Designed for AI companies to integrate voice activity detection,
speaker identification, and voice characteristic analysis into
their products.

Usage:
    from voicekit import VoiceKit

    kit = VoiceKit()
    result = kit.analyze("audio.wav")
    print(result.vad.segments)
    print(result.speaker.embedding)
    print(result.characteristics.emotion)
"""

from voicekit.core.types import (
    AnalysisResult,
    AudioConfig,
    AudioSegment,
    CharacteristicsResult,
    QualityResult,
    SpeakerResult,
    VADResult,
    VoiceSegment,
)
from voicekit.sdk import VoiceKit

__version__ = "1.0.0"
__all__ = [
    "VoiceKit",
    "AudioConfig",
    "AudioSegment",
    "AnalysisResult",
    "VADResult",
    "VoiceSegment",
    "SpeakerResult",
    "CharacteristicsResult",
    "QualityResult",
]
