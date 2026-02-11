"""Voice detection modules."""

from voicekit.detectors.vad import VoiceActivityDetector
from voicekit.detectors.speaker import SpeakerDetector

__all__ = ["VoiceActivityDetector", "SpeakerDetector"]
