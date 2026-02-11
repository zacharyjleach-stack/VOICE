"""VoiceKit Quickstart — Full analysis pipeline.

This example shows how to use VoiceKit for comprehensive
voice analysis in just a few lines of code.
"""

import numpy as np

from voicekit import VoiceKit

# Initialize the SDK
kit = VoiceKit()

# Generate sample audio (replace with real audio in production)
sr = 16000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Simulate speech: fundamental + harmonics with amplitude modulation
f0 = 150 + 20 * np.sin(2 * np.pi * 3 * t)
phase = np.cumsum(2 * np.pi * f0 / sr)
signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
signal = (signal * envelope * 0.7).astype(np.float32)

# Run full analysis
result = kit.analyze(signal)

# Voice Activity Detection
print("=== Voice Activity Detection ===")
print(f"Speech segments: {len(result.vad.segments)}")
print(f"Speech ratio: {result.vad.speech_ratio:.1%}")
for seg in result.vad.segments:
    print(f"  {seg.start_seconds:.2f}s - {seg.end_seconds:.2f}s (conf: {seg.confidence:.2f})")

# Speaker Identification
print(f"\n=== Speaker ===")
print(f"Embedding dimensions: {len(result.speaker.embedding)}")

# Voice Characteristics
print(f"\n=== Characteristics ===")
print(f"Emotion: {result.characteristics.emotion.value}")
print(f"Gender: {result.characteristics.gender.value} ({result.characteristics.gender_confidence:.0%})")
print(f"Age group: {result.characteristics.age_group.value}")
print(f"Pitch: {result.characteristics.pitch_mean_hz:.1f} Hz")
print(f"Speech rate: {result.characteristics.speech_rate_sps:.1f} syl/s")

# Quality Assessment
print(f"\n=== Quality ===")
print(f"Score: {result.quality.overall_score:.3f}")
print(f"SNR: {result.quality.snr_db:.1f} dB")
print(f"Usable: {result.quality.is_usable}")

print(f"\nProcessing time: {result.processing_time_seconds:.3f}s")
