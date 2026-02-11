"""VoiceKit Streaming VAD Example.

Demonstrates real-time voice activity detection, processing
audio frame-by-frame as it would arrive from a microphone.
"""

import numpy as np

from voicekit import VoiceKit

kit = VoiceKit()
sr = 16000
frame_size = 480  # 30ms at 16kHz

# Simulate 3 seconds of audio with speech and silence
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Speech from 0.5-1.5s and 2.0-2.8s, silence elsewhere
signal = np.zeros_like(t, dtype=np.float32)
speech_ranges = [(0.5, 1.5), (2.0, 2.8)]

for start, end in speech_ranges:
    mask = (t >= start) & (t < end)
    f0 = 150
    speech = np.sin(2 * np.pi * f0 * t[mask])
    speech += 0.5 * np.sin(2 * np.pi * 2 * f0 * t[mask])
    signal[mask] = (speech * 0.7).astype(np.float32)

# Process frame by frame (simulating real-time)
print("=== Streaming VAD ===\n")
print(f"{'Time':>8s}  {'Speech':>6s}  {'Confidence':>10s}")
print("-" * 30)

state = None
for i in range(0, len(signal) - frame_size, frame_size):
    frame = signal[i : i + frame_size]
    is_speech, confidence, state = kit.process_stream_frame(frame, state)

    time_s = i / sr
    if is_speech:
        print(f"{time_s:7.2f}s  {'YES':>6s}  {confidence:10.3f}")
    elif confidence > 0.2:
        print(f"{time_s:7.2f}s  {'  no':>6s}  {confidence:10.3f}")
