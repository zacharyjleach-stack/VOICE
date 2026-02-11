"""VoiceKit Speaker Verification Example.

Demonstrates enrolling speakers and verifying identity.
"""

import numpy as np

from voicekit import VoiceKit


def make_voice(f0: float, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a simple voice-like signal at a given fundamental frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = np.cumsum(2 * np.pi * f0 / sr * np.ones_like(t))
    signal = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    return (signal * 0.7 / np.max(np.abs(signal))).astype(np.float32)


kit = VoiceKit()

# Enroll two speakers with different voice characteristics
alice_audio = make_voice(f0=220)  # Higher pitch
bob_audio = make_voice(f0=110)  # Lower pitch

kit.enroll_speaker("alice", alice_audio)
kit.enroll_speaker("bob", bob_audio)

# Verify speakers
print("=== Speaker Verification ===\n")

# Test with Alice's voice
is_alice, score = kit.verify_speaker(alice_audio, "alice")
print(f"Alice's audio vs 'alice': match={is_alice}, score={score:.3f}")

is_bob, score = kit.verify_speaker(alice_audio, "bob")
print(f"Alice's audio vs 'bob':   match={is_bob}, score={score:.3f}")

# Compare two audio samples
similarity = kit.compare_speakers(alice_audio, bob_audio)
print(f"\nAlice vs Bob similarity: {similarity:.3f}")
