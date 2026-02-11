"""Wake word detection engine.

Supports two backends:
1. Porcupine (Picovoice) — ultra-low latency, custom wake words
2. Silero VAD + keyword matching — fully free, uses our existing VAD

The detector runs on raw audio frames and returns True when
the wake word is heard. Designed to consume <2% CPU when idle.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Optional

import numpy as np


class WakeWordBackend(str, Enum):
    """Available wake word detection backends."""
    PORCUPINE = "porcupine"
    VAD_WHISPER = "vad_whisper"


class WakeWordDetector:
    """Detects a configurable wake word from streaming audio.

    In PORCUPINE mode, uses Picovoice's on-device engine for
    sub-100ms latency wake word detection.

    In VAD_WHISPER mode, uses Silero VAD to detect speech, then
    runs Faster-Whisper on short clips to match the wake phrase.
    Fully free, no API key needed, but ~500ms latency.

    Args:
        wake_word: The trigger phrase (e.g., "orion", "hey jarvis").
        backend: Detection backend to use.
        porcupine_access_key: Picovoice access key (required for Porcupine).
        sensitivity: Detection sensitivity (0.0–1.0). Higher = more triggers.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        wake_word: str = "orion",
        backend: WakeWordBackend = WakeWordBackend.VAD_WHISPER,
        porcupine_access_key: str | None = None,
        sensitivity: float = 0.7,
        sample_rate: int = 16000,
    ) -> None:
        self.wake_word = wake_word.lower().strip()
        self.backend = backend
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate

        self._porcupine = None
        self._vad_model = None
        self._transcriber = None
        self._vad_buffer: list[np.ndarray] = []
        self._vad_state: dict | None = None
        self._speech_active = False
        self._silence_frames = 0
        self._max_silence_frames = 15  # ~450ms at 30ms frames

        if backend == WakeWordBackend.PORCUPINE:
            self._init_porcupine(porcupine_access_key)
        else:
            self._init_vad_whisper()

    def _init_porcupine(self, access_key: str | None) -> None:
        """Initialize Porcupine wake word engine."""
        if not access_key:
            raise ValueError(
                "Porcupine requires an access key. "
                "Get one free at https://console.picovoice.ai/"
            )

        try:
            import pvporcupine
        except ImportError:
            raise ImportError(
                "pvporcupine is required for Porcupine backend. "
                "Install with: pip install pvporcupine"
            )

        try:
            # Try built-in keyword first (e.g., "jarvis", "alexa", "computer")
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[self.wake_word],
                sensitivities=[self.sensitivity],
            )
        except pvporcupine.PorcupineInvalidArgumentError:
            # Custom wake word — needs a .ppn keyword file
            raise ValueError(
                f"'{self.wake_word}' is not a built-in Porcupine keyword. "
                f"Built-in options: alexa, americano, blueberry, bumblebee, "
                f"computer, grapefruit, grasshopper, hey google, hey siri, "
                f"jarvis, ok google, picovoice, porcupine, terminator. "
                f"For custom words, create a .ppn file at console.picovoice.ai"
            )

    def _init_vad_whisper(self) -> None:
        """Initialize VAD + Whisper wake word pipeline."""
        import torch

        # Load Silero VAD
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Silero VAD for wake word detection: {e}"
            ) from e

        # Load Faster-Whisper (small model for speed)
        try:
            from voicekit.core.transcriber import LocalTranscriber
            self._transcriber = LocalTranscriber(
                model_size="tiny.en",
                beam_size=3,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper for wake word detection: {e}"
            ) from e

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a single audio frame and check for wake word.

        Args:
            frame: Audio frame as float32 numpy array.
                   Porcupine: must be 512 samples at 16kHz.
                   VAD+Whisper: recommended 480-1536 samples at 16kHz.

        Returns:
            True if the wake word was detected in this frame.
        """
        if self.backend == WakeWordBackend.PORCUPINE:
            return self._process_porcupine(frame)
        else:
            return self._process_vad_whisper(frame)

    def _process_porcupine(self, frame: np.ndarray) -> bool:
        """Process frame through Porcupine engine."""
        if self._porcupine is None:
            return False

        # Porcupine expects int16
        int_frame = (frame * 32767).astype(np.int16)
        result = self._porcupine.process(int_frame)
        return result >= 0  # -1 = no detection, 0+ = keyword index

    def _process_vad_whisper(self, frame: np.ndarray) -> bool:
        """Process frame through VAD → Whisper pipeline.

        Strategy:
        1. Run Silero VAD on each frame
        2. When speech starts, buffer frames
        3. When speech ends (silence detected), run Whisper on buffer
        4. Check if transcription contains the wake word
        """
        import torch

        tensor = torch.from_numpy(frame).float().unsqueeze(0)

        with torch.no_grad():
            speech_prob = self._vad_model(tensor, self.sample_rate).item()

        is_speech = speech_prob >= 0.5

        if is_speech:
            self._vad_buffer.append(frame.copy())
            self._speech_active = True
            self._silence_frames = 0

            # Safety cap: don't buffer more than 4 seconds
            max_frames = int(4.0 * self.sample_rate / len(frame))
            if len(self._vad_buffer) > max_frames:
                self._vad_buffer = self._vad_buffer[-max_frames:]

        elif self._speech_active:
            self._silence_frames += 1
            if self._silence_frames >= self._max_silence_frames:
                # Speech ended — check for wake word
                detected = self._check_wake_word()
                self._vad_buffer.clear()
                self._speech_active = False
                self._silence_frames = 0
                self._vad_model.reset_states()
                return detected

        return False

    def _check_wake_word(self) -> bool:
        """Run Whisper on buffered audio and check for wake word."""
        if not self._vad_buffer or self._transcriber is None:
            return False

        audio = np.concatenate(self._vad_buffer)

        # Minimum 0.3s of audio to transcribe
        if len(audio) < self.sample_rate * 0.3:
            return False

        try:
            text = self._transcriber.transcribe_numpy(audio, self.sample_rate)
            text = text.lower().strip()
            # Fuzzy match — handle "orion" / "o.r.i.o.n." / "oh ryan" etc.
            return self._fuzzy_match(text, self.wake_word)
        except Exception:
            return False

    def _fuzzy_match(self, transcript: str, wake_word: str) -> bool:
        """Check if transcript contains the wake word with fuzzy matching."""
        # Strip punctuation and normalize
        clean = re.sub(r"[^a-z0-9\s]", "", transcript)
        target = re.sub(r"[^a-z0-9\s]", "", wake_word)

        # Exact substring
        if target in clean:
            return True

        # Handle common STT misheard variants
        # "orion" might become "oh ryan", "o'ryan", "ori on"
        target_nospace = target.replace(" ", "")
        clean_nospace = clean.replace(" ", "")
        if target_nospace in clean_nospace:
            return True

        # Phonetic proximity — check if >80% of characters match
        # in a sliding window
        if len(target_nospace) > 2:
            for i in range(len(clean_nospace) - len(target_nospace) + 1):
                window = clean_nospace[i : i + len(target_nospace)]
                matches = sum(a == b for a, b in zip(window, target_nospace))
                if matches / len(target_nospace) >= 0.8:
                    return True

        return False

    def reset(self) -> None:
        """Reset internal state (clear buffers, reset VAD)."""
        self._vad_buffer.clear()
        self._speech_active = False
        self._silence_frames = 0
        self._vad_state = None
        if self._vad_model is not None:
            self._vad_model.reset_states()

    def cleanup(self) -> None:
        """Release resources."""
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
