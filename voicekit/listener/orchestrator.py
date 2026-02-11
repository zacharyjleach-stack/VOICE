"""Voice Assistant orchestrator.

Ties together the full pipeline:
  Microphone → Wake Word → Record Utterance → STT → Callback → TTS

This is the "Jarvis loop" — always listening in the background,
activating on the wake word, transcribing the command, dispatching
it to your handler, and speaking the response.

Usage:
    from voicekit.listener import VoiceAssistant

    def handle_command(text: str) -> str:
        return f"You said: {text}"

    assistant = VoiceAssistant(
        wake_word="orion",
        on_command=handle_command,
        elevenlabs_api_key="sk-...",
    )
    assistant.run()  # Blocks forever, listening
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Callable, Optional

import numpy as np

from voicekit.listener.mic import MicrophoneListener
from voicekit.listener.wake_word import WakeWordBackend, WakeWordDetector
from voicekit.tts.engine import TTSBackend, TTSEngine


class VoiceAssistant:
    """Full voice assistant orchestrator.

    Manages the complete loop: listen → detect wake word → record
    command → transcribe → handle → speak response.

    Args:
        wake_word: Trigger phrase (e.g., "orion", "jarvis").
        on_command: Callback that receives the transcribed command text
                    and returns a response string to speak.
        wake_word_backend: Wake word detection backend.
        tts_backend: TTS output backend.
        elevenlabs_api_key: API key for ElevenLabs TTS.
        elevenlabs_voice_id: Voice ID for ElevenLabs.
        porcupine_access_key: API key for Porcupine wake word.
        whisper_model: Faster-Whisper model size for command transcription.
        sample_rate: Audio sample rate in Hz.
        max_command_duration: Max seconds to record a command.
        silence_timeout: Seconds of silence before ending command recording.
        greeting: Optional text to speak on startup.
    """

    def __init__(
        self,
        wake_word: str = "orion",
        on_command: Callable[[str], str] | None = None,
        wake_word_backend: WakeWordBackend = WakeWordBackend.VAD_WHISPER,
        tts_backend: TTSBackend = TTSBackend.LOCAL,
        elevenlabs_api_key: str | None = None,
        elevenlabs_voice_id: str | None = None,
        porcupine_access_key: str | None = None,
        whisper_model: str = "base.en",
        sample_rate: int = 16000,
        max_command_duration: float = 10.0,
        silence_timeout: float = 1.5,
        greeting: str | None = None,
    ) -> None:
        self.wake_word = wake_word
        self.on_command = on_command or self._default_handler
        self.sample_rate = sample_rate
        self.max_command_duration = max_command_duration
        self.silence_timeout = silence_timeout
        self.greeting = greeting

        self._running = False
        self._wake_detected = False

        # Initialize components
        self._mic = MicrophoneListener(sample_rate=sample_rate)

        self._wake_detector = WakeWordDetector(
            wake_word=wake_word,
            backend=wake_word_backend,
            porcupine_access_key=porcupine_access_key,
            sample_rate=sample_rate,
        )

        # TTS
        tts_kwargs: dict = {"backend": tts_backend}
        if tts_backend == TTSBackend.ELEVENLABS:
            tts_kwargs["api_key"] = elevenlabs_api_key
            tts_kwargs["voice_id"] = elevenlabs_voice_id
        self._tts = TTSEngine(**tts_kwargs)

        # Command transcriber (loaded lazy)
        self._transcriber = None
        self._whisper_model = whisper_model

    def run(self) -> None:
        """Start the assistant and block forever.

        Listens for the wake word, records commands, transcribes,
        dispatches to handler, and speaks responses.

        Press Ctrl+C to stop.
        """
        self._running = True

        # Graceful shutdown on Ctrl+C
        def _signal_handler(sig: int, frame: object) -> None:
            print("\nShutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)

        # Load transcriber
        self._load_transcriber()

        # Start mic
        self._mic.start()

        print(f"VoiceKit Assistant active — wake word: \"{self.wake_word}\"")
        print("Listening... (Ctrl+C to stop)\n")

        if self.greeting:
            self._tts.speak_sync(self.greeting)

        try:
            self._main_loop()
        finally:
            self.stop()

    def run_async(self) -> None:
        """Start the assistant in the background (non-blocking).

        Call stop() to shut down.
        """
        import threading

        self._running = True
        self._load_transcriber()
        self._mic.start()

        thread = threading.Thread(
            target=self._main_loop, daemon=True, name="voicekit-assistant"
        )
        thread.start()

        if self.greeting:
            self._tts.speak_sync(self.greeting)

    def stop(self) -> None:
        """Stop the assistant and release all resources."""
        self._running = False
        self._mic.stop()
        self._tts.stop()
        self._wake_detector.cleanup()

    def _main_loop(self) -> None:
        """Core assistant loop."""
        while self._running:
            frame = self._mic.read_frame(timeout=0.5)
            if frame is None:
                continue

            # Check for wake word
            if self._wake_detector.process_frame(frame):
                self._handle_activation()

    def _handle_activation(self) -> None:
        """Handle wake word detection — record, transcribe, respond."""
        print(f"[{self.wake_word}] Wake word detected! Listening for command...")

        # Record the user's command
        utterance = self._mic.record_utterance(
            max_duration=self.max_command_duration,
            silence_timeout=self.silence_timeout,
        )

        if len(utterance) < self.sample_rate * 0.3:
            print("[...] Too short, ignoring.")
            return

        # Transcribe
        print("[...] Transcribing...")
        try:
            command_text = self._transcriber.transcribe_numpy(
                utterance, self.sample_rate
            )
        except Exception as e:
            print(f"[!] Transcription failed: {e}")
            return

        if not command_text.strip():
            print("[...] No speech detected.")
            return

        print(f"[cmd] \"{command_text}\"")

        # Dispatch to handler
        try:
            response = self.on_command(command_text)
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"
            print(f"[!] Handler error: {e}")

        if response:
            print(f"[tts] \"{response}\"")
            self._tts.speak_sync(response)

        # Reset wake word detector state
        self._wake_detector.reset()

    def _load_transcriber(self) -> None:
        """Lazy-load the command transcriber."""
        if self._transcriber is not None:
            return

        from voicekit.core.transcriber import LocalTranscriber
        self._transcriber = LocalTranscriber(
            model_size=self._whisper_model,
            beam_size=5,
        )

    @staticmethod
    def _default_handler(text: str) -> str:
        """Default command handler — echoes the command."""
        return f"I heard you say: {text}"
