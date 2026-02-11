"""Text-to-Speech engine with ElevenLabs and local fallback.

Supports two backends:
1. ElevenLabs — studio-quality AI voice (Jarvis/Stark feel)
2. pyttsx3 — fully offline fallback (no API key needed)

The engine queues speech requests and plays them sequentially
through the system audio output.
"""

from __future__ import annotations

import io
import queue
import tempfile
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class TTSBackend(str, Enum):
    """Available TTS backends."""
    ELEVENLABS = "elevenlabs"
    LOCAL = "local"


class TTSEngine:
    """Text-to-Speech engine for voice assistant output.

    In ELEVENLABS mode, streams audio from the ElevenLabs API
    for studio-quality speech. Requires an API key.

    In LOCAL mode, uses pyttsx3 for fully offline speech
    synthesis — no internet or API key required.

    Args:
        backend: TTS backend to use.
        api_key: ElevenLabs API key (required for ElevenLabs backend).
        voice_id: ElevenLabs voice ID. Default is "Antoni" (Stark-like).
        model_id: ElevenLabs model. "eleven_turbo_v2" for lowest latency.
        speaking_rate: Speech rate multiplier (1.0 = normal).
        on_start: Callback fired when speech begins.
        on_end: Callback fired when speech finishes.
    """

    # ElevenLabs voices good for a Jarvis/assistant feel
    VOICE_PRESETS = {
        "jarvis": "ErXwobaYiN019PkySvjV",   # Antoni — British, clear
        "stark": "VR6AewLTigWG4xSOukaG",     # Arnold — confident, deep
        "friday": "EXAVITQu4vr4xnSDxMaL",    # Bella — warm, female
        "orion": "ErXwobaYiN019PkySvjV",      # Antoni — default
    }

    def __init__(
        self,
        backend: TTSBackend = TTSBackend.ELEVENLABS,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = "eleven_turbo_v2",
        speaking_rate: float = 1.0,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.speaking_rate = speaking_rate
        self.on_start = on_start
        self.on_end = on_end

        self._speech_queue: queue.Queue[str | None] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        if backend == TTSBackend.ELEVENLABS:
            self._init_elevenlabs(api_key, voice_id)
        else:
            self._init_local()

    def _init_elevenlabs(
        self, api_key: str | None, voice_id: str | None
    ) -> None:
        """Initialize ElevenLabs TTS client."""
        if not api_key:
            raise ValueError(
                "ElevenLabs requires an API key. "
                "Get one at https://elevenlabs.io/. "
                "Pass it as api_key= or set ELEVENLABS_API_KEY env var."
            )

        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            raise ImportError(
                "elevenlabs is required for ElevenLabs TTS. "
                "Install with: pip install elevenlabs"
            )

        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id or self.VOICE_PRESETS.get("orion", voice_id)

    def _init_local(self) -> None:
        """Initialize local pyttsx3 TTS engine."""
        try:
            import pyttsx3
        except ImportError:
            raise ImportError(
                "pyttsx3 is required for local TTS. "
                "Install with: pip install pyttsx3"
            )

        self._local_engine = pyttsx3.init()
        self._local_engine.setProperty("rate", int(175 * self.speaking_rate))

    def speak(self, text: str) -> None:
        """Queue text to be spoken.

        Non-blocking — the text is added to the speech queue and
        played by the background worker thread.

        Args:
            text: Text to speak aloud.
        """
        self._ensure_worker()
        self._speech_queue.put(text)

    def speak_sync(self, text: str) -> None:
        """Speak text and block until playback completes.

        Args:
            text: Text to speak aloud.
        """
        if self.on_start:
            self.on_start()

        if self.backend == TTSBackend.ELEVENLABS:
            self._speak_elevenlabs(text)
        else:
            self._speak_local(text)

        if self.on_end:
            self.on_end()

    def speak_to_file(self, text: str, output_path: str | Path) -> Path:
        """Generate speech audio and save to a file.

        Args:
            text: Text to synthesize.
            output_path: Path to save the audio file.

        Returns:
            Path to the saved audio file.
        """
        output_path = Path(output_path)

        if self.backend == TTSBackend.ELEVENLABS:
            audio_data = self._generate_elevenlabs(text)
            output_path.write_bytes(audio_data)
        else:
            # pyttsx3 can save to file
            self._local_engine.save_to_file(text, str(output_path))
            self._local_engine.runAndWait()

        return output_path

    def stop(self) -> None:
        """Stop the speech worker and clear the queue."""
        self._running = False
        self._speech_queue.put(None)  # Poison pill

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        # Drain queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    def _ensure_worker(self) -> None:
        """Start the background speech worker if not running."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="voicekit-tts"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background thread: process speech queue sequentially."""
        while self._running:
            try:
                text = self._speech_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                self.speak_sync(text)
            except Exception:
                pass  # Don't let one failure kill the worker

    def _speak_elevenlabs(self, text: str) -> None:
        """Synthesize and play speech via ElevenLabs streaming."""
        try:
            audio_bytes = self._generate_elevenlabs(text)
            self._play_audio_bytes(audio_bytes)
        except Exception as e:
            raise RuntimeError(f"ElevenLabs speech failed: {e}") from e

    def _generate_elevenlabs(self, text: str) -> bytes:
        """Generate audio bytes from ElevenLabs API."""
        audio_iter = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )

        # Collect streaming chunks
        chunks = []
        for chunk in audio_iter:
            if chunk:
                chunks.append(chunk)

        return b"".join(chunks)

    def _speak_local(self, text: str) -> None:
        """Speak using the local pyttsx3 engine."""
        self._local_engine.say(text)
        self._local_engine.runAndWait()

    def _play_audio_bytes(self, audio_bytes: bytes) -> None:
        """Play raw audio bytes through the system speaker."""
        # Save to temp file and play
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(audio_bytes)
        tmp.close()

        try:
            import platform
            import subprocess

            system = platform.system()
            if system == "Darwin":
                # macOS
                subprocess.run(
                    ["afplay", tmp.name],
                    check=True,
                    capture_output=True,
                )
            elif system == "Linux":
                # Linux — try mpv, then ffplay, then aplay
                for player in ["mpv --no-video", "ffplay -nodisp -autoexit", "aplay"]:
                    try:
                        parts = player.split() + [tmp.name]
                        subprocess.run(
                            parts,
                            check=True,
                            capture_output=True,
                        )
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif system == "Windows":
                # Windows — use PowerShell's media player
                subprocess.run(
                    [
                        "powershell",
                        "-c",
                        f'(New-Object Media.SoundPlayer "{tmp.name}").PlaySync()',
                    ],
                    check=True,
                    capture_output=True,
                )
        finally:
            import os
            os.unlink(tmp.name)

    def __enter__(self) -> TTSEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
