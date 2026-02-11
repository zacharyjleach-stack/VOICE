"""Always-on microphone listener.

Captures audio from the system microphone in a background thread,
feeds frames to a callback (wake word detector, VAD, etc.), and
records full utterances after the wake word triggers.

Designed for <2% idle CPU on Mac Studio / modern hardware.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np


class MicrophoneListener:
    """Background microphone capture with callback-driven processing.

    Opens the default system microphone in a non-blocking background
    thread and delivers audio frames to registered callbacks.

    Args:
        sample_rate: Capture sample rate in Hz (16000 recommended).
        frame_duration_ms: Duration of each audio frame in ms.
        channels: Number of audio channels (1 = mono).
        device_index: PyAudio device index, or None for default mic.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        channels: int = 1,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.device_index = device_index

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        self._stream = None
        self._audio = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    def on_frame(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback to receive each audio frame.

        The callback receives a float32 numpy array of shape (frame_size,).
        Callbacks run in the listener thread — keep them fast.

        Args:
            callback: Function called with each audio frame.
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start capturing audio from the microphone.

        Opens the mic and begins delivering frames to callbacks
        in a background thread.
        """
        if self._running:
            return

        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "PyAudio is required for microphone capture. "
                "Install with: pip install pyaudio"
            )

        self._audio = pyaudio.PyAudio()

        stream_kwargs = {
            "format": pyaudio.paFloat32,
            "channels": self.channels,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": self.frame_size,
        }
        if self.device_index is not None:
            stream_kwargs["input_device_index"] = self.device_index

        try:
            self._stream = self._audio.open(**stream_kwargs)
        except Exception as e:
            self._audio.terminate()
            raise RuntimeError(
                f"Failed to open microphone. Check that a mic is connected "
                f"and permissions are granted. Error: {e}"
            ) from e

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="voicekit-mic"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and release the microphone."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read a single frame from the capture queue.

        Useful for pull-based processing instead of callbacks.

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            Float32 numpy array, or None if timeout.
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_utterance(
        self,
        max_duration: float = 10.0,
        silence_timeout: float = 1.5,
        energy_threshold: float = -35.0,
    ) -> np.ndarray:
        """Record a complete utterance (speech until silence).

        Call this after a wake word triggers to capture the user's command.
        Blocks until the user stops speaking or max_duration is hit.

        Args:
            max_duration: Maximum recording duration in seconds.
            silence_timeout: Seconds of silence before stopping.
            energy_threshold: Frame energy (dB) below which = silence.

        Returns:
            Complete utterance as float32 numpy array.
        """
        frames: list[np.ndarray] = []
        silence_frames = 0
        max_silence_frames = int(
            silence_timeout / (self.frame_duration_ms / 1000)
        )
        max_frames = int(max_duration / (self.frame_duration_ms / 1000))

        for _ in range(max_frames):
            frame = self.read_frame(timeout=0.5)
            if frame is None:
                continue

            frames.append(frame)

            # Check energy
            energy = np.mean(frame ** 2)
            energy_db = 10 * np.log10(energy) if energy > 1e-10 else -100
            if energy_db < energy_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= max_silence_frames and len(frames) > 10:
                break

        if not frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(frames)

    def _capture_loop(self) -> None:
        """Background thread: read mic → queue + callbacks."""
        while self._running:
            try:
                raw = self._stream.read(
                    self.frame_size, exception_on_overflow=False
                )
                frame = np.frombuffer(raw, dtype=np.float32).copy()

                # Push to queue for pull-based consumers
                try:
                    self._audio_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame to prevent memory buildup
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._audio_queue.put_nowait(frame)

                # Deliver to callbacks
                for cb in self._callbacks:
                    try:
                        cb(frame)
                    except Exception:
                        pass  # Don't let a bad callback kill the capture loop

            except Exception:
                if self._running:
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                break

    @property
    def is_running(self) -> bool:
        """Whether the listener is actively capturing."""
        return self._running

    def __enter__(self) -> MicrophoneListener:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
