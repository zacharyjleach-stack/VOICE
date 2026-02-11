"""Speaker detection and biometric verification — SpeechBrain ECAPA-TDNN backend.

Uses the SpeechBrain ECAPA-TDNN model (pretrained on VoxCeleb) for
state-of-the-art speaker verification. Replaces the legacy MFCC-based
heuristics with a production-grade neural speaker encoder.

The model (~80MB) is downloaded automatically on first run.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from voicekit.core.types import AudioConfig, AudioSegment, SpeakerResult


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _segment_to_wav_path(segment: AudioSegment) -> str:
    """Write an AudioSegment to a temporary WAV file.

    SpeechBrain's verification API expects file paths, so we
    materialize in-memory audio to a temp file.

    Returns:
        Path to the temporary WAV file.
    """
    samples = np.array(segment.samples, dtype=np.float32)
    int_samples = (samples * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(segment.sample_rate)
            wf.writeframes(int_samples.tobytes())
    except Exception:
        os.unlink(tmp.name)
        raise
    return tmp.name


class VoiceBiometrics:
    """Neural speaker verification powered by SpeechBrain ECAPA-TDNN.

    Uses the pretrained `speechbrain/spkrec-ecapa-voxceleb` model for
    speaker embedding extraction and verification. Embeddings are
    192-dimensional and L2-normalized.

    Args:
        device: Compute device ("cpu", "cuda", "mps", or None for auto).
        threshold: Cosine similarity threshold for verification.
            Default 0.25 (SpeechBrain's recommended threshold for ECAPA).
        cache_dir: Directory to cache the downloaded model.
    """

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    EMBEDDING_DIM = 192

    def __init__(
        self,
        device: str | None = None,
        threshold: float = 0.25,
        cache_dir: str | None = None,
    ) -> None:
        self.device = device or _detect_device()
        self.threshold = threshold
        self._enrolled: dict[str, torch.Tensor] = {}

        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for speaker verification. "
                "Install with: pip install speechbrain"
            )

        savedir = cache_dir or os.path.join(
            tempfile.gettempdir(), "voicekit_speechbrain_cache"
        )

        try:
            self._model = SpeakerRecognition.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=savedir,
                run_opts={"device": self.device},
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SpeechBrain ECAPA-TDNN model. "
                f"The model (~80MB) is downloaded automatically on first run. "
                f"Ensure internet access and disk space. Error: {e}"
            ) from e

    def verify_speaker(
        self,
        incoming_audio_path: str | Path,
        reference_audio_path: str | Path,
    ) -> tuple[bool, float]:
        """Verify if two audio files are from the same speaker.

        Args:
            incoming_audio_path: Path to the audio to verify.
            reference_audio_path: Path to the reference/enrolled audio.

        Returns:
            Tuple of (is_match, confidence_score).
            is_match is True if similarity exceeds the threshold.
            confidence_score is the cosine similarity (-1.0 to 1.0).
        """
        try:
            score, prediction = self._model.verify_files(
                str(incoming_audio_path),
                str(reference_audio_path),
            )
        except Exception as e:
            raise RuntimeError(
                f"Speaker verification failed: {e}"
            ) from e

        similarity = float(score.item())
        is_match = similarity >= self.threshold

        return is_match, round(similarity, 4)

    def extract_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Extract a 192-dim speaker embedding from an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            L2-normalized numpy array of shape (192,).
        """
        try:
            embedding = self._model.encode_batch(
                self._load_audio_tensor(audio_path)
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding extraction failed for '{audio_path}': {e}"
            ) from e

        emb = embedding.squeeze().cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def extract_embedding_from_segment(
        self, segment: AudioSegment
    ) -> np.ndarray:
        """Extract speaker embedding from an in-memory AudioSegment.

        Args:
            segment: AudioSegment with audio data.

        Returns:
            L2-normalized numpy array of shape (192,).
        """
        tmp_path = _segment_to_wav_path(segment)
        try:
            return self.extract_embedding(tmp_path)
        finally:
            os.unlink(tmp_path)

    def enroll(self, speaker_id: str, audio_path: str | Path) -> np.ndarray:
        """Enroll a speaker for later identification.

        Args:
            speaker_id: Unique label for the speaker.
            audio_path: Path to enrollment audio.

        Returns:
            Speaker embedding vector.
        """
        embedding = self.extract_embedding(audio_path)
        self._enrolled[speaker_id] = torch.from_numpy(embedding)
        return embedding

    def enroll_from_segment(
        self, speaker_id: str, segment: AudioSegment
    ) -> np.ndarray:
        """Enroll a speaker from an in-memory AudioSegment.

        Args:
            speaker_id: Unique label for the speaker.
            segment: AudioSegment with speaker audio.

        Returns:
            Speaker embedding vector.
        """
        embedding = self.extract_embedding_from_segment(segment)
        self._enrolled[speaker_id] = torch.from_numpy(embedding)
        return embedding

    def identify(self, audio_path: str | Path) -> tuple[Optional[str], float]:
        """Identify a speaker from enrolled speakers.

        Args:
            audio_path: Path to audio to identify.

        Returns:
            Tuple of (speaker_id, confidence). speaker_id is None if
            no enrolled speaker matches above the threshold.
        """
        if not self._enrolled:
            return None, 0.0

        embedding = torch.from_numpy(self.extract_embedding(audio_path))

        best_id: Optional[str] = None
        best_score = -1.0

        for speaker_id, enrolled_emb in self._enrolled.items():
            score = float(
                torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    enrolled_emb.unsqueeze(0),
                ).item()
            )
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_score >= self.threshold:
            return best_id, round(best_score, 4)
        return None, round(best_score, 4)

    def detect(self, segment: AudioSegment) -> SpeakerResult:
        """Generate speaker embedding and ID from an AudioSegment.

        Maintains backward compatibility with the original SDK interface.

        Args:
            segment: Audio segment to analyze.

        Returns:
            SpeakerResult with neural embedding and optional identification.
        """
        embedding = self.extract_embedding_from_segment(segment)

        # Try identification if speakers are enrolled
        speaker_id: Optional[str] = None
        confidence = 0.0

        if self._enrolled:
            emb_tensor = torch.from_numpy(embedding)
            best_id = None
            best_score = -1.0

            for sid, enrolled_emb in self._enrolled.items():
                score = float(
                    torch.nn.functional.cosine_similarity(
                        emb_tensor.unsqueeze(0),
                        enrolled_emb.unsqueeze(0),
                    ).item()
                )
                if score > best_score:
                    best_score = score
                    best_id = sid

            if best_score >= self.threshold:
                speaker_id = best_id
            confidence = best_score

        return SpeakerResult(
            embedding=embedding.tolist(),
            speaker_count=1,
            speaker_id=speaker_id,
            confidence=round(confidence, 4),
        )

    def _load_audio_tensor(self, audio_path: str | Path) -> torch.Tensor:
        """Load audio file as a torch tensor for SpeechBrain."""
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio '{audio_path}': {e}"
            ) from e


# Backward-compatible alias so existing SDK imports keep working
SpeakerDetector = VoiceBiometrics
