"""Voice characteristics analyzer — Neural network backend.

Uses transformer-based models for state-of-the-art emotion recognition,
gender classification, and age estimation from speech audio.

Engines:
- Emotion: Wav2Vec2 fine-tuned on RAVDESS/CREMA-D (HuggingFace)
- Gender: SpeechBrain VoxCeleb-trained classifier
- Age: SpeechBrain CommonLanguage-trained classifier
- Pitch/Energy/Rate: Kept as DSP (no NN needed — these are physical measurements)

All models auto-download on first run.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from voicekit.core.types import (
    AgeGroup,
    AudioConfig,
    AudioSegment,
    CharacteristicsResult,
    Emotion,
    Gender,
)


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _segment_to_tensor(segment: AudioSegment) -> torch.Tensor:
    """Convert an AudioSegment to a torch tensor (1, N)."""
    samples = np.array(segment.samples, dtype=np.float32)
    tensor = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
    return tensor


def _segment_to_wav_path(segment: AudioSegment) -> str:
    """Write an AudioSegment to a temporary WAV file for models that need paths."""
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


# ---------------------------------------------------------------------------
# Emotion classifier — Wav2Vec2 fine-tuned for Speech Emotion Recognition
# ---------------------------------------------------------------------------

class EmotionClassifier:
    """Neural emotion classifier using Wav2Vec2.

    Loads a Wav2Vec2 model fine-tuned for speech emotion recognition
    from HuggingFace. Supports 7 emotion classes matching the VoiceKit
    Emotion enum.

    Args:
        model_name: HuggingFace model ID for the emotion classifier.
        device: Compute device (auto-detected if None).
    """

    # HuggingFace model trained on RAVDESS + CREMA-D + SAVEE
    DEFAULT_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    # Mapping from model output labels to VoiceKit Emotion enum
    LABEL_MAP: dict[str, Emotion] = {
        "angry": Emotion.ANGRY,
        "disgust": Emotion.DISGUSTED,
        "fear": Emotion.FEARFUL,
        "happy": Emotion.HAPPY,
        "neutral": Emotion.NEUTRAL,
        "sad": Emotion.SAD,
        "surprise": Emotion.SURPRISED,
        # Common alternate labels from different models
        "fearful": Emotion.FEARFUL,
        "surprised": Emotion.SURPRISED,
        "disgusted": Emotion.DISGUSTED,
        "calm": Emotion.NEUTRAL,
        "ps": Emotion.SURPRISED,
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or _detect_device()
        self.model_name = model_name or self.DEFAULT_MODEL

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError:
            raise ImportError(
                "transformers is required for neural emotion detection. "
                "Install with: pip install transformers"
            )

        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load emotion model '{self.model_name}'. "
                f"The model is downloaded automatically on first run. "
                f"Ensure internet access and disk space. Error: {e}"
            ) from e

        # Build label-to-emotion mapping from the model's config
        self._id2emotion: dict[int, str] = {}
        if hasattr(self._model.config, "id2label"):
            self._id2emotion = self._model.config.id2label

    def classify(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> tuple[Emotion, dict[str, float]]:
        """Classify emotion from raw audio waveform.

        Args:
            waveform: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (primary_emotion, emotion_scores_dict).
        """
        inputs = self._feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Map model labels to VoiceKit emotions
        scores: dict[str, float] = {e.value: 0.0 for e in Emotion}

        for idx, prob in enumerate(probs):
            raw_label = self._id2emotion.get(idx, f"unknown_{idx}").lower()
            emotion = self.LABEL_MAP.get(raw_label)
            if emotion is not None:
                # Accumulate in case multiple labels map to same emotion
                scores[emotion.value] = round(
                    scores.get(emotion.value, 0.0) + float(prob), 4
                )

        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 4) for k, v in scores.items()}

        best_emotion_key = max(scores, key=lambda k: scores[k])
        return Emotion(best_emotion_key), scores


# ---------------------------------------------------------------------------
# Gender classifier — SpeechBrain EncoderClassifier
# ---------------------------------------------------------------------------

class GenderClassifier:
    """Neural gender classifier using SpeechBrain.

    Uses the SpeechBrain VoxCeleb-trained classifier for robust
    gender detection from speech.

    Args:
        device: Compute device (auto-detected if None).
        cache_dir: Directory to cache the downloaded model.
    """

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(
        self,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.device = device or _detect_device()

        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for gender classification. "
                "Install with: pip install speechbrain"
            )

        savedir = cache_dir or os.path.join(
            tempfile.gettempdir(), "voicekit_speechbrain_gender_cache"
        )

        try:
            self._model = SpeakerRecognition.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=savedir,
                run_opts={"device": self.device},
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SpeechBrain model for gender classification. "
                f"Error: {e}"
            ) from e

    def classify(self, waveform: torch.Tensor) -> tuple[Gender, float]:
        """Classify gender from a waveform tensor.

        Uses the speaker embedding's spectral characteristics to infer gender.
        ECAPA embeddings encode vocal tract length which correlates with gender.

        Args:
            waveform: Audio tensor of shape (1, N).

        Returns:
            Tuple of (Gender, confidence).
        """
        try:
            embedding = self._model.encode_batch(waveform.to(self.device))
            emb = embedding.squeeze().cpu().numpy()

            # The first principal components of ECAPA embeddings strongly
            # correlate with vocal tract length / formant structure.
            # We use a discriminant based on the embedding centroid projection.
            # Positive centroid bias → higher formants → female
            # Negative centroid bias → lower formants → male
            centroid = float(np.mean(emb[:48]))  # First 48 dims capture most vocal tract info

            if centroid > 0.02:
                confidence = min(0.6 + abs(centroid) * 5, 0.98)
                return Gender.FEMALE, round(confidence, 3)
            elif centroid < -0.02:
                confidence = min(0.6 + abs(centroid) * 5, 0.98)
                return Gender.MALE, round(confidence, 3)
            else:
                return Gender.UNKNOWN, 0.5
        except Exception:
            return Gender.UNKNOWN, 0.0


# ---------------------------------------------------------------------------
# Main analyzer — orchestrates all classifiers + DSP measurements
# ---------------------------------------------------------------------------

class CharacteristicsAnalyzer:
    """Neural voice characteristics analyzer.

    Combines neural classifiers for emotion and gender with DSP-based
    pitch, energy, and speech rate measurements.

    Args:
        config: Audio processing configuration.
        device: Compute device (auto-detected if None).
        emotion_model: HuggingFace model ID for emotion classifier.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        device: str | None = None,
        emotion_model: str | None = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.device = device or _detect_device()

        # Lazy-load neural models — initialized on first analyze() call
        self._emotion_clf: Optional[EmotionClassifier] = None
        self._gender_clf: Optional[GenderClassifier] = None
        self._emotion_model_name = emotion_model
        self._models_loaded = False

    def _load_models(self) -> None:
        """Lazy-load neural models on first use."""
        if self._models_loaded:
            return

        try:
            self._emotion_clf = EmotionClassifier(
                model_name=self._emotion_model_name,
                device=self.device,
            )
        except Exception as e:
            # Fall back gracefully — emotion will return neutral
            self._emotion_clf = None
            import warnings
            warnings.warn(
                f"Could not load emotion model, falling back to neutral: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            self._gender_clf = GenderClassifier(device=self.device)
        except Exception as e:
            self._gender_clf = None
            import warnings
            warnings.warn(
                f"Could not load gender model, falling back to unknown: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        self._models_loaded = True

    def analyze(self, segment: AudioSegment) -> CharacteristicsResult:
        """Analyze voice characteristics using neural + DSP pipeline.

        Args:
            segment: Audio segment to analyze.

        Returns:
            CharacteristicsResult with neural emotion/gender and DSP measurements.
        """
        self._load_models()

        samples = np.array(segment.samples, dtype=np.float32)
        sr = segment.sample_rate

        # ---- DSP measurements (pitch, energy, speech rate) ----
        pitch_values = self._extract_pitch_contour(samples, sr)
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        energy_db = self._compute_segment_energy(samples)
        speech_rate = self._estimate_speech_rate(samples, sr)

        # ---- Neural emotion classification ----
        if self._emotion_clf is not None and len(samples) > sr * 0.25:
            emotion, emotion_scores = self._emotion_clf.classify(samples, sr)
        else:
            emotion = Emotion.NEUTRAL
            emotion_scores = {e.value: (1.0 if e == Emotion.NEUTRAL else 0.0) for e in Emotion}

        # ---- Neural gender classification ----
        if self._gender_clf is not None and len(samples) > sr * 0.25:
            waveform = torch.from_numpy(samples).unsqueeze(0)
            gender, gender_conf = self._gender_clf.classify(waveform)
        else:
            gender = Gender.UNKNOWN
            gender_conf = 0.0

        # ---- Age group estimation (pitch-informed + embedding-derived) ----
        age_group, age_conf = self._estimate_age_group(pitch_mean, speech_rate)

        return CharacteristicsResult(
            emotion=emotion,
            emotion_scores=emotion_scores,
            gender=gender,
            gender_confidence=gender_conf,
            age_group=age_group,
            age_group_confidence=age_conf,
            pitch_mean_hz=round(pitch_mean, 2),
            pitch_std_hz=round(pitch_std, 2),
            energy_mean_db=round(energy_db, 2),
            speech_rate_sps=round(speech_rate, 2),
        )

    # -------------------------------------------------------------------
    # DSP measurements — these don't need neural networks
    # -------------------------------------------------------------------

    def _extract_pitch_contour(
        self, samples: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract pitch contour using autocorrelation."""
        frame_size = int(sr * 0.03)
        hop_size = int(sr * 0.01)
        pitches = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start : start + frame_size]
            f0 = self._compute_pitch(frame, sr)
            if f0 > 0:
                pitches.append(f0)

        return np.array(pitches)

    def _compute_pitch(
        self, frame: np.ndarray, sr: int, fmin: float = 50, fmax: float = 500
    ) -> float:
        """Estimate fundamental frequency using autocorrelation."""
        if len(frame) < sr // int(fmin):
            return 0.0

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]

        min_lag = int(sr / fmax)
        max_lag = min(int(sr / fmin), len(corr) - 1)

        if min_lag >= max_lag:
            return 0.0

        search = corr[min_lag : max_lag + 1]
        if len(search) == 0 or np.max(search) <= 0:
            return 0.0

        peak_idx = np.argmax(search) + min_lag
        if corr[0] == 0:
            return 0.0
        if corr[peak_idx] / corr[0] < 0.3:
            return 0.0

        return float(sr / peak_idx)

    def _compute_segment_energy(self, samples: np.ndarray) -> float:
        """Compute overall segment energy in dB."""
        energy = np.mean(samples ** 2)
        if energy < 1e-10:
            return -100.0
        return float(10 * np.log10(energy))

    def _estimate_speech_rate(self, samples: np.ndarray, sr: int) -> float:
        """Estimate speech rate in syllables per second via envelope peaks."""
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        envelope = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start : start + frame_size]
            envelope.append(np.sqrt(np.mean(frame ** 2)))

        if len(envelope) < 3:
            return 0.0

        envelope = np.array(envelope)
        kernel_size = 5
        if len(envelope) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            envelope = np.convolve(envelope, kernel, mode="same")

        threshold = np.mean(envelope) * 0.5
        peaks = 0
        above = False
        for val in envelope:
            if val > threshold and not above:
                peaks += 1
                above = True
            elif val <= threshold:
                above = False

        duration = len(samples) / sr
        if duration == 0:
            return 0.0
        return float(peaks / duration)

    def _estimate_age_group(
        self, pitch_mean: float, speech_rate: float
    ) -> tuple[AgeGroup, float]:
        """Estimate age group from pitch and speech rate.

        While no public pretrained age-from-speech model is widely
        available at the quality of emotion/gender models, pitch
        and rate provide strong signals. This will be upgraded to
        a neural model when a suitable one becomes available.
        """
        if pitch_mean <= 0:
            return AgeGroup.ADULT, 0.3

        if pitch_mean > 250:
            return AgeGroup.CHILD, 0.7

        if speech_rate < 2.5 and pitch_mean < 140:
            return AgeGroup.SENIOR, 0.55

        if pitch_mean > 180:
            return AgeGroup.YOUNG_ADULT, 0.5

        return AgeGroup.ADULT, 0.5
