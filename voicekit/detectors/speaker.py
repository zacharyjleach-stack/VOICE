"""Speaker detection, embedding, and identification engine.

Generates speaker embeddings that can be used for:
- Speaker verification (is this the same person?)
- Speaker identification (who is speaking?)
- Speaker diarization (segment by speaker)
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from voicekit.core.audio import AudioEngine
from voicekit.core.types import AudioConfig, AudioSegment, SpeakerResult


class SpeakerDetector:
    """Generates speaker embeddings and performs speaker identification.

    Uses MFCC-based statistical features to create speaker-discriminative
    embeddings. Supports enrollment-based identification against a speaker
    database.
    """

    def __init__(
        self, config: AudioConfig | None = None, engine: AudioEngine | None = None
    ) -> None:
        self.config = config or AudioConfig()
        self.engine = engine or AudioEngine(self.config)
        self._enrolled: dict[str, np.ndarray] = {}

    def detect(self, segment: AudioSegment) -> SpeakerResult:
        """Generate speaker embedding and optionally identify the speaker.

        Args:
            segment: Audio segment to analyze.

        Returns:
            SpeakerResult with embedding and optional speaker identification.
        """
        embedding = self.extract_embedding(segment)
        speaker_id, confidence = self._identify(embedding)

        return SpeakerResult(
            embedding=embedding.tolist(),
            speaker_count=1,
            speaker_id=speaker_id,
            confidence=confidence,
        )

    def extract_embedding(self, segment: AudioSegment) -> np.ndarray:
        """Extract a fixed-length speaker embedding from audio.

        The embedding captures speaker-discriminative features derived
        from MFCC statistics (means, variances, deltas).

        Args:
            segment: Audio segment to process.

        Returns:
            Numpy array of shape (embedding_dim,).
        """
        mfccs = self.engine.extract_mfcc(segment, n_mfcc=20)

        if mfccs.shape[1] == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Statistical features from MFCCs
        features = []

        # Mean and std of each coefficient
        features.append(np.mean(mfccs, axis=1))
        features.append(np.std(mfccs, axis=1))

        # Delta features (first derivative)
        if mfccs.shape[1] > 2:
            deltas = np.diff(mfccs, axis=1)
            features.append(np.mean(deltas, axis=1))
            features.append(np.std(deltas, axis=1))

            # Delta-delta features (second derivative)
            if deltas.shape[1] > 2:
                delta_deltas = np.diff(deltas, axis=1)
                features.append(np.mean(delta_deltas, axis=1))
                features.append(np.std(delta_deltas, axis=1))

        # Skewness and kurtosis for more discriminative power
        features.append(self._safe_skewness(mfccs))
        features.append(self._safe_kurtosis(mfccs))

        raw_embedding = np.concatenate(features)

        # Project to target dimension
        embedding = self._project_to_dim(
            raw_embedding, self.config.embedding_dim
        )

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def enroll(self, speaker_id: str, segment: AudioSegment) -> np.ndarray:
        """Enroll a speaker for later identification.

        Args:
            speaker_id: Unique identifier for the speaker.
            segment: Audio sample of the speaker.

        Returns:
            The enrolled speaker's embedding.
        """
        embedding = self.extract_embedding(segment)
        self._enrolled[speaker_id] = embedding
        return embedding

    def verify(
        self, segment: AudioSegment, claimed_id: str
    ) -> tuple[bool, float]:
        """Verify if audio matches a claimed speaker identity.

        Args:
            segment: Audio segment to verify.
            claimed_id: Speaker ID to verify against.

        Returns:
            Tuple of (is_match, similarity_score).
        """
        if claimed_id not in self._enrolled:
            return False, 0.0

        embedding = self.extract_embedding(segment)
        enrolled = self._enrolled[claimed_id]
        similarity = self._cosine_similarity(embedding, enrolled)

        # Threshold for verification
        threshold = 0.75
        return similarity >= threshold, float(similarity)

    def compare(
        self, segment_a: AudioSegment, segment_b: AudioSegment
    ) -> float:
        """Compare two audio segments and return speaker similarity.

        Args:
            segment_a: First audio segment.
            segment_b: Second audio segment.

        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        emb_a = self.extract_embedding(segment_a)
        emb_b = self.extract_embedding(segment_b)
        return float(self._cosine_similarity(emb_a, emb_b))

    def _identify(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        """Identify a speaker from enrolled speakers."""
        if not self._enrolled:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for speaker_id, enrolled_emb in self._enrolled.items():
            score = self._cosine_similarity(embedding, enrolled_emb)
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_score >= 0.75:
            return best_id, float(best_score)
        return None, float(best_score)

    def _project_to_dim(
        self, features: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """Deterministically project features to target dimensionality.

        Uses a seeded random projection matrix for consistent embeddings.
        """
        if len(features) == target_dim:
            return features

        # Deterministic projection using feature-derived seed
        seed_bytes = hashlib.sha256(
            np.array(features.shape).tobytes()
        ).digest()
        seed = int.from_bytes(seed_bytes[:4], "big") % (2**31)
        rng = np.random.RandomState(seed)

        if len(features) > target_dim:
            # Random projection to lower dimension
            proj_matrix = rng.randn(target_dim, len(features)).astype(
                np.float32
            )
            proj_matrix /= np.sqrt(len(features))
            return proj_matrix @ features
        else:
            # Pad and project
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: len(features)] = features
            return padded

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _safe_skewness(data: np.ndarray) -> np.ndarray:
        """Compute skewness along axis 1, handling edge cases."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)
        return np.mean(((data - mean) / std) ** 3, axis=1)

    @staticmethod
    def _safe_kurtosis(data: np.ndarray) -> np.ndarray:
        """Compute kurtosis along axis 1, handling edge cases."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)
        return np.mean(((data - mean) / std) ** 4, axis=1) - 3
