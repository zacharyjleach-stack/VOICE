"""Serialization helpers for converting SDK types to API-friendly dicts.

Used for webhooks, message queues, and other non-HTTP integrations.
"""

from __future__ import annotations

from typing import Any

from voicekit.core.types import AnalysisResult


def analysis_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Convert a full AnalysisResult to a JSON-serializable dict.

    Useful for sending results to webhooks, message queues,
    or storing in databases.

    Args:
        result: Complete analysis result.

    Returns:
        Nested dictionary representation.
    """
    return {
        "audio": {
            "sample_rate": result.audio.sample_rate,
            "channels": result.audio.channels,
            "duration_seconds": result.audio.duration_seconds,
        },
        "vad": {
            "segments": [
                {
                    "start_seconds": s.start_seconds,
                    "end_seconds": s.end_seconds,
                    "duration_seconds": s.duration_seconds,
                    "confidence": s.confidence,
                }
                for s in result.vad.segments
            ],
            "speech_ratio": result.vad.speech_ratio,
            "total_speech_seconds": result.vad.total_speech_seconds,
            "total_duration_seconds": result.vad.total_duration_seconds,
        },
        "speaker": {
            "embedding": result.speaker.embedding,
            "speaker_count": result.speaker.speaker_count,
            "speaker_id": result.speaker.speaker_id,
            "confidence": result.speaker.confidence,
        },
        "characteristics": {
            "emotion": result.characteristics.emotion.value,
            "emotion_scores": result.characteristics.emotion_scores,
            "gender": result.characteristics.gender.value,
            "gender_confidence": result.characteristics.gender_confidence,
            "age_group": result.characteristics.age_group.value,
            "age_group_confidence": result.characteristics.age_group_confidence,
            "pitch_mean_hz": result.characteristics.pitch_mean_hz,
            "pitch_std_hz": result.characteristics.pitch_std_hz,
            "energy_mean_db": result.characteristics.energy_mean_db,
            "speech_rate_sps": result.characteristics.speech_rate_sps,
        },
        "quality": {
            "overall_score": result.quality.overall_score,
            "snr_db": result.quality.snr_db,
            "clipping_ratio": result.quality.clipping_ratio,
            "silence_ratio": result.quality.silence_ratio,
            "is_usable": result.quality.is_usable,
            "issues": result.quality.issues,
        },
        "processing_time_seconds": result.processing_time_seconds,
    }
