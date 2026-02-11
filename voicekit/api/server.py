"""FastAPI REST API server for VoiceKit.

Provides HTTP endpoints for voice analysis that can be deployed
as a microservice or cloud function.

Usage:
    uvicorn voicekit.api.server:app --host 0.0.0.0 --port 8000

Or via CLI:
    voicekit serve --port 8000
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from voicekit.core.types import AudioConfig

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install voicekit[api]"
    )


# --- Response Models ---


class VoiceSegmentResponse(BaseModel):
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    confidence: float


class VADResponse(BaseModel):
    segments: list[VoiceSegmentResponse]
    speech_ratio: float
    total_speech_seconds: float
    total_duration_seconds: float


class SpeakerResponse(BaseModel):
    embedding: list[float]
    speaker_count: int
    speaker_id: Optional[str]
    confidence: float


class EmotionScores(BaseModel):
    neutral: float = 0
    happy: float = 0
    sad: float = 0
    angry: float = 0
    fearful: float = 0
    surprised: float = 0
    disgusted: float = 0


class CharacteristicsResponse(BaseModel):
    emotion: str
    emotion_scores: dict[str, float]
    gender: str
    gender_confidence: float
    age_group: str
    age_group_confidence: float
    pitch_mean_hz: float
    pitch_std_hz: float
    energy_mean_db: float
    speech_rate_sps: float


class QualityResponse(BaseModel):
    overall_score: float
    snr_db: float
    clipping_ratio: float
    silence_ratio: float
    is_usable: bool
    issues: list[str]


class AnalysisResponse(BaseModel):
    vad: Optional[VADResponse] = None
    speaker: Optional[SpeakerResponse] = None
    characteristics: Optional[CharacteristicsResponse] = None
    quality: Optional[QualityResponse] = None
    processing_time_seconds: float
    audio_duration_seconds: float


class VerifyResponse(BaseModel):
    is_match: bool
    similarity: float


class CompareResponse(BaseModel):
    similarity: float


class HealthResponse(BaseModel):
    status: str
    version: str


# --- App Setup ---


def create_app(config: AudioConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional audio processing configuration.

    Returns:
        Configured FastAPI application.
    """
    from voicekit import __version__
    from voicekit.sdk import VoiceKit

    app = FastAPI(
        title="VoiceKit API",
        description="Production-grade voice detection and analysis API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    kit = VoiceKit(config)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version=__version__)

    @app.post("/v1/analyze", response_model=AnalysisResponse)
    async def analyze(
        file: UploadFile = File(...),
        include_vad: bool = Query(True),
        include_speaker: bool = Query(True),
        include_characteristics: bool = Query(True),
        include_quality: bool = Query(True),
    ) -> AnalysisResponse:
        """Run full voice analysis on an uploaded audio file.

        Accepts WAV, FLAC, OGG, and MP3 files. Returns comprehensive
        analysis including VAD, speaker identification, voice characteristics,
        and audio quality assessment.
        """
        try:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file")

            result = kit.analyze(
                audio_bytes,
                include_vad=include_vad,
                include_speaker=include_speaker,
                include_characteristics=include_characteristics,
                include_quality=include_quality,
            )

            return _build_analysis_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    @app.post("/v1/vad", response_model=VADResponse)
    async def vad(file: UploadFile = File(...)) -> VADResponse:
        """Run voice activity detection only."""
        try:
            audio_bytes = await file.read()
            result = kit.detect_voice_activity(audio_bytes)
            return _vad_to_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/speaker/identify", response_model=SpeakerResponse)
    async def identify_speaker(file: UploadFile = File(...)) -> SpeakerResponse:
        """Extract speaker embedding and identify speaker."""
        try:
            audio_bytes = await file.read()
            result = kit.identify_speaker(audio_bytes)
            return _speaker_to_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/speaker/enroll")
    async def enroll_speaker(
        speaker_id: str = Query(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Enroll a speaker for identification."""
        try:
            audio_bytes = await file.read()
            embedding = kit.enroll_speaker(speaker_id, audio_bytes)
            return {
                "speaker_id": speaker_id,
                "embedding_dim": len(embedding),
                "enrolled": True,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/speaker/verify", response_model=VerifyResponse)
    async def verify_speaker(
        claimed_id: str = Query(...),
        file: UploadFile = File(...),
    ) -> VerifyResponse:
        """Verify a speaker's claimed identity."""
        try:
            audio_bytes = await file.read()
            is_match, similarity = kit.verify_speaker(audio_bytes, claimed_id)
            return VerifyResponse(is_match=is_match, similarity=similarity)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/characteristics", response_model=CharacteristicsResponse)
    async def characteristics(
        file: UploadFile = File(...),
    ) -> CharacteristicsResponse:
        """Analyze voice characteristics (emotion, gender, age)."""
        try:
            audio_bytes = await file.read()
            result = kit.analyze_characteristics(audio_bytes)
            return _characteristics_to_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/quality", response_model=QualityResponse)
    async def quality(file: UploadFile = File(...)) -> QualityResponse:
        """Assess audio quality."""
        try:
            audio_bytes = await file.read()
            result = kit.assess_quality(audio_bytes)
            return _quality_to_response(result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def _build_analysis_response(result: Any) -> AnalysisResponse:
    """Convert SDK AnalysisResult to API response."""
    return AnalysisResponse(
        vad=_vad_to_response(result.vad) if result.vad.segments else None,
        speaker=_speaker_to_response(result.speaker),
        characteristics=_characteristics_to_response(result.characteristics),
        quality=_quality_to_response(result.quality),
        processing_time_seconds=result.processing_time_seconds,
        audio_duration_seconds=result.audio.duration_seconds,
    )


def _vad_to_response(result: Any) -> VADResponse:
    return VADResponse(
        segments=[
            VoiceSegmentResponse(
                start_seconds=s.start_seconds,
                end_seconds=s.end_seconds,
                duration_seconds=s.duration_seconds,
                confidence=s.confidence,
            )
            for s in result.segments
        ],
        speech_ratio=result.speech_ratio,
        total_speech_seconds=result.total_speech_seconds,
        total_duration_seconds=result.total_duration_seconds,
    )


def _speaker_to_response(result: Any) -> SpeakerResponse:
    return SpeakerResponse(
        embedding=result.embedding,
        speaker_count=result.speaker_count,
        speaker_id=result.speaker_id,
        confidence=result.confidence,
    )


def _characteristics_to_response(result: Any) -> CharacteristicsResponse:
    return CharacteristicsResponse(
        emotion=result.emotion.value,
        emotion_scores=result.emotion_scores,
        gender=result.gender.value,
        gender_confidence=result.gender_confidence,
        age_group=result.age_group.value,
        age_group_confidence=result.age_group_confidence,
        pitch_mean_hz=result.pitch_mean_hz,
        pitch_std_hz=result.pitch_std_hz,
        energy_mean_db=result.energy_mean_db,
        speech_rate_sps=result.speech_rate_sps,
    )


def _quality_to_response(result: Any) -> QualityResponse:
    return QualityResponse(
        overall_score=result.overall_score,
        snr_db=result.snr_db,
        clipping_ratio=result.clipping_ratio,
        silence_ratio=result.silence_ratio,
        is_usable=result.is_usable,
        issues=result.issues,
    )


# Default app instance for uvicorn
app = create_app()
