# VoiceKit

Production-grade voice detection and analysis SDK designed for AI companies to integrate into their products.

## Features

| Module | Capability |
|--------|-----------|
| **Voice Activity Detection** | Real-time and batch speech segment detection with adaptive thresholding |
| **Speaker Identification** | Speaker embeddings, enrollment, verification, and comparison |
| **Voice Characteristics** | Emotion, gender, age group, pitch, energy, and speech rate analysis |
| **Audio Quality Assessment** | SNR estimation, clipping detection, silence analysis, usability scoring |

## Installation

```bash
# Core SDK
pip install .

# With REST API server
pip install ".[api]"

# With ML model support
pip install ".[ml]"

# Everything (including dev tools)
pip install ".[all]"
```

## Quick Start

```python
from voicekit import VoiceKit

kit = VoiceKit()
result = kit.analyze("recording.wav")

# Voice activity segments
for segment in result.vad.segments:
    print(f"Speech: {segment.start_seconds:.1f}s - {segment.end_seconds:.1f}s")

# Speaker embedding (192-dim vector)
embedding = result.speaker.embedding

# Voice characteristics
print(f"Emotion: {result.characteristics.emotion.value}")
print(f"Gender: {result.characteristics.gender.value}")
print(f"Pitch: {result.characteristics.pitch_mean_hz:.0f} Hz")

# Audio quality
print(f"Quality: {result.quality.overall_score:.2f}")
print(f"Usable: {result.quality.is_usable}")
```

## API Reference

### `VoiceKit` — Main SDK Class

```python
kit = VoiceKit(config=AudioConfig())

# Full pipeline
result = kit.analyze(source, include_vad=True, include_speaker=True, ...)

# Individual modules
vad = kit.detect_voice_activity(source)
speaker = kit.identify_speaker(source)
chars = kit.analyze_characteristics(source)
quality = kit.assess_quality(source)

# Speaker verification
kit.enroll_speaker("alice", enrollment_audio)
is_match, score = kit.verify_speaker(test_audio, "alice")
similarity = kit.compare_speakers(audio_a, audio_b)

# Streaming VAD (real-time, frame-by-frame)
is_speech, confidence, state = kit.process_stream_frame(frame, state)
```

### Input Sources

VoiceKit accepts audio from multiple sources:

```python
kit.analyze("path/to/file.wav")        # File path
kit.analyze(wav_bytes)                  # Raw WAV bytes
kit.analyze(numpy_array)               # NumPy array of samples
```

### Configuration

```python
from voicekit import AudioConfig

config = AudioConfig(
    sample_rate=16000,           # Target sample rate (Hz)
    frame_duration_ms=30,        # Analysis frame duration
    min_speech_duration_ms=250,  # Minimum speech segment to keep
    max_silence_duration_ms=300, # Max silence before splitting segments
    vad_threshold=0.5,           # VAD confidence threshold (0.0-1.0)
    embedding_dim=192,           # Speaker embedding dimensions
)
kit = VoiceKit(config)
```

## REST API

Start the API server:

```bash
voicekit serve --port 8000
# Or with Docker
docker compose up
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/v1/analyze` | Full analysis pipeline |
| `POST` | `/v1/vad` | Voice activity detection |
| `POST` | `/v1/speaker/identify` | Speaker identification |
| `POST` | `/v1/speaker/enroll` | Enroll a speaker |
| `POST` | `/v1/speaker/verify` | Verify speaker identity |
| `POST` | `/v1/characteristics` | Voice characteristics |
| `POST` | `/v1/quality` | Audio quality assessment |

Interactive docs available at `http://localhost:8000/docs`.

### Example API Call

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -F "file=@recording.wav" \
  -F "include_vad=true" \
  -F "include_speaker=true"
```

## CLI

```bash
voicekit analyze recording.wav          # Full analysis
voicekit analyze recording.wav --json   # JSON output
voicekit vad recording.wav              # VAD only
voicekit quality recording.wav          # Quality check
voicekit serve --port 8000              # Start API server
voicekit version                        # Show version
```

## Architecture

```
voicekit/
  core/
    types.py          # Data types (AudioSegment, VADResult, etc.)
    audio.py          # Audio engine (loading, resampling, features)
  detectors/
    vad.py            # Voice Activity Detection
    speaker.py        # Speaker embedding & identification
  analyzers/
    characteristics.py # Emotion, gender, age, pitch analysis
    quality.py        # Audio quality assessment
  api/
    server.py         # FastAPI REST server
    models.py         # Serialization helpers
  sdk.py              # Main VoiceKit class
  cli.py              # Command-line interface
```

## Development

```bash
# Install with dev dependencies
pip install ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=voicekit

# Type checking
mypy voicekit

# Linting
ruff check voicekit
```

## License

MIT
