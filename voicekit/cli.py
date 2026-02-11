"""Command-line interface for VoiceKit.

Usage:
    voicekit analyze audio.wav
    voicekit vad audio.wav
    voicekit quality audio.wav
    voicekit serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys

from voicekit.api.models import analysis_to_dict
from voicekit.core.types import AudioConfig


def main() -> None:
    """VoiceKit CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="voicekit",
        description="VoiceKit — Voice detection and analysis SDK",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run full voice analysis on an audio file"
    )
    analyze_parser.add_argument("file", help="Path to audio file")
    analyze_parser.add_argument(
        "--no-vad", action="store_true", help="Skip voice activity detection"
    )
    analyze_parser.add_argument(
        "--no-speaker", action="store_true", help="Skip speaker identification"
    )
    analyze_parser.add_argument(
        "--no-characteristics",
        action="store_true",
        help="Skip voice characteristics analysis",
    )
    analyze_parser.add_argument(
        "--no-quality", action="store_true", help="Skip quality assessment"
    )
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # vad command
    vad_parser = subparsers.add_parser(
        "vad", help="Run voice activity detection"
    )
    vad_parser.add_argument("file", help="Path to audio file")

    # quality command
    quality_parser = subparsers.add_parser(
        "quality", help="Assess audio quality"
    )
    quality_parser.add_argument("file", help="Path to audio file")

    # serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the REST API server"
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "version":
        from voicekit import __version__

        print(f"VoiceKit v{__version__}")
        return

    if args.command == "serve":
        _serve(args)
        return

    if args.command == "analyze":
        _analyze(args)
        return

    if args.command == "vad":
        _vad(args)
        return

    if args.command == "quality":
        _quality(args)
        return


def _analyze(args: argparse.Namespace) -> None:
    """Run full analysis."""
    from voicekit.sdk import VoiceKit

    kit = VoiceKit()
    result = kit.analyze(
        args.file,
        include_vad=not args.no_vad,
        include_speaker=not args.no_speaker,
        include_characteristics=not args.no_characteristics,
        include_quality=not args.no_quality,
    )

    if args.json:
        print(json.dumps(analysis_to_dict(result), indent=2))
    else:
        _print_analysis(result)


def _vad(args: argparse.Namespace) -> None:
    """Run VAD only."""
    from voicekit.sdk import VoiceKit

    kit = VoiceKit()
    result = kit.detect_voice_activity(args.file)
    print(f"Speech segments: {len(result.segments)}")
    print(f"Speech ratio:    {result.speech_ratio:.1%}")
    print(f"Total speech:    {result.total_speech_seconds:.2f}s")
    print(f"Total duration:  {result.total_duration_seconds:.2f}s")
    print()
    for i, seg in enumerate(result.segments):
        print(
            f"  [{i+1}] {seg.start_seconds:.2f}s - {seg.end_seconds:.2f}s "
            f"({seg.duration_seconds:.2f}s, conf: {seg.confidence:.2f})"
        )


def _quality(args: argparse.Namespace) -> None:
    """Run quality assessment only."""
    from voicekit.sdk import VoiceKit

    kit = VoiceKit()
    result = kit.assess_quality(args.file)
    print(f"Overall score:   {result.overall_score:.3f}")
    print(f"SNR:             {result.snr_db:.1f} dB")
    print(f"Clipping ratio:  {result.clipping_ratio:.4f}")
    print(f"Silence ratio:   {result.silence_ratio:.1%}")
    print(f"Usable:          {'Yes' if result.is_usable else 'No'}")
    if result.issues:
        print(f"Issues:")
        for issue in result.issues:
            print(f"  - {issue}")


def _print_analysis(result: object) -> None:
    """Pretty-print analysis results."""
    r = result  # type: ignore
    print(f"=== VoiceKit Analysis ===")
    print(f"Audio: {r.audio.duration_seconds:.2f}s @ {r.audio.sample_rate}Hz")
    print(f"Processing time: {r.processing_time_seconds:.3f}s")
    print()

    if r.vad.segments:
        print(f"--- Voice Activity ---")
        print(f"Speech segments: {len(r.vad.segments)}")
        print(f"Speech ratio:    {r.vad.speech_ratio:.1%}")
        for i, seg in enumerate(r.vad.segments):
            print(
                f"  [{i+1}] {seg.start_seconds:.2f}s - {seg.end_seconds:.2f}s "
                f"(conf: {seg.confidence:.2f})"
            )
        print()

    print(f"--- Speaker ---")
    print(f"Embedding dim:   {len(r.speaker.embedding)}")
    if r.speaker.speaker_id:
        print(f"Speaker ID:      {r.speaker.speaker_id}")
    print()

    print(f"--- Characteristics ---")
    print(f"Emotion:         {r.characteristics.emotion.value}")
    print(f"Gender:          {r.characteristics.gender.value} ({r.characteristics.gender_confidence:.0%})")
    print(f"Age group:       {r.characteristics.age_group.value} ({r.characteristics.age_group_confidence:.0%})")
    print(f"Pitch:           {r.characteristics.pitch_mean_hz:.1f} Hz (std: {r.characteristics.pitch_std_hz:.1f})")
    print(f"Speech rate:     {r.characteristics.speech_rate_sps:.1f} syl/s")
    print()

    print(f"--- Quality ---")
    print(f"Overall score:   {r.quality.overall_score:.3f}")
    print(f"SNR:             {r.quality.snr_db:.1f} dB")
    print(f"Usable:          {'Yes' if r.quality.is_usable else 'No'}")
    if r.quality.issues:
        for issue in r.quality.issues:
            print(f"  ! {issue}")


def _serve(args: argparse.Namespace) -> None:
    """Start the API server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install voicekit[api]")
        sys.exit(1)

    print(f"Starting VoiceKit API server on {args.host}:{args.port}")
    uvicorn.run(
        "voicekit.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
