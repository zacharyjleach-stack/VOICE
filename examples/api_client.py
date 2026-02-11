"""VoiceKit API Client Example.

Shows how to call the VoiceKit REST API from Python.
Start the server first with: voicekit serve --port 8000
"""

import json
import sys

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

BASE_URL = "http://localhost:8000"


def main() -> None:
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # Health check
    resp = client.get("/health")
    print(f"Server status: {resp.json()}")

    # Full analysis
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_path:
        print("\nUsage: python api_client.py <audio_file.wav>")
        print("No audio file provided, showing available endpoints:\n")
        print("  POST /v1/analyze              - Full analysis")
        print("  POST /v1/vad                  - Voice activity detection")
        print("  POST /v1/speaker/identify     - Speaker identification")
        print("  POST /v1/speaker/enroll       - Enroll a speaker")
        print("  POST /v1/speaker/verify       - Verify speaker identity")
        print("  POST /v1/characteristics      - Voice characteristics")
        print("  POST /v1/quality              - Audio quality assessment")
        print("  GET  /docs                    - Interactive API docs")
        return

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/wav")}
        resp = client.post("/v1/analyze", files=files)

    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    main()
