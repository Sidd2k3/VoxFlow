"""Run one of your own recordings through several recognisers and print what each one heard.

    python -m eval.try_recording ~/Downloads/my-recording.wav                  # English engines
    python -m eval.try_recording ~/Downloads/my-recording.wav --lang bn-IN     # Indian-language engines

Download the WAV from the app (Text Output -> "Download my-recording.wav").
"""

import argparse
import time
from pathlib import Path

from app import asr
from app.audio import prepare

ENGINES = ["mlx:large-v3-turbo:fast", "mlx:large-v3", "parakeet:v3", "google:en-IN"]
INDIAN_ENGINES = ["indicconformer:ctc", "indicconformer:rnnt", "indictranscribe", "google"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--lang", default="en-US", choices=["en-US", "hi-IN", "mr-IN", "bn-IN", "pa-IN"])
    parser.add_argument("--engines", nargs="+", default=None)
    args = parser.parse_args()
    engines = args.engines or (ENGINES if args.lang == "en-US" else INDIAN_ENGINES)
    samples, _ = prepare(args.wav.expanduser().read_bytes())
    print(f"{len(samples) / 16000:.1f} s of audio\n")
    for spec in engines:
        engine = asr.make_engine(spec)
        if hasattr(engine, "available") and not engine.available():
            print(f"{spec}\n  skipped: needs `pip install mlx-whisper parakeet-mlx`\n")
            continue
        t0 = time.perf_counter()
        try:
            if hasattr(engine, "preload"):
                engine.preload()           # loading the model is not counted in the time
            t0 = time.perf_counter()
            heard = asr.transcribe(samples, args.lang, engine)
        except asr.ASRServiceError as exc:
            print(f"{spec}\n  error: {exc}\n")
            continue
        unsure = f"   (unsure: {', '.join(heard.uncertain)})" if heard.uncertain else ""
        print(f"{spec}  [{round((time.perf_counter() - t0) * 1000)} ms]\n  {heard.text}{unsure}\n")


if __name__ == "__main__":
    main()
