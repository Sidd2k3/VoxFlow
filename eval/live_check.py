"""Compare speech recognisers on your own live recordings (saved by the app with SAVE_RECORDINGS=1).

    SAVE_RECORDINGS=1 python -m app        # then record sentences in the app as usual
    python -m eval.live_check               # every saved recording through each engine, side by side

Indian languages: IndicConformer CTC and RNNT, Indic-Transcribe-core and Google. English: Whisper turbo (fast) and Google.
The recordings stay on your computer (eval/live/ is in .gitignore).
"""

import json
import time

from app import asr, config
from app.audio import prepare

ENGINES = {"en-US": ["mlx:large-v3-turbo:fast", "google:en-IN"],
           "indian": ["indicconformer:ctc", "indicconformer:rnnt", "indictranscribe", "google"]}


def main():
    files = sorted(config.LIVE_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No recordings yet. Start the app with: SAVE_RECORDINGS=1 python -m app")
    engines: dict[str, object] = {}
    for meta_file in files:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        code = meta["language"]
        samples, seconds = prepare(meta_file.with_suffix(".wav").read_bytes())
        print(f"\n{meta_file.stem}  ({seconds:.1f} s)   the app heard: {meta['heard']}")
        for spec in ENGINES["en-US" if code == "en-US" else "indian"]:
            if spec not in engines:
                engines[spec] = asr.make_engine(spec)
                if hasattr(engines[spec], "preload"):
                    try:
                        engines[spec].preload()      # loading is not counted in the time
                    except asr.ASRServiceError:
                        pass
            t0 = time.perf_counter()
            try:
                text = asr.transcribe(samples, code, engines[spec]).text
            except asr.ASRServiceError as exc:
                text = f"(error: {exc})"
            print(f"  {spec:24s} {round((time.perf_counter() - t0) * 1000):5d} ms   {text}")


if __name__ == "__main__":
    main()
