"""Which recogniser understands Indian-accented English best? (Svarah clips)

    python -m eval.fetch_svarah --n 60     # once
    python -m eval.compare_english

Compares Google en-US / en-IN, Whisper large-v3-turbo on the CPU (normal and verbatim prompt), and
Whisper large-v3 and large-v3-turbo on the Mac GPU (MLX). Reports WER, CER, missed words (spoken words that
are absent from the transcript) and speed. Writes eval/results/indian_english.md and .json.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from app import asr
from app.audio import prepare

from .metrics import corpus_cer, corpus_missed_words, corpus_wer
from .run_eval import Cache

HERE = Path(__file__).resolve().parent
DATA, OUT_DIR, CACHE_DIR = HERE / "data_svarah", HERE / "results", HERE / "cache"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["google:en-US", "google:en-IN", "whisper:large-v3-turbo", "whisper:large-v3-turbo:verbatim",
                                 "mlx:large-v3", "mlx:large-v3:verbatim", "mlx:large-v3-turbo",
                                 "mlx:large-v3-turbo:fast", "parakeet:v2", "parakeet:v3"])
    args = parser.parse_args()
    manifest = DATA / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit("No clips yet. Run: python -m eval.fetch_svarah")
    clips = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = Cache(CACHE_DIR / "asr.jsonl")
    results = {}
    for spec in args.engines:
        engine = asr.make_engine(spec)
        if hasattr(engine, "available") and not engine.available():
            print(f"{spec:30s} skipped: needs `pip install mlx-whisper parakeet-mlx` on an Apple Silicon Mac")
            continue
        # the verbatim prompt is part of the engine, so a changed prompt must not reuse old results
        tag = spec + (f"|{hashlib.sha1(asr.VERBATIM_PROMPT.encode()).hexdigest()[:8]}" if "verbatim" in spec else "")
        if spec.startswith("google"):
            tag += "|t2"     # re-measured: older results counted the pause below in the time
        refs, hyps, times = [], [], []
        for i, clip in enumerate(clips, 1):
            def run():
                samples, _ = prepare((DATA / clip["file"]).read_bytes())
                t0 = time.perf_counter()
                try:
                    heard = asr.transcribe(samples, "en-US", engine)
                except asr.ASRServiceError as exc:
                    raise SystemExit(f"Stopped: {exc}. Results so far are cached; run again to continue.")
                ms = round((time.perf_counter() - t0) * 1000)
                if spec.startswith("google"):
                    time.sleep(0.5)    # be gentle with the free endpoint (not counted in the time)
                return {"text": heard.text, "ms": ms}
            got = cache.get_or_compute(f"svarah|{tag}|{clip['file']}", run)
            refs.append(clip["reference"]); hyps.append(got["text"]); times.append(got["ms"])
            print(f"  {spec} {i}/{len(clips)}", end="\r", flush=True)
        times.sort()
        results[spec] = {"wer": round(corpus_wer(refs, hyps), 4), "cer": round(corpus_cer(refs, hyps), 4),
                         "missed": round(corpus_missed_words(refs, hyps), 4),
                         "median_ms": times[len(times) // 2], "clips": len(clips)}
        r = results[spec]
        print(f"{spec:30s} WER {r['wer']:.1%}  CER {r['cer']:.1%}  missed words {r['missed']:.1%}  "
              f"median {r['median_ms']} ms")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "indian_english.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = [f"# Indian-accented English (AI4Bharat Svarah, {len(clips)} clips)", "",
          "| Engine | WER | CER | Missed words | Median time per clip |", "|---|---|---|---|---|"]
    md += [f"| {e} | {r['wer']:.1%} | {r['cer']:.1%} | {r['missed']:.1%} | {r['median_ms']} ms |"
           for e, r in results.items()]
    (OUT_DIR / "indian_english.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {OUT_DIR / 'indian_english.md'}")


if __name__ == "__main__":
    main()
