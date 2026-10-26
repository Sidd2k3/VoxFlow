"""Compare speech recognisers on the FLEURS clips: raw ASR word/character error rate per language.

    python -m eval.compare_asr --engines google indicconformer --langs hi-IN mr-IN bn-IN pa-IN
    python -m eval.compare_asr --engines google mlx:large-v3 --langs en-US

No LLM is used. Whisper runs locally; Google results come from the main evaluation's cache.
Writes eval/results/asr_comparison.md and .json.
"""

import argparse
import json
import time
from pathlib import Path

from app import asr
from app.audio import prepare
from app.languages import LANGUAGES

from .metrics import corpus_cer, corpus_wer
from .run_eval import Cache

HERE = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR, CACHE_DIR = HERE / "data", HERE / "results", HERE / "cache"


def cache_key(engine: str, code: str, file: str) -> str:
    return f"v2|{code}|{file}" if engine == "google" else f"{engine}|{code}|{file}"


def recognise(engine: str, samples, code: str) -> dict:
    t0 = time.perf_counter()
    recognizer = asr.make_engine(engine)
    result = asr.transcribe(samples, code, recognizer)
    ms = round((time.perf_counter() - t0) * 1000)
    if engine == "google":
        time.sleep(0.5)       # be gentle with the free endpoint (not counted in the time)
    return {"text": result.text, "confidence": result.confidence, "preprocessed": result.preprocessed,
            "alternatives": result.alternatives, "ms": ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["google", "indicconformer"])
    parser.add_argument("--langs", nargs="+", default=["hi-IN", "mr-IN", "bn-IN", "pa-IN"], choices=list(LANGUAGES))
    args = parser.parse_args()
    langs = args.langs

    cache = Cache(CACHE_DIR / "asr.jsonl")
    results = {}
    for engine in args.engines:
        results[engine] = {}
        for code in langs:
            clips = [json.loads(l) for l in (DATA_DIR / code / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            hyps, refs, times = [], [], []
            for i, clip in enumerate(clips, 1):
                def run():
                    samples, _ = prepare((DATA_DIR / code / clip["file"]).read_bytes())
                    return recognise(engine, samples, code)
                heard = cache.get_or_compute(cache_key(engine, code, clip["file"]), run)
                hyps.append(heard["text"]); refs.append(clip["reference"]); times.append(heard["ms"])
                print(f"  {engine} {code} {i}/{len(clips)}", end="\r", flush=True)
            times.sort()
            results[engine][code] = {"wer": round(corpus_wer(refs, hyps), 4), "cer": round(corpus_cer(refs, hyps), 4),
                                     "median_ms": times[len(times) // 2]}
            print(f"{engine:28s} {code}: WER {results[engine][code]['wer']:.1%}  CER {results[engine][code]['cer']:.1%}"
                  f"  median {results[engine][code]['median_ms']} ms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(langs) if len(langs) < len(LANGUAGES) else "all"
    tag = "indian" if set(langs) == {"hi-IN", "mr-IN", "bn-IN", "pa-IN"} else tag
    (OUT_DIR / f"asr_comparison_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    names = [LANGUAGES[c].name for c in langs]
    md = ["# Speech recogniser comparison (FLEURS test clips, 30 per language)", "",
          "Word error rate (lower is better); character error rate in brackets.", "",
          "| Engine | " + " | ".join(names) + " | Median time per clip |", "|---|" + "---|" * (len(names) + 1)]
    for engine, per in results.items():
        cells = [f"{per[c]['wer']:.1%} ({per[c]['cer']:.1%})" for c in langs]
        median = sorted(per[c]["median_ms"] for c in langs)[len(langs) // 2]
        md.append(f"| {engine} | " + " | ".join(cells) + f" | {median} ms |")
    (OUT_DIR / f"asr_comparison_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / f'asr_comparison_{tag}.md'}")


if __name__ == "__main__":
    main()
