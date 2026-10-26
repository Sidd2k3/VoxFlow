"""Head-to-head on the same recordings: Google's speech service (the engine behind Chrome's mic) vs
VoxFlow (IndicConformer on this computer), for Hindi, Marathi, Bengali and Punjabi.

    python -m eval.speed_test              # 10 FLEURS clips per language (needs eval.fetch_fleurs first)
    python -m eval.speed_test --n 30

Every clip is sent to both, live (nothing cached), taking turns which goes first.
Time = from "the recording is finished" to "the text is back", i.e. what you wait for after you stop
speaking. Google is called the way the app always called it: the whole clip in one request.
Also reported: word error rate on the same clips, and how long VoxFlow's grammar steps take
(rules, then the LLM), so the full wait is visible too.
Writes eval/results/speed_test.md and .json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app import asr, stage1, stage2
from app.audio import prepare
from app.config import ASR_ROUTES, STAGE2_SKIP_LANGUAGES
from app.languages import LANGUAGES

from .metrics import corpus_wer

HERE = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR = HERE / "data", HERE / "results"
INDIAN = ["hi-IN", "mr-IN", "bn-IN", "pa-IN"]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - t0) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="clips per language")
    parser.add_argument("--langs", nargs="+", choices=INDIAN,
                        default=[c for c in INDIAN if not ASR_ROUTES[c].startswith("google")])   # Google vs itself is moot
    parser.add_argument("--no-llm", action="store_true", help="skip timing the LLM step")
    args = parser.parse_args()

    google = asr.GoogleRecognizer()
    # the app's own engine per language (no fallback: a measurement must not silently use Google)
    engines = {code: asr.make_engine(ASR_ROUTES[code].split("|")[0]) for code in args.langs}
    print("Loading IndicConformer (first time downloads ~2.5 GB)...")
    next(iter(engines.values())).preload()   # loading the model is not counted, as in the running app
    llm = stage2.OllamaClient()
    use_llm = not args.no_llm and llm.status().get("model_available")

    results, rows = {}, []
    for code in args.langs:
        clips = [json.loads(l) for l in (DATA_DIR / code / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        clips = clips[:args.n]
        g_ms, o_ms, rules_ms, llm_ms, refs, g_txt, o_txt = [], [], [], [], [], [], []
        for i, clip in enumerate(clips):
            samples, _ = prepare((DATA_DIR / code / clip["file"]).read_bytes())
            ours = engines[code]
            order = [("g", google), ("o", ours)] if i % 2 == 0 else [("o", ours), ("g", google)]
            got = {}
            for name, engine in order:
                try:
                    got[name] = timed(lambda: engine.recognize(samples, code)[0])
                except asr.ASRServiceError as exc:
                    raise SystemExit(f"Stopped: {exc}")
            refs.append(clip["reference"])
            g_txt.append(got["g"][0]); g_ms.append(got["g"][1])
            o_txt.append(got["o"][0]); o_ms.append(got["o"][1])
            s1, ms = timed(lambda: stage1.correct(got["o"][0], code))
            rules_ms.append(ms)
            if use_llm and code not in STAGE2_SKIP_LANGUAGES and s1:
                llm_ms.append(timed(lambda: stage2.correct(s1, code, 50, llm))[1])
            print(f"  {code} {i + 1}/{len(clips)}  Google {got['g'][1]} ms  VoxFlow {got['o'][1]} ms", flush=True)
        results[code] = {
            "clips": len(clips),
            "google_median_ms": round(statistics.median(g_ms)), "ours_median_ms": round(statistics.median(o_ms)),
            "ours_faster_clips": sum(o < g for o, g in zip(o_ms, g_ms)),
            "google_wer": round(corpus_wer(refs, g_txt), 4), "ours_wer": round(corpus_wer(refs, o_txt), 4),
            "rules_median_ms": round(statistics.median(rules_ms)),
            "llm_median_ms": round(statistics.median(llm_ms)) if llm_ms else None,
        }
        r = results[code]
        rows.append(f"| {LANGUAGES[code].name} | {r['clips']} | {r['google_median_ms']} ms | {r['ours_median_ms']} ms | "
                    f"{r['ours_faster_clips']}/{r['clips']} | {r['google_wer']:.1%} | {r['ours_wer']:.1%} | "
                    f"{r['rules_median_ms']} ms | {'–' if r['llm_median_ms'] is None else str(r['llm_median_ms']) + ' ms'} |")

    md = ["# Google (Chrome's speech engine) vs VoxFlow, same recordings", "",
          "Median time from the end of the recording to the text; word error rate (lower is better). FLEURS test clips. "
          "VoxFlow = IndicConformer as the app runs it: " + ", ".join(f"{LANGUAGES[c].name} {ASR_ROUTES[c].split('|')[0]}"
                                                                          for c in args.langs) + ".", "",
          "| Language | Clips | Google time | VoxFlow time | VoxFlow faster on | Google WER | VoxFlow WER "
          "| + grammar rules | + LLM |", "|---|---|---|---|---|---|---|---|---|", *rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "speed_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT_DIR / "speed_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / 'speed_test.md'}")


if __name__ == "__main__":
    main()
