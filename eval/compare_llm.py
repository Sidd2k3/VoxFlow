"""Which local LLM should do Stage 2? Compares Ollama models on grammar correction, on DEVELOPMENT
data only (the test sets stay untouched for the final numbers):

    ollama pull gemma3:4b          # once per model you want to try
    caffeinate -i python -m eval.compare_llm --models llama3.2 gemma3:4b gemma3:12b

Per model: the full pipeline (rules -> LLM + edit filter -> rules, exactly as the app) on
  - English: JFLEG validation split, first --n-en sentences (source-aware GLEU, 4 references)
  - Hindi and Bengali: IndicGEC 2025 dev split (shared-task GLEU)
and the median LLM time per sentence. A model that isn't pulled is skipped.
Writes eval/results/llm_comparison.md and .json. Answers are cached in eval/cache/llm.jsonl.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app import stage1, stage2

from . import run_gec, run_indic_gec
from .run_eval import Cache

HERE = Path(__file__).resolve().parent
OUT_DIR, CACHE_DIR = HERE / "results", HERE / "cache"


def pipeline(text: str, code: str, ask) -> tuple[str, str | None]:
    """The app's correction on typed text: rules -> LLM + filter -> rules. Returns (output, raw answer)."""
    s1 = stage1.correct(text, code)
    raw = ask(s1, code)
    if not raw:
        return s1, None
    r = stage2.finalize(s1, raw, code, 50, "")
    return (stage1.correct(r.text, code) if r.status == "applied" else r.text), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["llama3.2", "gemma3:4b", "gemma3:12b"])
    parser.add_argument("--n-en", type=int, default=150)
    args = parser.parse_args()

    cache = Cache(CACHE_DIR / "llm.jsonl")
    english = run_gec.load_jfleg(args.n_en, split="validation")
    indic = {code: run_indic_gec.load(code, split="dev") for code in run_indic_gec.FOLDERS}
    baseline = {"en-US": run_gec.gleu_multi([r["sentence"] for r in english], [r["sentence"] for r in english],
                                            [[r["corrections"][k] for r in english] for k in range(4)])}
    for code, rows in indic.items():
        baseline[code] = run_indic_gec.shared_task_gleu([r["reference"] for r in rows], [r["source"] for r in rows])

    results = {}
    for model in args.models:
        client = stage2.OllamaClient(model=model)
        st = client.status()
        if not st["online"]:
            raise SystemExit("Ollama is not running. Open the Ollama app and try again.")
        if not st["model_available"]:
            print(f"{model}: skipped (run: ollama pull {model})")
            continue
        times = []

        def ask(text: str, code: str) -> str | None:
            key = f"cmp|p{stage2.PROMPT_VERSION}|{model}|{code}|{text}"

            def run():
                t0 = time.perf_counter()
                try:
                    return {"raw_output": client.chat(stage2.build_messages(text, code)),
                            "ms": round((time.perf_counter() - t0) * 1000)}
                except Exception as exc:
                    return {"raw_output": None, "error": str(exc), "ms": 0}
            got = cache.get_or_compute(key, run)
            if got.get("ms"):
                times.append(got["ms"])
            return got.get("raw_output")

        row = {}
        outs = []
        for i, r in enumerate(english, 1):
            outs.append(pipeline(r["sentence"], "en-US", ask)[0])
            print(f"  {model} English {i}/{len(english)}", end="\r", flush=True)
        row["en-US"] = run_gec.gleu_multi([r["sentence"] for r in english], outs,
                                          [[r["corrections"][k] for r in english] for k in range(4)])
        for code, rows in indic.items():
            outs = []
            for i, r in enumerate(rows, 1):
                outs.append(pipeline(r["source"], code, ask)[0])
                print(f"  {model} {code} {i}/{len(rows)}      ", end="\r", flush=True)
            row[code] = run_indic_gec.shared_task_gleu([r["reference"] for r in rows], outs)
        row["median_ms"] = round(statistics.median(times)) if times else None
        results[model] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}
        print(f"\n{model}: English {row['en-US']:.2f}  Hindi {row['hi-IN']:.2f}  Bengali {row['bn-IN']:.2f}  "
              f"median {row['median_ms']} ms per sentence")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Stage 2 model comparison (development data)", "",
          f"Full pipeline GLEU, higher is better. English: JFLEG validation ({len(english)} sentences); "
          f"Hindi/Bengali: IndicGEC 2025 dev ({len(indic['hi-IN'])}/{len(indic['bn-IN'])}).", "",
          "| Stage 2 model | English | Hindi | Bengali | Median time per sentence |", "|---|---|---|---|---|",
          f"| none (no correction) | {baseline['en-US']:.2f} | {baseline['hi-IN']:.2f} | {baseline['bn-IN']:.2f} | – |"]
    md += [f"| {m} | {r['en-US']:.2f} | {r['hi-IN']:.2f} | {r['bn-IN']:.2f} | {r['median_ms']} ms |"
           for m, r in results.items()]
    (OUT_DIR / "llm_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT_DIR / "llm_comparison.json").write_text(json.dumps(
        {"baseline": {k: round(v, 2) for k, v in baseline.items()}, "models": results}, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / 'llm_comparison.md'}")


if __name__ == "__main__":
    main()
