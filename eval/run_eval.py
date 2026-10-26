"""Measure the pipeline ourselves: WER/CER of each stage on the FLEURS clips (an ablation).

For every clip:
    asr     = raw Google transcript
    stage1  = Stage 1 rules on the transcript
    stage2  = Stage 2 LLM on the transcript (no rules)
    full    = Stage 1 then Stage 2 (what the app does)
Each version is compared with the human reference transcript.

    python -m eval.fetch_fleurs --n 30      # once, downloads the clips
    python -m eval.run_eval                 # needs internet (Google ASR) and Ollama running

ASR and LLM answers are cached in eval/cache, so a re-run only calls what is new.
"""

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from app import asr, stage1, stage2
from app.audio import prepare
from app.config import ASR_ENGINE, ASR_MODEL, RESULTS_FILE, STAGE2_MODEL, STAGE2_SKIP_LANGUAGES
from app.languages import LANGUAGES

from .metrics import corpus_cer, corpus_wer

HERE = Path(__file__).resolve().parent
DATA_DIR, CACHE_DIR = HERE / "data", HERE / "cache"
CONDITIONS = ("asr", "stage1", "stage2", "full")


class Cache:
    """A JSON-lines file of key -> value, so repeated runs don't repeat slow/limited calls."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self.data[item["key"]] = item["value"]

    def get_or_compute(self, key: str, compute):
        if key not in self.data:
            self.data[key] = compute()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": self.data[key]}, ensure_ascii=False) + "\n")
        return self.data[key]


from app.config import ASR_ROUTES

ENGINE = {"google": "google", "whisper": f"whisper:{ASR_MODEL}"}.get(ASR_ENGINE, "routed")


def _full_spec(spec: str) -> str:
    return f"whisper:{ASR_MODEL}" if spec == "whisper" else spec


def engine_for(code: str) -> str:
    """The exact recogniser the app uses for this language (e.g. mlx:large-v3 for English on a Mac)."""
    if ENGINE != "routed":
        return ENGINE
    # no silent fallback in a measurement: "indicconformer|google" is measured as IndicConformer
    return _full_spec(ASR_ROUTES.get(code, "google").split("|")[0])


def asr_key(code: str, file: str) -> str:
    # same keys as eval/compare_asr.py, so recognition results are shared between the two scripts
    engine = engine_for(code)
    # Whisper results are keyed "w2|..." since they also store per-word uncertain words
    return f"v2|{code}|{file}" if engine == "google" else f"w2|{engine}|{code}|{file}"


def run_asr(path: Path, code: str) -> dict:
    samples, _ = prepare(path.read_bytes())
    t0 = time.perf_counter()
    result = asr.transcribe(samples, code, asr.make_engine(engine_for(code)))
    ms = round((time.perf_counter() - t0) * 1000)
    if engine_for(code) == "google":
        time.sleep(0.5)  # be gentle with the free Google endpoint (not counted in the time)
    return {"text": result.text, "confidence": result.confidence, "preprocessed": result.preprocessed,
            "alternatives": result.alternatives, "uncertain": result.uncertain, "ms": ms}


def ask_llm(text: str, code: str, client: stage2.OllamaClient, alternatives: list[str],
            uncertain: list[str] | None = None) -> dict:
    """The expensive part: one LLM call. Only the raw answer is cached; the filter runs afterwards."""
    t0 = time.perf_counter()
    try:
        answer = client.chat(stage2.build_messages(text, code, alternatives, uncertain))
    except Exception as exc:
        return {"raw_output": None, "error": str(exc), "ms": round((time.perf_counter() - t0) * 1000)}
    return {"raw_output": answer, "ms": round((time.perf_counter() - t0) * 1000)}


def apply_stage2(text: str, code: str, intensity: int, model: str, alternatives: list[str], cached: dict,
                 uncertain: list[str] | None = None, rules_after: bool = False) -> dict:
    """Filter a cached LLM answer exactly as the app does (including the recogniser's unsure words)."""
    if not text:
        return {"text": "", "status": "skipped", "kept": 0, "reverted": 0, "raw_output": "", "ms": 0}
    if cached.get("raw_output") is None:
        return {"text": text, "status": "unavailable", "kept": 0, "reverted": 0, "raw_output": "", "ms": cached["ms"]}
    r = stage2.finalize(text, cached["raw_output"], code, intensity, model, alternatives, uncertain)
    after = stage1.correct(r.text, code) if rules_after and r.status == "applied" else stage1.normalize(r.text, code)
    return {"text": after, "status": r.status, "kept": r.kept_edits,
            "reverted": r.reverted_edits, "raw_output": r.raw_output, "ms": cached["ms"]}


def evaluate_language(code: str, intensity: int, client: stage2.OllamaClient, asr_cache: Cache,
                      llm_cache: Cache, limit: int | None) -> tuple[dict, list[dict]]:
    manifest = DATA_DIR / code / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"No test clips for {code}. Run: python -m eval.fetch_fleurs")
    clips = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()][:limit]

    rows = []
    for i, clip in enumerate(clips, 1):
        heard = asr_cache.get_or_compute(asr_key(code, clip["file"]),
                                         lambda: run_asr(DATA_DIR / code / clip["file"], code))
        raw, alts, unc = heard["text"], heard.get("alternatives", []), heard.get("uncertain", [])
        s1 = stage1.correct(raw, code) if raw else ""
        # the cached LLM answer depends on the prompt and model, not on the intensity or the filter
        key = f"p{stage2.PROMPT_VERSION}|{client.model}|{intensity}|{code}|{json.dumps(alts, ensure_ascii=False)}|"
        if unc:  # the prompt mentions uncertain words, so the cached answer depends on them
            key += f"u{json.dumps(unc, ensure_ascii=False)}|"
        raw_norm = stage1.normalize(raw, code) if raw else ""
        only2 = apply_stage2(raw_norm, code, intensity, client.model, alts,
                             llm_cache.get_or_compute(key + "raw|" + raw, lambda: ask_llm(raw_norm, code, client, alts, unc))
                             if raw else {}, unc)
        full = apply_stage2(s1, code, intensity, client.model, alts,
                            llm_cache.get_or_compute(key + "s1|" + s1, lambda: ask_llm(s1, code, client, alts, unc))
                            if raw else {}, unc, rules_after=True)   # as in the app (pipeline.py)
        if code in STAGE2_SKIP_LANGUAGES:   # the app runs rules only for this language
            full = {**full, "text": s1, "status": "skipped", "kept": 0, "reverted": 0, "ms": 0}
        rows.append({"file": clip["file"], "reference": clip["reference"], "asr": raw, "stage1": s1,
                     "stage2": only2["text"], "full": full["text"], "full_status": full["status"],
                     "asr_confidence": heard["confidence"], "asr_alternatives": alts, "asr_uncertain": unc,
                     "full_llm_raw": full.get("raw_output", ""), "full_kept": full.get("kept", 0),
                     "full_reverted": full.get("reverted", 0),
                     "asr_ms": heard["ms"], "stage2_ms": full["ms"]})
        print(f"  {code} {i}/{len(clips)}  stage 2: {full['status']}", flush=True)

    refs = [r["reference"] for r in rows]
    statuses = [r["full_status"] for r in rows]
    summary = {
        "n": len(rows),
        "wer": {c: round(corpus_wer(refs, [r[c] for r in rows]), 4) for c in CONDITIONS},
        "cer": {c: round(corpus_cer(refs, [r[c] for r in rows]), 4) for c in CONDITIONS},
        "empty_asr": sum(1 for r in rows if not r["asr"]),
        "clips_with_alternatives": sum(1 for r in rows if r["asr_alternatives"]),
        "llm_edits": {"kept": sum(r["full_kept"] for r in rows), "reverted": sum(r["full_reverted"] for r in rows)},
        "stage2_status": {s: statuses.count(s) for s in sorted(set(statuses))},
        "latency_ms": {
            "asr_median": statistics.median(r["asr_ms"] for r in rows) if rows else 0,
            "stage2_median": statistics.median(r["stage2_ms"] for r in rows) if rows else 0,
        },
    }
    return summary, rows


def describe_asr(asr_setting) -> str:
    if isinstance(asr_setting, str):
        return f"`{asr_setting}`"
    return ", ".join(f"{LANGUAGES[c].name} `{e}`" for c, e in asr_setting.items())


def write_markdown(results: dict, path: Path) -> None:
    lines = [f"# Evaluation results ({results['date']})", "",
             f"Dataset: {results['dataset']}. Speech recognition: {describe_asr(results['asr'])}. "
             f"Stage 2 model: `{results['model']}`, intensity {results['intensity']}.",
             "", "## Word error rate (lower is better)", "",
             "| Language | Clips | Raw ASR | + Stage 1 | Stage 2 only | Full pipeline |",
             "|---|---|---|---|---|---|"]
    for code, r in results["per_language"].items():
        w = r["wer"]
        lines.append(f"| {LANGUAGES[code].name} | {r['n']} | {w['asr']:.1%} | {w['stage1']:.1%} | "
                     f"{w['stage2']:.1%} | {w['full']:.1%} |")
    lines += ["", "## Character error rate", "",
              "| Language | Raw ASR | + Stage 1 | Stage 2 only | Full pipeline |", "|---|---|---|---|---|"]
    for code, r in results["per_language"].items():
        c = r["cer"]
        lines.append(f"| {LANGUAGES[code].name} | {c['asr']:.1%} | {c['stage1']:.1%} | {c['stage2']:.1%} | {c['full']:.1%} |")
    lines += ["", "## Stage 2 outcomes and latency", "",
              "| Language | Stage 2 outcomes | LLM edits kept / reverted | Clips with ASR alternatives | "
              "Median ASR ms | Median Stage 2 ms |", "|---|---|---|---|---|---|"]
    for code, r in results["per_language"].items():
        outcomes = ", ".join(f"{k} {v}" for k, v in r["stage2_status"].items())
        lines.append(f"| {LANGUAGES[code].name} | {outcomes} | {r['llm_edits']['kept']} / {r['llm_edits']['reverted']} | "
                     f"{r['clips_with_alternatives']}/{r['n']} | {r['latency_ms']['asr_median']:.0f} | "
                     f"{r['latency_ms']['stage2_median']:.0f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--limit", type=int, default=None, help="use only the first N clips per language")
    parser.add_argument("--intensity", type=int, default=50)
    parser.add_argument("--model", default=STAGE2_MODEL)
    args = parser.parse_args()

    client = stage2.OllamaClient(model=args.model)
    llm_status = client.status()
    if not llm_status["online"] or not llm_status["model_available"]:
        raise SystemExit(f"Ollama is not ready ({llm_status}). Start Ollama and run: ollama pull {args.model}")

    asr_cache, llm_cache = Cache(CACHE_DIR / "asr.jsonl"), Cache(CACHE_DIR / "llm.jsonl")
    results = {"dataset": "FLEURS test split (Google)",
               "asr": ENGINE if ENGINE != "routed" else {c: engine_for(c) for c in LANGUAGES}, "model": args.model, "intensity": args.intensity,
               "date": date.today().isoformat(), "per_language": {}}
    out_dir = RESULTS_FILE.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in args.langs:
        try:
            summary, rows = evaluate_language(code, args.intensity, client, asr_cache, llm_cache, args.limit)
        except asr.ASRServiceError as exc:
            raise SystemExit(f"\nStopped: {exc}\nEverything finished so far is cached. "
                             f"Check your internet and run the same command again to continue.")
        results["per_language"][code] = summary
        with open(out_dir / f"{code}_details.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{code}: WER {summary['wer']}")

    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, out_dir / "results.md")
    print(f"Saved {RESULTS_FILE} and {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
