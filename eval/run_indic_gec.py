"""Measure grammar correction in Hindi and Bengali: the IndicGEC 2025 shared task test sets
(BHASHA workshop, IJCNLP-AACL 2025): real sentences with grammar/spelling mistakes, each with a
human correction.

    caffeinate -i python -m eval.run_indic_gec            # Hindi (236) + Bengali (330 sentences)
    python -m eval.run_indic_gec --langs hi-IN --n 50     # quick check

Data: https://github.com/BHASHA-Workshop/IndicGEC2025 (downloaded at run time into eval/data_indicgec/,
not stored in this repository). Marathi and Punjabi have no such public test set.

Score: GLEU as defined by the shared task's own scorer (n-gram 1-4, min(precision, recall) against
the correction, averaged), re-implemented here from its description, so results are comparable with
the shared task (best Hindi system: 85.69, GPT-4.1 mini). Higher is better. One difference: both
texts are Unicode-normalised (NFC) first, so invisible encoding differences (precomposed "য়" vs
"য" + nukta, which the app's normaliser produces) are not counted as errors.
Systems: source (no correction), stage1 (rules), stage2 (LLM + edit filter), full (the app:
rules -> LLM + filter -> rules), llm_raw (rules + unfiltered LLM).
Writes eval/results/indic_gec.md and indic_gec.json. LLM answers are cached in eval/cache/llm.jsonl.
"""

import argparse
import csv
import io
import json
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from app import stage1, stage2
from app.config import STAGE2_MODEL, STAGE2_SKIP_LANGUAGES

from .run_eval import Cache

HERE = Path(__file__).resolve().parent
DATA, OUT_DIR, CACHE_DIR = HERE / "data_indicgec", HERE / "results", HERE / "cache"
COMMIT = "b460f85689ac863adfce14f489fafd1db6ceac46"     # pinned version of the shared task repository
URL = "https://raw.githubusercontent.com/BHASHA-Workshop/IndicGEC2025/{commit}/{folder}/{split}.csv"
FOLDERS = {"hi-IN": "Hindi", "bn-IN": "Bangla"}
NAMES = {"source": "No correction (source)", "stage1": "Stage 1 rules only", "stage2": "Stage 2 LLM + filter only",
         "full": "Full pipeline (as in the app)", "llm_raw": "Rules + unfiltered LLM"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", text), flags=re.UNICODE)


def shared_task_gleu(refs: list[str], hyps: list[str]) -> float:
    """Corpus GLEU used by IndicGEC 2025: for n = 1..4, min(precision, recall) of n-gram matches
    against the reference correction, averaged, x100."""
    scores = []
    for n in range(1, 5):
        ref_total = hyp_total = matches = 0
        for ref, hyp in zip(refs, hyps):
            r_t, h_t = tokenize(ref.strip()), tokenize(hyp.strip())
            r = Counter(tuple(r_t[i:i + n]) for i in range(max(0, len(r_t) - n + 1)))
            h = Counter(tuple(h_t[i:i + n]) for i in range(max(0, len(h_t) - n + 1)))
            ref_total += sum(r.values())
            hyp_total += sum(h.values())
            matches += sum((r & h).values())
        precision = matches / hyp_total if hyp_total else 0.0
        recall = matches / ref_total if ref_total else 0.0
        scores.append(min(precision, recall))
    return sum(scores) / len(scores) * 100


def load(code: str, split: str = "test") -> list[dict]:
    path = DATA / FOLDERS[code] / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.format(commit=COMMIT, folder=FOLDERS[code], split=split)
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    return [{"source": r["Input sentence"].strip(), "reference": r["Output sentence"].strip()}
            for r in rows if (r.get("Input sentence") or "").strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="+", default=list(FOLDERS), choices=list(FOLDERS))
    parser.add_argument("--n", type=int, default=None, help="only the first N sentences per language")
    parser.add_argument("--model", default=STAGE2_MODEL)
    parser.add_argument("--intensity", type=int, default=50)
    args = parser.parse_args()

    client = stage2.OllamaClient(model=args.model)
    st = client.status()
    if not st["online"] or not st["model_available"]:
        raise SystemExit(f"Ollama is not ready ({st}). Start Ollama and run: ollama pull {args.model}")
    cache = Cache(CACHE_DIR / "llm.jsonl")

    def ask(text: str, code: str) -> str | None:
        key = f"igec|p{stage2.PROMPT_VERSION}|{client.model}|{code}|{text}"

        def run():
            try:
                return {"raw_output": client.chat(stage2.build_messages(text, code))}
            except Exception as exc:
                return {"raw_output": None, "error": str(exc)}
        return cache.get_or_compute(key, run).get("raw_output")

    results, md = {}, []
    t0 = time.perf_counter()
    for code in args.langs:
        rows = load(code)[:args.n]
        out = {k: [] for k in NAMES}
        details = []
        for i, r in enumerate(rows, 1):
            src = r["source"]
            s1 = stage1.correct(src, code)
            src_n = stage1.normalize(src, code)
            raw_src = ask(src_n, code)
            raw_s1 = raw_src if s1 == src_n else ask(s1, code)
            only2 = stage2.finalize(src_n, raw_src, code, args.intensity, client.model).text if raw_src else src
            r2 = stage2.finalize(s1, raw_s1, code, args.intensity, client.model) if raw_s1 else None
            full = (stage1.correct(r2.text, code) if r2.status == "applied" else r2.text) if r2 else s1
            if code in STAGE2_SKIP_LANGUAGES:     # the app runs rules only for this language
                full = s1
            for k, v in (("source", src), ("stage1", s1), ("stage2", only2), ("full", full), ("llm_raw", raw_s1 or s1)):
                out[k].append(v)
            details.append({"source": src, "reference": r["reference"], "stage1": s1, "full": full,
                            "llm_raw": raw_s1})
            print(f"  {code} {i}/{len(rows)}", end="\r", flush=True)
        refs = [r["reference"] for r in rows]
        results[code] = {k: round(shared_task_gleu(refs, v), 2) for k, v in out.items()}
        results[code]["sentences"] = len(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / f"indic_gec_{code}_details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n{code}: " + "  ".join(f"{k} {results[code][k]:.2f}" for k in NAMES))

    langs = list(results)
    titles = {"hi-IN": "Hindi", "bn-IN": "Bengali"}
    md = ["# Grammar correction in Indian languages (IndicGEC 2025 test sets)", "",
          f"Shared-task GLEU, higher is better. Stage 2 model `{client.model}`, intensity {args.intensity}.", "",
          "| System | " + " | ".join(f"{titles[c]} ({results[c]['sentences']})" for c in langs) + " |",
          "|---" * (len(langs) + 1) + "|"]
    md += [f"| {NAMES[k]} | " + " | ".join(f"{results[c][k]:.2f}" for c in langs) + " |" for k in NAMES]
    (OUT_DIR / "indic_gec.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT_DIR / "indic_gec.json").write_text(json.dumps(
        {"dataset": "IndicGEC 2025 test", "commit": COMMIT, "model": client.model, "intensity": args.intensity,
         "metric": "shared-task GLEU", "results": results,
         "minutes": round((time.perf_counter() - t0) / 60, 1)}, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / 'indic_gec.md'}")


if __name__ == "__main__":
    main()
