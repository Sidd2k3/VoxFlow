"""Measure grammar correction itself: JFLEG test set (sentences with real grammar errors, 4 human corrections each).

    python -m eval.run_gec            # the full JFLEG test set (747 sentences)

The edit filter was tuned by looking at sentences 1-300 ("dev"). Sentences 301-747 ("held-out") were
not looked at, so their score is the honest one.

Scores (GLEU, the standard JFLEG metric; higher is better, 0-100):
    source   - the uncorrected sentence (the baseline every system must beat)
    stage1   - rules only
    stage2   - LLM + edit filter only
    full     - rules then LLM + edit filter (what the app does)
    llm_raw  - the LLM's answer with no filter (shows what the filter costs or saves)

GLEU is implemented from scratch (Napoles et al., 2015/2016): n-gram precision against the human
corrections, minus credit for n-grams that were left uncorrected from the source. It is computed
against each of the 4 reference sets and averaged.
Writes eval/results/gec.md and gec.json. LLM answers are cached in eval/cache/llm.jsonl.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from app import stage1, stage2
from app.config import STAGE2_MODEL
from app.text_utils import tokenize

from .run_eval import Cache

HERE = Path(__file__).resolve().parent
OUT_DIR, CACHE_DIR = HERE / "results", HERE / "cache"
REPO = "jhu-clsp/jfleg"
DEV = 300   # sentences 1-300 were inspected while designing the edit filter


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def toks(text: str) -> list[str]:
    return [t.lower() for t in tokenize(text)]


def gleu(sources: list[str], hyps: list[str], refs: list[str], max_n: int = 4) -> float:
    """Corpus GLEU for one reference set."""
    num = [0] * max_n
    den = [0] * max_n
    hyp_len = ref_len = 0
    for s, h, r in zip(sources, hyps, refs):
        s_t, h_t, r_t = toks(s), toks(h), toks(r)
        hyp_len += len(h_t)
        ref_len += len(r_t)
        for n in range(1, max_n + 1):
            h_ng, r_ng, s_ng = ngrams(h_t, n), ngrams(r_t, n), ngrams(s_t, n)
            source_only = s_ng - r_ng                      # n-grams that should have been changed
            num[n - 1] += max(0, sum((h_ng & r_ng).values()) - sum((h_ng & source_only).values()))
            den[n - 1] += max(0, len(h_t) - n + 1)
    if hyp_len == 0 or any(d == 0 for d in den):
        return 0.0
    log_p = sum(math.log(max(nu, 1e-9) / d) for nu, d in zip(num, den)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(log_p) * 100


def gleu_multi(sources, hyps, ref_sets) -> float:
    return sum(gleu(sources, hyps, refs) for refs in ref_sets) / len(ref_sets)


def load_jfleg(n: int, split: str = "test") -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision="refs/convert/parquet")
    test = sorted(f for f in files if f"/{split}/" in f"/{f}" and f.endswith(".parquet"))
    if not test:
        raise SystemExit(f"No test parquet in {REPO}: {files}")
    rows = []
    for f in test:
        table = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", revision="refs/convert/parquet"))
        rows += table.to_pylist()
    rows = [r for r in rows if r.get("sentence") and len(r.get("corrections") or []) >= 4][:n]
    for r in rows:
        r["sentence"] = r["sentence"].strip()
        r["corrections"] = [c.strip() for c in r["corrections"][:4]]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=747)
    parser.add_argument("--model", default=STAGE2_MODEL)
    parser.add_argument("--intensity", type=int, default=50)
    args = parser.parse_args()

    client = stage2.OllamaClient(model=args.model)
    st = client.status()
    if not st["online"] or not st["model_available"]:
        raise SystemExit(f"Ollama is not ready ({st}). Start Ollama and run: ollama pull {args.model}")
    rows = load_jfleg(args.n)
    cache = Cache(CACHE_DIR / "llm.jsonl")
    code = "en-US"

    def ask(text: str) -> str | None:
        key = f"gec|p{stage2.PROMPT_VERSION}|{client.model}|{text}"

        def run():
            try:
                return {"raw_output": client.chat(stage2.build_messages(text, code))}
            except Exception as exc:
                return {"raw_output": None, "error": str(exc)}
        return cache.get_or_compute(key, run).get("raw_output")

    out = {k: [] for k in ("source", "stage1", "stage2", "full", "llm_raw")}
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        src = r["sentence"]
        s1 = stage1.correct(src, code)
        raw_src, raw_s1 = ask(stage1.normalize(src, code)), ask(s1)
        only2 = stage2.finalize(stage1.normalize(src, code), raw_src, code, args.intensity, client.model).text \
            if raw_src else src
        r2 = stage2.finalize(s1, raw_s1, code, args.intensity, client.model) if raw_s1 else None
        full = (stage1.correct(r2.text, code) if r2.status == "applied" else r2.text) if r2 else s1   # as in the app
        out["source"].append(src)
        out["stage1"].append(s1)
        out["stage2"].append(only2)
        out["full"].append(full)
        out["llm_raw"].append(raw_s1 or s1)
        print(f"  JFLEG {i}/{len(rows)}", end="\r", flush=True)

    def score(lo: int, hi: int) -> dict:
        part = rows[lo:hi]
        if not part:
            return {}
        sources = [r["sentence"] for r in part]
        ref_sets = [[r["corrections"][k] for r in part] for k in range(4)]
        return {name: round(gleu_multi(sources, hyps[lo:hi], ref_sets), 2) for name, hyps in out.items()}

    splits = {"held_out": score(DEV, len(rows)), "dev": score(0, DEV), "all": score(0, len(rows))}
    splits = {k: v for k, v in splits.items() if v}
    scores = splits.get("held_out") or splits["all"]
    result = {"dataset": f"JFLEG test, {len(rows)} sentences (dev = 1-{DEV}, held-out = {DEV + 1}-{len(rows)})",
              "model": client.model, "metric": "GLEU (higher is better)", "scores": scores, "splits": splits,
              "minutes": round((time.perf_counter() - t0) / 60, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gec.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(OUT_DIR / "gec_details.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"source": r["sentence"], "references": r["corrections"],
                                **{k: v[i] for k, v in out.items() if k != "source"}}, ensure_ascii=False) + "\n")
    names = {"source": "No correction (source)", "stage1": "Stage 1 rules only", "stage2": "Stage 2 LLM + filter only",
             "full": "Full pipeline (rules + LLM + filter)", "llm_raw": "Rules + unfiltered LLM"}
    cols = list(splits)
    titles = {"held_out": f"Held-out ({DEV + 1}-{len(rows)})", "dev": f"Dev (1-{DEV})", "all": "All"}
    md = [f"# Grammar correction (JFLEG test, {len(rows)} sentences)", "",
          f"GLEU, higher is better. Stage 2 model `{client.model}`. The edit filter was designed on the dev "
          "sentences; the held-out score is the unbiased one.", "",
          "| System | " + " | ".join(titles[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    md += [f"| {names[k]} | " + " | ".join(f"{splits[c][k]:.2f}" for c in cols) + " |" for k in names]
    (OUT_DIR / "gec.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print()
    print("GLEU (higher is better):  " + "   ".join(titles[c] for c in cols))
    for k in names:
        print(f"{names[k]:40s} " + "   ".join(f"{splits[c][k]:6.2f}" for c in cols))
    print(f"Saved {OUT_DIR / 'gec.md'}")


if __name__ == "__main__":
    main()
