"""Does the sound-alike spelling repair (stage1.repair_spelling, e.g. বারি -> বাড়ি) help or hurt?

    python -m eval.spelling_check

Compares the grammar rules with the repair off and on, for Bengali and Hindi:
  - speech: the 30 FLEURS clips per language as IndicConformer heard them (from eval.compare_asr's
    cache; run that first), word error rate against the human transcript, lower is better;
  - text: the IndicGEC 2025 test sentences, shared-task GLEU against the human correction, higher is better.
Prints every word the repair changed, so each change can be checked by eye.
Writes eval/results/spelling_check.md.
"""

import json
from pathlib import Path

from app import stage1
from app.text_utils import tokenize

from .metrics import corpus_wer
from .run_eval import Cache
from .run_indic_gec import load, shared_task_gleu

HERE = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR, CACHE_DIR = HERE / "data", HERE / "results", HERE / "cache"
LANGS = {"bn-IN": "Bengali", "hi-IN": "Hindi"}


def changes(before: str, after: str) -> list[str]:
    a, b = tokenize(before), tokenize(after)
    return [f"{x} -> {y}" for x, y in zip(a, b) if x != y] if len(a) == len(b) else []


def main():
    cache = Cache(CACHE_DIR / "asr.jsonl")
    md = ["# Sound-alike spelling repair: off vs on", "",
          "| Language | FLEURS speech WER, off | on | IndicGEC text GLEU, off | on |", "|---|---|---|---|---|"]
    for code, name in LANGS.items():
        clips = [json.loads(l) for l in (DATA_DIR / code / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        refs, off, on, changed = [], [], [], []
        for clip in clips:
            heard = cache.data.get(f"indicconformer|{code}|{clip['file']}")
            if heard is None:
                raise SystemExit("Run first: python -m eval.compare_asr")
            a, b = stage1.correct(heard["text"], code, spelling=False), stage1.correct(heard["text"], code, spelling=True)
            refs.append(clip["reference"]); off.append(a); on.append(b)
            changed += changes(a, b)
        rows = load(code)
        g_refs = [r["reference"] for r in rows]
        g_off = [stage1.correct(r["source"], code, spelling=False) for r in rows]
        g_on = [stage1.correct(r["source"], code, spelling=True) for r in rows]
        g_changed = [c for x, y in zip(g_off, g_on) for c in changes(x, y)]
        line = (f"| {name} | {corpus_wer(refs, off):.2%} | {corpus_wer(refs, on):.2%} | "
                f"{shared_task_gleu(g_refs, g_off):.2f} | {shared_task_gleu(g_refs, g_on):.2f} |")
        md.append(line)
        print(line)
        print(f"  {name} speech, words changed ({len(changed)}): " + "; ".join(changed))
        print(f"  {name} text, words changed ({len(g_changed)}): " + "; ".join(g_changed[:40]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "spelling_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / 'spelling_check.md'}")


if __name__ == "__main__":
    main()
