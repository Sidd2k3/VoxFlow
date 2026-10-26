"""Check the whole app (rules + real LLM + filter) on everyday mistakes in all five languages,
exactly as typed text goes through the website (language selected).

    python -m eval.check_languages              # short everyday mistakes
    python -m eval.check_languages --set real   # longer realistic sentences
    python -m eval.check_languages --set hard   # two sentences with 2-3 mistakes each

For each sentence: is the output the expected correction?
Writes eval/results/language_check.md.
"""

import argparse
from pathlib import Path

from app.languages import LANGUAGES
from app.pipeline import build_router

import re
import unicodedata

from .language_cases import ALSO_CORRECT, CASES, HARD_CASES, REAL_CASES


def same(a: str, b: str) -> bool:
    """Equal, ignoring the nukta dot (रोज = रोज़, both standard), a comma after a time word, and spacing."""
    def n(t):
        t = unicodedata.normalize("NFD", t).replace("\u093c", "").replace("\u0a3c", "")
        t = re.sub(r"^(Yesterday|Tomorrow|Today),", r"\1", t.strip())
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t))
    return n(a) == n(b)

OUT = Path(__file__).resolve().parent / "results" / "language_check.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["short", "real", "hard"], default="short")
    args = parser.parse_args()
    cases_by_lang = {"short": CASES, "real": REAL_CASES, "hard": HARD_CASES}[args.set]
    out_file = OUT if args.set == "short" else OUT.with_name(f"language_check_{args.set}.md")
    from app import stage2
    router = build_router(stage2.default_client(wait_on_rate_limit=True))
    st = router.llm.status()
    if not st["online"] or not st["model_available"]:
        raise SystemExit(f"No grammar model is reachable ({st}). Check the keys in .env, or open the Ollama app.")
    print(f"Grammar model: {st['model']}")
    md = [f"# Everyday mistakes in all five languages, set '{args.set}' (typed text, the whole app)", "",
          f"Grammar model: {st['model']}", "",
          "| Language | Input | Output | Correct |", "|---|---|---|---|"]
    totals = {}
    for code, cases in cases_by_lang.items():
        ok = 0
        for wrong, right in cases:
            out = router.process_text(wrong, code, 50)
            good = any(same(out["final_text"], r) for r in {right} | ALSO_CORRECT.get(wrong, set()))
            ok += good
            mark = "yes" if good else f"no (expected: {right})"
            md.append(f"| {LANGUAGES[code].name} | {wrong} | {out['final_text']} | {mark} |")
            print(f"{'OK  ' if good else 'MISS'} {code}  {wrong}  ->  {out['final_text']}"
                  + ("" if good else f"\n       expected: {right}   [{out['stage2']['status']}: {out['stage2']['model']}]"))
        totals[code] = (ok, len(cases))
    md += ["", "| Language | Corrected as expected |", "|---|---|"]
    md += [f"| {LANGUAGES[c].name} | {o}/{n} |" for c, (o, n) in totals.items()]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(md) + "\n", encoding="utf-8")
    print()
    for c, (o, n) in totals.items():
        print(f"{LANGUAGES[c].name:8s} corrected {o}/{n}")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
