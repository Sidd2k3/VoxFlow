"""Word error rate (WER) and character error rate (CER), implemented from scratch.

WER = (substitutions + deletions + insertions) / number of words in the reference.
Before scoring, both texts are normalised the same way (lower-case, punctuation removed),
so adding a full stop or a capital letter is not counted as an error.
CER is reported as well because word boundaries in Indic scripts are less consistent.
"""

import re
import unicodedata


def normalize_for_scoring(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") or ch in "।॥" else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    errors = words = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize_for_scoring(ref).split(), normalize_for_scoring(hyp).split()
        errors += edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    errors = chars = 0
    for ref, hyp in zip(refs, hyps):
        r = list(normalize_for_scoring(ref).replace(" ", ""))
        h = list(normalize_for_scoring(hyp).replace(" ", ""))
        errors += edit_distance(r, h)
        chars += len(r)
    return errors / chars if chars else 0.0


def edit_ops(ref: list, hyp: list) -> dict:
    """Count substitutions, deletions (reference words missing from the output) and insertions."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    d = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = {"sub": 0, "del": 0, "ins": 0}
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops["sub"] += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops["del"] += 1
            i -= 1
        else:
            ops["ins"] += 1
            j -= 1
    return ops


def corpus_missed_words(refs: list[str], hyps: list[str]) -> float:
    """Share of spoken (reference) words that are missing from the transcript."""
    missed = words = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize_for_scoring(ref).split(), normalize_for_scoring(hyp).split()
        missed += edit_ops(r, h)["del"]
        words += len(r)
    return missed / words if words else 0.0
