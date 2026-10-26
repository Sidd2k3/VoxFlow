"""Turn "raw transcript -> final text" into a list of corrections the user can accept or reject.

The response contains `segments`: the final text split into unchanged parts and changed parts.
Each changed part holds both versions, so the frontend rebuilds the text from the user's choices
(rejecting a correction puts the original words back). In the original app, accept/reject only
flipped a flag and never changed the text.

Each correction also says which stage made it ("rule" = Stage 1, "llm" = Stage 2), instead of the
original's hardcoded 88% confidence.
"""

from difflib import SequenceMatcher

from .text_utils import tokenize


def _llm_touched(stage1_tokens: list[str], final_tokens: list[str]) -> tuple[set[int], set[int]]:
    """Indices of final tokens changed by Stage 2, and final positions where Stage 2 deleted tokens."""
    changed, deletion_points = set(), set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, stage1_tokens, final_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        changed.update(range(j1, j2))
        if j1 == j2:
            deletion_points.add(j1)
    return changed, deletion_points


def build(raw: str, stage1: str, final: str) -> tuple[list[dict], list[dict]]:
    """Return (segments, corrections) describing how `raw` became `final`."""
    raw_t, s1_t, fin_t = tokenize(raw), tokenize(stage1), tokenize(final)
    llm_changed, llm_deleted_at = _llm_touched(s1_t, fin_t)

    segments: list[dict] = []
    corrections: list[dict] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, raw_t, fin_t, autojunk=False).get_opcodes():
        if tag == "equal":
            segments.append({"kind": "same", "tokens": fin_t[j1:j2]})
            continue
        by_llm = any(j in llm_changed for j in range(j1, j2)) or (j1 == j2 and (j1 in llm_deleted_at))
        correction = {
            "id": f"c{len(corrections) + 1}",
            "type": {"replace": "modification", "insert": "insertion", "delete": "deletion"}[tag],
            "original": raw_t[i1:i2],
            "corrected": fin_t[j1:j2],
            "source": "llm" if by_llm else "rule",
        }
        corrections.append(correction)
        segments.append({"kind": "change", "id": correction["id"]})
    return segments, corrections
