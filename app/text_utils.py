"""Tokenising and joining text so that punctuation is its own token.

Treating "park." as two tokens ("park", ".") means adding a full stop shows up as a
small insertion instead of the whole word looking "modified".
"""

import re
import unicodedata

# Punctuation that attaches to the previous word (no space before it).
CLOSING = set(".,!?;:।॥)]}\"'”’")
# Punctuation that attaches to the next word (no space after it).
OPENING = set("([{“‘")
PUNCT = CLOSING | OPENING


# Dependent vowel signs (matras) of Devanagari, Bengali and Gurmukhi. Two in a row never occur in real
# text; speech recognisers sometimes write them (live: "খাইছেি"), so the second one is dropped.
_VOWEL_SIGNS = "[\u093E-\u094C\u0962\u0963\u09BE-\u09CC\u09E2\u09E3\u0A3E-\u0A4C]"
_DOUBLE_VOWEL_SIGN = re.compile(f"({_VOWEL_SIGNS}){_VOWEL_SIGNS}+")


def normalize_unicode(text: str) -> str:
    """NFC-normalise, drop doubled vowel signs and collapse whitespace."""
    text = _DOUBLE_VOWEL_SIGN.sub(r"\1", unicodedata.normalize("NFC", text))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split on whitespace, then peel leading/trailing punctuation into separate tokens.

    Apostrophes inside words ("doesn't") stay inside the word.
    """
    tokens: list[str] = []
    for chunk in normalize_unicode(text).split(" "):
        if not chunk:
            continue
        lead: list[str] = []
        trail: list[str] = []
        while chunk and chunk[0] in PUNCT:
            lead.append(chunk[0])
            chunk = chunk[1:]
        while chunk and chunk[-1] in PUNCT:
            trail.insert(0, chunk[-1])
            chunk = chunk[:-1]
        tokens.extend(lead)
        if chunk:
            tokens.append(chunk)
        tokens.extend(trail)
    return tokens


def join_tokens(tokens: list[str]) -> str:
    """Inverse of tokenize: put spaces back between words, not before closing punctuation."""
    out = ""
    for tok in tokens:
        if not out:
            out = tok
        elif tok in CLOSING or out[-1] in OPENING:
            out += tok
        else:
            out += " " + tok
    return out


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the end mark with each sentence."""
    parts = re.findall(r"[^.!?।]+[.!?।]*", text)
    return [p.strip() for p in parts if p.strip()]
