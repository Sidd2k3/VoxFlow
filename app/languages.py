"""The five supported languages and everything the app needs to know about each one."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str          # BCP-47 code used by Google ASR, e.g. "hi-IN"
    name: str          # English name shown in the UI
    script: str        # writing system, used in the LLM prompt
    script_range: tuple[int, int] | None  # Unicode block of the script (None = Latin)
    voice: str         # edge-tts (Microsoft) neural voice for Speak Back, tried first
    end_mark: str      # sentence-final punctuation added by Stage 1
    gtts: str          # Google Translate TTS language code, the fallback voice


LANGUAGES: dict[str, Language] = {
    "en-US": Language("en-US", "English", "Latin", None, "en-US-AriaNeural", ".", "en"),
    "hi-IN": Language("hi-IN", "Hindi", "Devanagari", (0x0900, 0x097F), "hi-IN-SwaraNeural", "।", "hi"),
    "mr-IN": Language("mr-IN", "Marathi", "Devanagari", (0x0900, 0x097F), "mr-IN-AarohiNeural", "।", "mr"),
    "bn-IN": Language("bn-IN", "Bengali", "Bengali", (0x0980, 0x09FF), "bn-IN-TanishaaNeural", "।", "bn"),
    "pa-IN": Language("pa-IN", "Punjabi", "Gurmukhi", (0x0A00, 0x0A7F), "pa-IN-VaaniNeural", "।", "pa"),
}

def get_language(code: str) -> Language:
    """Return the language for a code, raising ValueError for unknown codes (no silent fallback)."""
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"Unsupported language '{code}'. Use one of: {', '.join(LANGUAGES)}.")


def script_ratio(text: str, lang: Language) -> float:
    """Fraction of letters in `text` that belong to the language's script (1.0 = all of them)."""
    letters = [ch for ch in text if ch.isalpha() or _is_combining_indic(ch)]
    if not letters:
        return 0.0
    if lang.script_range is None:
        hits = sum(1 for ch in letters if ord(ch) < 0x0250)  # Basic Latin + Latin extensions
    else:
        lo, hi = lang.script_range
        hits = sum(1 for ch in letters if lo <= ord(ch) <= hi)
    return hits / len(letters)


def _is_combining_indic(ch: str) -> bool:
    # Indic vowel signs and viramas are not "alpha" in Python, but they are part of words.
    return 0x0900 <= ord(ch) <= 0x0DFF
