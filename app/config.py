"""Settings read from environment variables (or a .env file in the project root)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
STAGE2_MODEL = os.getenv("STAGE2_MODEL", "llama3.2")
# Languages where the LOCAL model (llama3.2) is not used; the cloud models are used for all five.
STAGE2_SKIP_REASONS = {
    # lost accuracy on the IndicGEC dev and test sets with every model tried (llama3.2, gemma3:12b)
    "bn-IN": "it made Bengali text less accurate in testing",
    # no measured gain (FLEURS: rules 23.4% WER, + LLM 23.6%) and the slowest language for llama3.2
    # (Gurmukhi becomes many tokens: 5.4 s median per sentence vs 0.25 s for the speech recognition)
    "pa-IN": "it did not make Punjabi text more accurate in testing and took 4-5 s",
    # IndicGEC dev: 88.78 with no correction vs 88.13 with llama3.2; 1000 generated mistakes (seed 2):
    # rules alone 96% fixed / 0 false alarms, with the LLM 55% / 93 false alarms (it rewrote correct
    # sentences, e.g. गया -> गया था, बैठे -> बैठेंगे, and undid correct rule fixes)
    "hi-IN": "it changed correct Hindi sentences and undid correct fixes in testing",
    # 1000 generated mistakes (seed 2): rules alone 97% / 0 false alarms, with the LLM 90% / 17
    # (आहात -> आहेत, गेला -> गेला होता)
    "mr-IN": "it changed correct Marathi sentences in testing",
}
STAGE2_SKIP_LANGUAGES = {c.strip() for c in os.getenv("STAGE2_SKIP_LANGUAGES", "bn-IN,pa-IN,hi-IN,mr-IN").split(",")
                         if c.strip()}
STAGE2_TIMEOUT = float(os.getenv("STAGE2_TIMEOUT", "60"))

# Large free cloud LLMs for the grammar step (Stage 2); used first when a key is set, in this order:
# Gemini (aistudio.google.com/apikey) -> Groq (console.groq.com/keys) -> the local Ollama model above.
# Only text is sent, never audio. Without keys the app works offline as before.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()     # empty = newest free Flash-Lite, chosen automatically
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()         # empty = llama-3.3-70b-versatile (or the best available)
# which cloud model is asked first. Measured on the hard set (2 sentences, 2-3 mistakes each, 5 languages):
# Groq gpt-oss-120b 18/20 in 0.6 s, Gemini 3.5 Flash-Lite 18/20 in 2.3 s -> Groq first, Gemini as the backup
STAGE2_ORDER = [p.strip() for p in os.getenv("STAGE2_ORDER", "groq,gemini").split(",") if p.strip()]

# Speech recognition engine:
#   "routed" (default) - each language goes to the engine that measured best on FLEURS (ASR_ROUTES)
#   "whisper"          - local Whisper for every language
#   "google"           - Google's free web endpoint for every language (needs internet)
ASR_ENGINE = os.getenv("ASR_ENGINE", "routed")


def _default_english_asr() -> str:
    """Whisper large-v3-turbo on the Apple Silicon GPU without per-word timing when available. Measured on
    60 Indian-accented clips (Svarah): 0.95 s and 4.9% WER, vs Google 1.34 s / 13.0%, large-v3 3.25 s / 4.1%,
    Parakeet v3 0.20 s / 9.4% (README). Elsewhere large-v3-turbo on the CPU."""
    import importlib.util
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_whisper"):
        return "mlx:large-v3-turbo:fast"
    return "whisper"


# English engine: "mlx:large-v3-turbo:fast" (default on a Mac), "mlx:large-v3", "parakeet:v3" or "whisper" (CPU)
ENGLISH_ASR = os.getenv("ENGLISH_ASR") or _default_english_asr()
# Indian languages: AI4Bharat IndicConformer on this computer, Google (Chrome's engine) if it can't load.
# Decoding mode per language, measured on 30 FLEURS clips each (README): RNNT for Hindi (11.1% WER vs 11.7%
# with CTC) and Marathi (15.5% vs 17.9%); CTC for Punjabi (15.9% vs 16.3%).
# Bengali uses Google: IndicConformer was better on FLEURS (13.1% vs 16.7%) but misheard short spoken
# sentences in the live test (2 of 3, "গতকাল" -> "কতকাল", "বই পড়ে" -> "বোয়ে পরে") that Google heard right.
INDIAN_ASR_DEFAULT = {"hi-IN": "indicconformer:rnnt|google", "mr-IN": "indicconformer:rnnt|google",
                      "bn-IN": "google", "pa-IN": "indicconformer:ctc|google"}
INDIAN_ASR = os.getenv("INDIAN_ASR")          # one engine for all four, e.g. "google" (overrides the above)
ASR_ROUTES = {"en-US": ENGLISH_ASR, **{c: INDIAN_ASR or spec for c, spec in INDIAN_ASR_DEFAULT.items()}}
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")   # Whisper size used for transcription
# Prompt Whisper to transcribe English verbatim, keeping the speaker's grammar mistakes (see asr.py)
WHISPER_VERBATIM = os.getenv("WHISPER_VERBATIM", "0").lower() in {"1", "true", "yes"}

# Save every recording + what was heard to eval/live/ (for eval/live_check.py). Off by default: it's your voice.
SAVE_RECORDINGS = os.getenv("SAVE_RECORDINGS", "0").lower() in {"1", "true", "yes"}
LIVE_DIR = ROOT / "eval" / "live"

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8010"))   # 8000 is often taken by other projects

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 2000

FEWSHOT_DIR = ROOT / "data" / "fewshot"
RESULTS_FILE = ROOT / "eval" / "results" / "results.json"
