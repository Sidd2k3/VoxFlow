"""The correction pipeline, one strategy object per language (strategy pattern).

The original had five copy-pasted classes that differed only in a file name and a language
word. Here one class is configured per language and registered in a router.

    audio -> ASR -> Stage 1 (rules) -> Stage 2 (LLM + guard) -> corrections -> response
    (Stage 2 = free cloud LLMs when keys are set: Gemini -> Groq -> local Ollama; see cloud_llm.py)
    text  ---------> Stage 1 (rules) -> Stage 2 (LLM + guard) -> corrections -> response
"""

import time

from . import asr, config, diff, stage1, stage2
from .audio import prepare
from .languages import LANGUAGES, Language, get_language


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class LanguagePipeline:
    """Correction strategy for one language."""

    def __init__(self, language: Language, llm: stage2.OllamaClient):
        self.language = language
        self.llm = llm

    def correct(self, text: str, intensity: int, alternatives: list[str] | None = None,
                uncertain: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        s1 = stage1.correct(text, self.language.code)
        stage1_ms = _ms(t0)

        t0 = time.perf_counter()
        supports = getattr(self.llm, "supports", None)
        if not (supports(self.language.code) if supports else self.language.code not in config.STAGE2_SKIP_LANGUAGES):
            why = config.STAGE2_SKIP_REASONS.get(self.language.code, "it is switched off in the settings")
            s2 = stage2.Stage2Result(s1, "skipped", f"the LLM step is off for {self.language.name} because {why} "
                                     f"(rules only)", self.llm.model)
        else:
            s2 = stage2.correct(s1, self.language.code, intensity, self.llm, alternatives, uncertain)
        # the rules run once more on the LLM's answer: it may fix one part (जाता -> गया) and leave
        # a part the rules know (गया हूँ -> गया था with कल)
        final = stage1.correct(s2.text, self.language.code) if s2.status == "applied" else \
            stage1.normalize(s2.text, self.language.code)
        stage2_ms = _ms(t0)

        segments, corrections = diff.build(text, s1, final)
        return {
            "language": self.language.code,
            "language_name": self.language.name,
            "stage1_text": s1,
            "stage2_text": final,
            "final_text": final,
            "stage2": {"status": s2.status, "reason": s2.reason, "model": s2.model, "raw_output": s2.raw_output,
                       "kept_edits": s2.kept_edits, "reverted_edits": s2.reverted_edits},
            "segments": segments,
            "corrections": corrections,
            "latency_ms": {"stage1": stage1_ms, "stage2": stage2_ms},
        }


class PipelineRouter:
    def __init__(self, llm: stage2.OllamaClient | None = None, recognizer=None):
        self.llm = llm or stage2.default_client()
        self.recognizer = recognizer or asr.default_recognizer()
        self.strategies: dict[str, LanguagePipeline] = {}

    def register(self, pipeline: LanguagePipeline) -> None:
        self.strategies[pipeline.language.code] = pipeline

    def get(self, code: str) -> LanguagePipeline:
        get_language(code)  # raises ValueError for unknown codes (no silent English fallback)
        return self.strategies[code]

    def process_text(self, text: str, lang_code: str, intensity: int, heard: dict | None = None) -> dict:
        """Correct text in the given language. `heard` carries the recogniser's details when the text came
        from /api/transcribe (alternatives, unsure words, timing), so the result is the same as /api/audio."""
        t0 = time.perf_counter()
        heard = heard or {}
        result = self.get(lang_code).correct(text, intensity, heard.get("asr_alternatives"),
                                             heard.get("asr_uncertain"))
        result.update({"recognized_text": text, "asr_confidence": None, "preprocessed": False, "input": "text"})
        result.update({k: v for k, v in heard.items() if k not in ("latency_ms", "stage1_text")})
        result["latency_ms"].update({"asr": heard.get("latency_ms", {}).get("asr", 0),
                                     "total": _ms(t0) + heard.get("latency_ms", {}).get("asr", 0)})
        return result

    def transcribe_audio(self, wav_bytes: bytes, lang_code: str) -> dict:
        """Speech -> text only (fast), so the screen can show what was heard before the correction."""
        t0 = time.perf_counter()
        samples, duration = prepare(wav_bytes)
        heard = asr.transcribe(samples, lang_code, self.recognizer)
        # the rules take ~0 ms, so their correction is shown at once too; the LLM's comes with /api/text
        return {"recognized_text": heard.text, "language": heard.language, "asr_confidence": heard.confidence,
                "stage1_text": stage1.correct(heard.text, heard.language) if heard.text else "",
                "preprocessed": heard.preprocessed, "asr_alternatives": heard.alternatives,
                "asr_uncertain": heard.uncertain, "asr_engine": heard.engine,
                "audio_seconds": round(duration, 2), "input": "audio", "latency_ms": {"asr": _ms(t0)},
                "llm_step": self._llm_step(heard.language)}

    def _llm_step(self, code: str) -> bool:
        """Will the grammar model be asked for this language? (the screen says so while it works)"""
        supports = getattr(self.llm, "supports", None)
        return bool(supports(code)) if supports else code not in config.STAGE2_SKIP_LANGUAGES

    def process_audio(self, wav_bytes: bytes, lang_code: str, intensity: int) -> dict:
        heard = self.transcribe_audio(wav_bytes, lang_code)
        if not heard["recognized_text"]:
            return heard
        return self.process_text(heard["recognized_text"], heard["language"], intensity, heard)


def build_router(llm: stage2.OllamaClient | None = None, recognizer=None) -> PipelineRouter:
    router = PipelineRouter(llm, recognizer)
    for language in LANGUAGES.values():
        router.register(LanguagePipeline(language, router.llm))
    return router
