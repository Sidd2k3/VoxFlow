import asyncio

import numpy as np

from app import tts


def test_tts_falls_back_to_google_voice(monkeypatch):
    async def no_voice(text, voice):
        raise tts.TTSError("No audio was received")
    monkeypatch.setattr(tts, "_edge", no_voice)
    monkeypatch.setattr(tts, "_gtts", lambda text, lang: b"ID3fake-mp3-" + lang.encode())
    audio, voice = asyncio.run(tts.synthesize("ਮੈਂ ਜਾਂਦਾ ਹਾਂ।", "pa-IN"))
    assert audio == b"ID3fake-mp3-pa" and voice == "Google TTS (pa)"


def test_tts_uses_microsoft_voice_first(monkeypatch):
    async def ok(text, voice):
        return b"mp3"
    monkeypatch.setattr(tts, "_edge", ok)
    assert asyncio.run(tts.synthesize("Hello", "en-US")) == (b"mp3", "en-US-AriaNeural")


def test_tts_reports_both_failures(monkeypatch):
    async def fail(text, voice):
        raise RuntimeError("offline")
    def fail_g(text, lang):
        raise RuntimeError("offline too")
    monkeypatch.setattr(tts, "_edge", fail)
    monkeypatch.setattr(tts, "_gtts", fail_g)
    try:
        asyncio.run(tts.synthesize("Hello", "en-US"))
        assert False
    except tts.TTSError as exc:
        assert "offline" in str(exc) and "offline too" in str(exc)


def test_whisper_recognizer_reports_confidence_and_uncertain_words(monkeypatch):
    from types import SimpleNamespace as W
    from app import asr, whisper_models

    class FakeModel:
        def transcribe(self, audio, language, **kw):
            self.language, self.kw = language, kw
            words = [W(word=" C", probability=0.21), W(word=" doesn't", probability=0.95),
                     W(word=" know.", probability=0.9)]
            return iter([W(text=" C doesn't know.", avg_logprob=-0.3, words=words)]), None

    model = FakeModel()
    monkeypatch.setattr(whisper_models, "get", lambda size: model)
    text, conf, alts, uncertain = asr.WhisperRecognizer("fake").recognize(np.zeros(16000, np.float32), "en-US")
    assert text == "C doesn't know." and model.language == "en" and model.kw["word_timestamps"] is True
    assert uncertain == ["C"] and alts == [] and abs(conf - (0.21 + 0.95 + 0.9) / 3) < 1e-3


def test_whisper_load_failure_is_a_service_error(monkeypatch):
    from app import asr, whisper_models

    def broken(size):
        raise whisper_models.WhisperLoadError("no model")
    monkeypatch.setattr(whisper_models, "get", broken)
    try:
        asr.WhisperRecognizer("fake").recognize(np.zeros(16000, np.float32), "en-US")
        assert False
    except asr.ASRServiceError:
        pass



def test_verbatim_prompt_only_for_english(monkeypatch):
    from types import SimpleNamespace as W
    from app import asr, whisper_models
    seen = {}

    class FakeModel:
        def transcribe(self, audio, language, **kw):
            seen[language] = kw.get("initial_prompt")
            return iter([W(text=" ok", avg_logprob=0.0, words=[W(word=" ok", probability=0.9)])]), None

    monkeypatch.setattr(whisper_models, "get", lambda size: FakeModel())
    rec = asr.make_engine("whisper:fake:verbatim")
    rec.recognize(np.zeros(16000, np.float32), "en-US")
    rec.recognize(np.zeros(16000, np.float32), "hi-IN")
    assert seen["en"] == asr.WhisperRecognizer.VERBATIM_PROMPT and seen["hi"] is None
    assert asr.make_engine("whisper:fake").verbatim is False
