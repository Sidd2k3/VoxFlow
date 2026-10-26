"""The free cloud LLMs (Gemini -> Groq -> local), with the network replaced by fakes."""

import json

import pytest

from app import cloud_llm, config, stage2
from app.cloud_llm import ChainClient, GeminiClient, GroqClient, RateLimited, pick_gemini_model, pick_groq_model


class Resp:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}
        self.text = text or json.dumps(data or {})

    def json(self):
        return self._data


def gemini_answer(text):
    return Resp(data={"candidates": [{"content": {"parts": [{"text": json.dumps({"corrected": text},
                                                                                   ensure_ascii=False)}]}}]})


def test_picks_newest_stable_flash_lite():
    names = ["models/gemini-2.5-flash", "models/gemini-2.5-flash-lite", "models/gemini-3.5-flash-lite",
             "models/gemini-3.5-flash-lite-preview-09-2026", "models/gemini-3.8-flash", "models/gemini-3.5-flash-tts",
             "models/gemini-3.1-pro", "models/text-embedding-004", "models/gemini-3.5-flash-live"]
    assert pick_gemini_model(names) == "gemini-3.5-flash-lite"
    assert pick_gemini_model(["models/gemini-3.8-flash", "models/gemini-3.8-flash-tts"]) == "gemini-3.8-flash"
    assert pick_gemini_model(["models/gemini-3.1-pro"]) is None


def test_picks_groq_model():
    assert pick_groq_model(["whisper-large-v3", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]) == \
        "llama-3.3-70b-versatile"
    assert pick_groq_model(["openai/gpt-oss-120b", "llama-3.1-8b-instant"]) == "openai/gpt-oss-120b"


def test_gemini_chat_sends_system_and_reads_json(monkeypatch):
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return gemini_answer("मेरे पापा रोज़ सुबह पार्क में टहलते हैं।")
    monkeypatch.setattr(cloud_llm.httpx, "post", post)
    client = GeminiClient(key="k", model="gemini-3.5-flash-lite")
    out = client.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "Text: x"},
                       {"role": "assistant", "content": "{}"}, {"role": "user", "content": "Text: y"}], "hi-IN")
    assert out == "मेरे पापा रोज़ सुबह पार्क में टहलते हैं।"
    assert sent["url"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert sent["headers"] == {"x-goog-api-key": "k"}            # key in a header, not in the URL
    assert sent["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert [c["role"] for c in sent["body"]["contents"]] == ["user", "model", "user"]


def test_gemini_falls_back_to_simpler_request_settings(monkeypatch):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        g = json["generationConfig"]
        calls.append(("thinkingConfig" in g, "responseSchema" in g))
        # this model accepts neither a thinking budget nor a schema
        return Resp(400, text="Request contains an invalid argument.") if len(calls) < 3 else gemini_answer("ok")
    monkeypatch.setattr(cloud_llm.httpx, "post", post)
    client = GeminiClient(key="k", model="m")
    assert client.chat([{"role": "user", "content": "x"}]) == "ok"
    assert calls == [(True, True), (False, True), (False, False)]
    client.chat([{"role": "user", "content": "x"}])          # remembered: straight to the setting that worked
    assert calls[-1] == (False, False) and len(calls) == 4


def test_gemini_reports_a_400_that_no_setting_fixes(monkeypatch):
    monkeypatch.setattr(cloud_llm.httpx, "post", lambda *a, **k: Resp(400, text="bad"))
    with pytest.raises(cloud_llm.CloudLLMError, match="400"):
        GeminiClient(key="k", model="m").chat([{"role": "user", "content": "x"}])


def test_rate_limit_falls_back_to_groq(monkeypatch):
    def post(url, json=None, headers=None, timeout=None):
        if "generativelanguage" in url:
            return Resp(429, text="quota", headers={"retry-after": "30"})
        return Resp(data={"choices": [{"message": {"content": '{"corrected": "माझा भाऊ काल शाळेत गेला."}'}}]})
    monkeypatch.setattr(cloud_llm.httpx, "post", post)
    gem, groq = GeminiClient(key="k", model="g"), GroqClient(key="k", model="llama-3.3-70b-versatile")
    chain = ChainClient([gem, groq])
    assert chain.chat([{"role": "user", "content": "x"}], "mr-IN") == "माझा भाऊ काल शाळेत गेला."
    assert chain.last_model == "llama-3.3-70b-versatile" and chain.last_strong
    assert not gem.supports("mr-IN")            # skipped for a while after the rate limit
    with pytest.raises(RateLimited):
        GeminiClient(key="k", model="g").chat([{"role": "user", "content": "x"}])


def test_chain_skips_local_model_for_indian_languages(monkeypatch):
    from tests.conftest import FakeLLM
    local = FakeLLM("wrong")
    local.supports = lambda code: code == "en-US"
    chain = ChainClient([local])
    assert not chain.supports("hi-IN") and chain.supports("en-US")
    with pytest.raises(cloud_llm.CloudLLMError):
        chain.chat([{"role": "user", "content": "x"}], "hi-IN")
    assert local.calls == []


def test_no_keys_means_local_model_only(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert isinstance(stage2.default_client(), stage2.OllamaClient)
    monkeypatch.setattr(config, "GROQ_API_KEY", "x")
    client = stage2.default_client()
    assert [type(c).__name__ for c in client.clients] == ["GroqClient", "OllamaClient"]
    monkeypatch.setattr(config, "GEMINI_API_KEY", "y")
    client = stage2.default_client()
    assert [type(c).__name__ for c in client.clients] == ["GroqClient", "GeminiClient", "OllamaClient"]
    monkeypatch.setattr(config, "STAGE2_ORDER", ["gemini", "groq"])
    assert [type(c).__name__ for c in stage2.default_client().clients][:2] == ["GeminiClient", "GroqClient"]


class StrongFake:
    """A large model: answers with a fixed text."""
    strong, name = True, "fake-cloud"

    def __init__(self, answer):
        self.answer, self.model, self.calls = answer, "fake-cloud", []

    def supports(self, code):
        return True

    def chat(self, messages, lang_code=None):
        self.calls.append(messages)
        return self.answer

    def status(self):
        return {"online": True, "model": self.model, "model_available": True}


def run(code, heard, answer):
    from app.languages import get_language
    from app.pipeline import LanguagePipeline
    llm = ChainClient([StrongFake(answer)])
    return LanguagePipeline(get_language(code), llm).correct(heard, 50), llm


def test_strong_model_fixes_several_mistakes_in_two_sentences():
    heard = "मेरी बहन कल बाजार जाता है। वह वहाँ से सब्जी खरीदा।"
    fixed = "मेरी बहन कल बाजार गई थी। उसने वहाँ से सब्जी खरीदी।"
    out, llm = run("hi-IN", heard, fixed)
    assert out["final_text"] == fixed and out["stage2"]["status"] == "applied"
    assert "several" in llm.clients[0].calls[0][0]["content"]     # the large-model prompt was used


def test_strong_model_is_used_for_all_indian_languages():
    for code, heard, fixed in [("mr-IN", "माझा भाऊ काल शाळेत गेली।", "माझा भाऊ काल शाळेत गेला।"),
                               ("bn-IN", "আমি গতকাল বাজারে যাই।", "আমি গতকাল বাজারে গিয়েছিলাম।"),
                               ("pa-IN", "ਕੁੜੀ ਸਕੂਲ ਜਾਂਦਾ ਹੈ।", "ਕੁੜੀ ਸਕੂਲ ਜਾਂਦੀ ਹੈ।")]:
        out, _ = run(code, heard, fixed)
        assert out["final_text"] == fixed, code


def test_strong_model_may_repair_a_misheard_word():
    out, _ = run("hi-IN", "मेरे पापा रोजे पार्क में टहलती हैं।", "मेरे पापा रोज़ पार्क में टहलते हैं।")
    assert out["final_text"] == "मेरे पापा रोज़ पार्क में टहलते हैं।"


def test_strong_model_cannot_change_the_speakers_gender():
    out, _ = run("hi-IN", "मैं रोज़ स्कूल जाती हूँ।", "मैं रोज़ स्कूल जाता हूँ।")
    assert out["final_text"] == "मैं रोज़ स्कूल जाती हूँ।"


def test_strong_model_translation_or_rewrite_is_rejected():
    out, _ = run("hi-IN", "लड़की स्कूल जाता है।", "The girl goes to school.")
    assert out["stage2"]["status"] == "rejected" and out["final_text"] == "लड़की स्कूल जाती है।"   # rules still fix it
    out, _ = run("en-US", "He go to school every day.", "Every single morning, he walks to his school.")
    assert out["stage2"]["status"] == "rejected"


def test_cloud_failure_keeps_the_rules_result():
    class Down(StrongFake):
        def chat(self, messages, lang_code=None):
            raise cloud_llm.CloudLLMError("offline")
    from app.languages import get_language
    from app.pipeline import LanguagePipeline
    out = LanguagePipeline(get_language("mr-IN"), ChainClient([Down("")])).correct("ती घरी जातो।", 50)
    assert out["final_text"] == "ती घरी जाते।" and out["stage2"]["status"] == "unavailable"


def test_speaker_gender_guard_is_limited_to_the_speakers_clause():
    # the friend's verb is not the speaker's
    out, _ = run("hi-IN", "मैं कल दिल्ली जाऊँगा और मेरी दोस्त भी मेरे साथ आएगा।",
                 "मैं कल दिल्ली जाऊँगा और मेरी दोस्त भी मेरे साथ आएगी।")
    assert out["final_text"] == "मैं कल दिल्ली जाऊँगा और मेरी दोस्त भी मेरे साथ आएगी।"
    # Punjabi ਮੈਂ + past transitive: the verb agrees with ਰੋਟੀ
    out, _ = run("pa-IN", "ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧਾ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤਾ।", "ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧੀ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤੀ।")
    assert out["final_text"] == "ਮੈਂ ਕੱਲ੍ਹ ਰੋਟੀ ਖਾਧੀ। ਮੇਰੇ ਭਰਾ ਨੇ ਚਾਹ ਪੀਤੀ।"
    # but the speaker's own present-tense verb stays as spoken
    out, _ = run("pa-IN", "ਮੈਂ ਸਕੂਲ ਜਾਂਦੀ ਹਾਂ।", "ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ।")
    assert out["final_text"] == "ਮੈਂ ਸਕੂਲ ਜਾਂਦੀ ਹਾਂ।"
