from app.asr import ASRServiceError
from tests.conftest import FakeLLM, FakeRecognizer, tone_wav


def test_text_endpoint_runs_both_stages(client_with):
    client = client_with(FakeLLM("There are many problems."))
    r = client.post("/api/text", json={"text": "there is many problem", "language": "en-US"})
    assert r.status_code == 200
    data = r.json()
    assert data["stage1_text"] == "There are many problem."
    assert data["final_text"] == "There are many problems."
    assert data["stage2"]["status"] == "applied"
    assert {c["source"] for c in data["corrections"]} == {"rule", "llm"}
    assert data["latency_ms"]["total"] >= 0 and data["asr_confidence"] is None


def test_text_validation(client_with):
    client = client_with()
    assert client.post("/api/text", json={"text": "  "}).status_code == 400
    assert client.post("/api/text", json={"text": "a" * 2001}).status_code == 400
    assert client.post("/api/text", json={"text": "hi", "language": "xx-XX"}).status_code == 400
    assert client.post("/api/text", json={"text": "hi", "intensity": 101}).status_code == 422


def test_audio_endpoint(client_with):
    client = client_with(FakeLLM(), FakeRecognizer({"en-US": ("i go to park yesterday", 0.91)}))
    r = client.post("/api/audio", files={"file": ("a.wav", tone_wav(1.0, 48000), "audio/wav")},
                    data={"language": "en-US", "intensity": "50"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["recognized_text"] == "i go to park yesterday"
    assert data["final_text"] == "I went to the park yesterday."
    assert data["asr_confidence"] == 0.91 and data["audio_seconds"] == 1.0


def test_audio_errors_are_specific(client_with):
    client = client_with(recognizer=FakeRecognizer({}))
    wav = {"file": ("a.wav", tone_wav(1.0), "audio/wav")}
    assert client.post("/api/audio", files=wav).status_code == 422             # nothing heard
    assert client.post("/api/audio", files={"file": ("a.mp3", b"ID3...", "audio/mpeg")}).status_code == 400

    class Down:
        def recognize(self, samples, code):
            raise ASRServiceError("no internet")
    assert client_with(recognizer=Down()).post("/api/audio", files=wav).status_code == 502


def test_tts_failure_is_an_error_not_empty_audio(client_with, monkeypatch):
    from app import main
    from app.tts import TTSError

    async def broken(text, lang):
        raise TTSError("offline")
    monkeypatch.setattr(main, "synthesize", broken)
    r = client_with().post("/api/tts", json={"text": "Hello", "language": "en-US"})
    assert r.status_code == 502


def test_status_languages_metrics_and_website(client_with):
    client = client_with()
    assert client.get("/api/status").json()["stage2"]["online"] is True
    assert len(client.get("/api/languages").json()) == 5
    assert "available" in client.get("/api/metrics").json()
    page = client.get("/")
    assert page.status_code == 200 and "VoxFlow" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_rules_run_again_on_the_llm_answer(monkeypatch):
    # live case: the LLM fixed जाता -> गया but left हूँ; with कल the rules make it गया था
    # (the LLM is off for Hindi by default; switched on here to test the hand-over)
    from app import config
    monkeypatch.setattr(config, "STAGE2_SKIP_LANGUAGES", set())
    from app.pipeline import LanguagePipeline
    from app.languages import get_language
    from tests.conftest import FakeLLM
    out = LanguagePipeline(get_language("hi-IN"), FakeLLM("मैं कल बाजार गया हूँ।")).correct("मैं कल बाजार जाता हूँ।", 50)
    assert out["final_text"] == "मैं कल बाजार गया था।"


def test_bengali_uses_rules_only():
    from app.pipeline import LanguagePipeline
    from app.languages import get_language
    from tests.conftest import FakeLLM
    llm = FakeLLM("anything")
    out = LanguagePipeline(get_language("bn-IN"), llm).correct("আমি বাড়ি যাই।", 50)
    assert out["stage2"]["status"] == "skipped" and llm.calls == [] and "less accurate" in out["stage2"]["reason"]


def test_transcribe_then_correct_matches_the_one_step_audio_call(client_with):
    client = client_with(FakeLLM("She knows me."), FakeRecognizer({"en-US": ("she know me", 0.9)}))
    wav = {"file": ("a.wav", tone_wav(1.0), "audio/wav")}
    heard = client.post("/api/transcribe", files=wav, data={"language": "en-US"})
    assert heard.status_code == 200 and heard.json()["recognized_text"] == "she know me"
    assert heard.json()["stage1_text"] == "She knows me."          # the rules' fix is shown at once
    two_step = client.post("/api/text", json={"text": heard.json()["recognized_text"], "language": "en-US",
                                              "heard": heard.json()}).json()
    one_step = client.post("/api/audio", files={"file": ("a.wav", tone_wav(1.0), "audio/wav")},
                           data={"language": "en-US"}).json()
    assert two_step["final_text"] == one_step["final_text"] == "She knows me."
    assert two_step["input"] == "audio" and two_step["asr_confidence"] == 0.9
    assert client.post("/api/transcribe", files=wav, data={"language": "auto"}).status_code == 400


def test_punjabi_uses_rules_only():
    from app.pipeline import LanguagePipeline
    from app.languages import get_language
    from tests.conftest import FakeLLM
    llm = FakeLLM("anything")
    out = LanguagePipeline(get_language("pa-IN"), llm).correct("ਮੈਂ ਕੱਲ੍ਹ ਬਾਜ਼ਾਰ ਗਿਆ ਹੈ।", 50)
    assert out["stage2"]["status"] == "skipped" and llm.calls == []
    assert out["final_text"] == "ਮੈਂ ਕੱਲ੍ਹ ਬਾਜ਼ਾਰ ਗਿਆ ਸੀ।"


def test_recordings_are_saved_only_when_switched_on(client_with, tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path)
    client = client_with(recognizer=FakeRecognizer({"hi-IN": ("मैं घर जाता हूँ", None)}))
    wav = {"file": ("a.wav", tone_wav(1.0), "audio/wav")}
    monkeypatch.setattr(config, "SAVE_RECORDINGS", False)
    client.post("/api/transcribe", files=wav, data={"language": "hi-IN"})
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(config, "SAVE_RECORDINGS", True)
    client.post("/api/transcribe", files=wav, data={"language": "hi-IN"})
    saved = sorted(p.suffix for p in tmp_path.iterdir())
    assert saved == [".json", ".wav"]


def test_hindi_and_marathi_use_rules_only():
    from app.pipeline import LanguagePipeline
    from app.languages import get_language
    from tests.conftest import FakeLLM
    for code, heard, fixed in [("hi-IN", "मेरे पापा रोज सुबह पार्क में टहलती हैं।", "मेरे पापा रोज सुबह पार्क में टहलते हैं।"),
                               ("mr-IN", "माझा भाऊ काल शाळेत गेली।", "माझा भाऊ काल शाळेत गेला।")]:
        # an LLM that would undo the fix must not be asked
        out = LanguagePipeline(get_language(code), FakeLLM(heard)).correct(heard, 50)
        assert out["final_text"] == fixed
