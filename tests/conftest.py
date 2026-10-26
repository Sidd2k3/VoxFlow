import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import to_wav_bytes


class FakeLLM:
    """Stands in for Ollama: returns a fixed answer (or raises) without any network."""

    def __init__(self, answer=None, error=None, model="fake-llm"):
        self.answer, self.error, self.model, self.calls = answer, error, model, []

    def chat(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.answer if self.answer is not None else messages[-1]["content"].removeprefix("Transcript: ")

    def status(self):
        return {"online": True, "model": self.model, "model_available": True}


class FakeRecognizer:
    """Stands in for Google ASR: maps language code -> (text, confidence[, alternatives])."""

    def __init__(self, answers, silent_first_pass=False):
        self.answers, self.silent_first_pass, self.calls = answers, silent_first_pass, 0

    def recognize(self, samples, lang_code):
        self.calls += 1
        if self.silent_first_pass and self.calls == 1:
            return "", None, []
        answer = self.answers.get(lang_code, ("", None))
        return answer if len(answer) == 3 else (*answer, [])


def tone_wav(seconds=1.0, rate=44100):
    t = np.arange(int(seconds * rate)) / rate
    return to_wav_bytes((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


@pytest.fixture
def client_with():
    """Build a TestClient whose router uses the given fake LLM / recognizer."""
    from app.main import app
    from app.pipeline import build_router

    original = app.state.router

    def make(llm=None, recognizer=None):
        app.state.router = build_router(llm or FakeLLM(), recognizer)
        return TestClient(app)

    yield make
    app.state.router = original
