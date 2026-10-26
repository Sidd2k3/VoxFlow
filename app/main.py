"""FastAPI app: the JSON API plus the website (served from app/static, same origin, no CORS needed)."""

import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import asr, config
from .asr import ASRServiceError
from .audio import AudioError
from .languages import LANGUAGES
from .pipeline import build_router
from .tts import TTSError, synthesize

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the local speech models in the background, so the first recording doesn't wait for them
    def preload():
        _try(app.state.router.llm.warm_up)
        recognizer = app.state.router.recognizer
        if hasattr(recognizer, "preload"):
            _try(recognizer.preload)
    threading.Thread(target=preload, daemon=True).start()
    yield


def _try(fn):
    try:
        fn()
    except Exception:
        pass   # reported by /api/status and on first use


app = FastAPI(title="VoxFlow", description="Multilingual speech recognition with two-stage correction.",
              lifespan=lifespan)
app.state.router = build_router()


def _check_language(code: str) -> str:
    if code in LANGUAGES:
        return code
    raise HTTPException(400, f"Unsupported language '{code}'. Use one of: {', '.join(LANGUAGES)}.")


class TextRequest(BaseModel):
    text: str
    language: str = "en-US"
    intensity: int = Field(50, ge=0, le=100)
    heard: dict | None = None     # what /api/transcribe returned, when the text came from the microphone


class TTSRequest(BaseModel):
    text: str
    language: str = "en-US"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/languages")
def languages():
    return [{"code": l.code, "name": l.name, "script": l.script, "voice": l.voice} for l in LANGUAGES.values()]


def _asr_description() -> str:
    recognizer = app.state.router.recognizer
    describe = getattr(recognizer, "describe", None)
    if describe is None:
        return "custom recogniser"
    if isinstance(recognizer, asr.RoutedRecognizer):
        return f"{describe('en-US')} for English, {describe('hi-IN').split(' (')[0]} for Indian languages"
    return describe("en-US")


def _asr_problems() -> list[str]:
    """Why a local engine fell back to its backup (e.g. IndicConformer not downloaded)."""
    engines = getattr(app.state.router.recognizer, "_engines", {}).values()
    return [e.failed for e in engines if isinstance(e, asr.FallbackRecognizer) and e.failed]


@app.get("/api/status")
def status():
    """Real component status for the dashboard (the original showed hardcoded text)."""
    return {
        "asr": {"engine": _asr_description(), "problems": _asr_problems()},
        "stage1": {"engine": "rule-based", "languagetool": os.getenv("ENABLE_LANGUAGETOOL", "") in {"1", "true", "yes"}},
        "stage2": app.state.router.llm.status(),
        "tts": {"engine": "edge-tts", "note": "Microsoft neural voices, needs internet"},
    }


@app.get("/api/metrics")
def metrics():
    """Benchmark results measured by eval/run_eval.py, if it has been run."""
    if not config.RESULTS_FILE.exists():
        return {"available": False}
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        return {"available": True, **json.load(f)}


@app.post("/api/text")
def correct_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > config.MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text is longer than {config.MAX_TEXT_CHARS} characters.")
    _check_language(req.language)
    return app.state.router.process_text(req.text, req.language, req.intensity, req.heard)


async def _read_audio(file: UploadFile) -> bytes:
    data = await file.read(config.MAX_AUDIO_BYTES + 1)
    if len(data) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB.")
    return data


def _run_audio(fn):
    try:
        result = fn()
    except AudioError as exc:
        raise HTTPException(400, str(exc))
    except ASRServiceError as exc:
        raise HTTPException(502, str(exc))
    if not result["recognized_text"]:
        raise HTTPException(422, "No speech was recognised, even after noise reduction. Try again closer to the mic.")
    return result


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("en-US")):
    """Speech -> text only. The website shows this at once, then sends it to /api/text for correction."""
    _check_language(language)
    data = await _read_audio(file)
    result = _run_audio(lambda: app.state.router.transcribe_audio(data, language))
    if config.SAVE_RECORDINGS:
        _save_recording(data, result)
    return result


def _save_recording(data: bytes, result: dict) -> None:
    """Keep the recording and what was heard, so engines can be compared on real live speech."""
    import time
    config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{result['language']}"
    (config.LIVE_DIR / f"{name}.wav").write_bytes(data)
    (config.LIVE_DIR / f"{name}.json").write_text(json.dumps(
        {"language": result["language"], "heard": result["recognized_text"], "engine": result["asr_engine"]},
        ensure_ascii=False), encoding="utf-8")


@app.post("/api/audio")
async def correct_audio(file: UploadFile = File(...), language: str = Form("en-US"),
                        intensity: int = Form(50)):
    """Speech -> text -> correction in one call."""
    _check_language(language)
    if not 0 <= intensity <= 100:
        raise HTTPException(400, "Intensity must be between 0 and 100.")
    data = await _read_audio(file)
    return _run_audio(lambda: app.state.router.process_audio(data, language, intensity))


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > config.MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text is longer than {config.MAX_TEXT_CHARS} characters.")
    _check_language(req.language)
    try:
        audio, voice = await synthesize(req.text, req.language)
    except TTSError as exc:
        raise HTTPException(502, str(exc))
    return Response(content=audio, media_type="audio/mpeg", headers={"X-Voice": voice})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
