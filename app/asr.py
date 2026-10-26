"""Speech recognition, in the language the user picked.

- English: Whisper large-v3, local (Apple GPU via MLX; faster-whisper on the CPU elsewhere).
- Hindi, Marathi, Bengali, Punjabi: AI4Bharat IndicConformer, local (trained on Indian speech),
  with Google's free Web Speech API (the engine behind Chrome's mic) as a fallback.
See README for the measured error rates and times.

Fixes over the original:
- the recogniser's own confidence score is returned instead of a hardcoded number;
- a network/quota failure is reported as an error, not silently treated as "no speech".
"""

import io
import time
from dataclasses import dataclass, field

import numpy as np

from .audio import TARGET_RATE, clean, resample
from .languages import get_language


class ASRServiceError(RuntimeError):
    """The speech service could not be reached or refused the request."""


@dataclass
class ASRResult:
    text: str
    confidence: float | None          # Google's confidence for the top transcript (0-1), if given
    language: str
    preprocessed: bool = False        # True if the noise-reduction + VAD retry produced this
    alternatives: list[str] = field(default_factory=list)  # Google's other hypotheses (n-best)
    uncertain: list[str] = field(default_factory=list)     # words Whisper was unsure about (low probability)
    engine: str = ""                  # which recogniser produced the text


def encode_flac(samples: np.ndarray, rate: int = TARGET_RATE) -> bytes:
    """16-bit mono FLAC bytes, encoded in-process with libsndfile (the `soundfile` package).

    Google's endpoint wants FLAC. SpeechRecognition normally shells out to a `flac` program it
    bundles, but its macOS copy is Intel-only and fails on Apple Silicon ("Bad CPU type").
    """
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(samples, -1.0, 1.0), rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _flac_audio_data(samples: np.ndarray):
    """A SpeechRecognition AudioData whose FLAC conversion uses encode_flac instead of the binary."""
    import speech_recognition as sr

    class FlacAudioData(sr.AudioData):
        def get_flac_data(self, convert_rate=None, convert_width=None):
            if convert_rate and convert_rate != TARGET_RATE:
                return encode_flac(resample(samples, TARGET_RATE, convert_rate), convert_rate)
            return encode_flac(samples)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return FlacAudioData(pcm, TARGET_RATE, 2)


class GoogleRecognizer:
    """Thin wrapper so the rest of the app (and the tests) don't depend on the library directly.

    Network hiccups (Wi-Fi drop, timeout, reset connection) are retried with a short back-off;
    if every attempt fails, the error becomes ASRServiceError (HTTP 502), never a crash.
    """

    def __init__(self, attempts: int = 3, backoff_seconds: float = 2.0, locale: str | None = None):
        # locale overrides the language code sent to Google, e.g. "en-IN" for Indian English
        self.attempts, self.backoff_seconds, self.locale = attempts, backoff_seconds, locale

    def describe(self, lang_code: str) -> str:
        return f"Google ({self.locale or lang_code})"

    def _call(self, audio, lang_code: str):
        import speech_recognition as sr
        return sr.Recognizer().recognize_google(audio, language=self.locale or lang_code, show_all=True)

    def recognize(self, samples: np.ndarray, lang_code: str) -> tuple[str, float | None, list[str]]:
        """Return (best transcript, its confidence or None, the other hypotheses)."""
        import speech_recognition as sr
        audio = _flac_audio_data(samples)
        last_error = None
        for attempt in range(self.attempts):
            try:
                response = self._call(audio, lang_code)
                break
            except sr.UnknownValueError:
                return "", None, []     # Google heard no speech it could transcribe in this language
            except (sr.RequestError, OSError, TimeoutError) as exc:   # URLError is an OSError
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(self.backoff_seconds * (attempt + 1))
        else:
            raise ASRServiceError(f"Google speech service unreachable after {self.attempts} tries: {last_error}")
        if not response or not response.get("alternative"):
            return "", None, []
        hypotheses = response["alternative"]
        best = hypotheses[0]
        others = [h.get("transcript", "").strip() for h in hypotheses[1:] if h.get("transcript")]
        return best.get("transcript", "").strip(), best.get("confidence"), others


# Whisper continues the style of its prompt. An example with everyday grammar mistakes and filler words
# makes it write what the speaker actually said ("like she knowed me") instead of a tidied-up guess
# ("she wasn't me"), so the correction stages can fix the real mistake.
VERBATIM_PROMPT = "Umm, so like, I goed there yesterday and she don't know nothing, like, I seen it myself."
WHISPER_CODE = {"en-US": "en", "hi-IN": "hi", "mr-IN": "mr", "bn-IN": "bn", "pa-IN": "pa"}
UNCERTAIN_BELOW = 0.5   # a word whose probability is below this may be misheard


def _from_words(text: str, words: list[tuple[str, float]]):
    """(transcript, mean word probability, [], words with probability < UNCERTAIN_BELOW)."""
    if not text:
        return "", None, [], []
    if not words:
        return text, None, [], []
    confidence = sum(p for _, p in words) / len(words)
    uncertain = [w.strip(" .,!?;:\"").strip() for w, p in words if p < UNCERTAIN_BELOW]
    return text, round(confidence, 4), [], [u for u in uncertain if u]


class WhisperRecognizer:
    """Local speech recognition with faster-whisper. Same interface as GoogleRecognizer."""

    WHISPER_CODE = WHISPER_CODE

    VERBATIM_PROMPT = VERBATIM_PROMPT

    def __init__(self, model_size: str | None = None, verbatim: bool | None = None):
        from . import config
        self.model_size = model_size or config.ASR_MODEL
        self.verbatim = config.WHISPER_VERBATIM if verbatim is None else verbatim

    def describe(self, lang_code: str) -> str:
        return f"Whisper {self.model_size}" + (" (verbatim)" if self.verbatim and lang_code == "en-US" else "")

    def preload(self):
        from . import whisper_models
        whisper_models.get(self.model_size)

    UNCERTAIN_BELOW = UNCERTAIN_BELOW

    def recognize(self, samples: np.ndarray, lang_code: str):
        """Return (transcript, confidence, alternatives, uncertain words).

        confidence = mean word probability; uncertain = words Whisper gave probability < 0.5.
        """
        from . import whisper_models
        try:
            model = whisper_models.get(self.model_size)
        except whisper_models.WhisperLoadError as exc:
            raise ASRServiceError(str(exc))
        prompt = self.VERBATIM_PROMPT if self.verbatim and lang_code == "en-US" else None
        segments, _ = model.transcribe(samples.astype(np.float32), language=self.WHISPER_CODE[lang_code],
                                       beam_size=5, condition_on_previous_text=False, vad_filter=False,
                                       word_timestamps=True, initial_prompt=prompt)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        return _from_words(text, [(w.word, w.probability) for s in segments for w in (s.words or [])])


class MlxWhisperRecognizer:
    """Whisper on the Mac's GPU (Apple Silicon) with Apple's MLX framework. Same interface as above.

    The full large-v3 model (32 decoder layers) is more accurate than large-v3-turbo (4 decoder layers)
    but too slow on the CPU; on the M-series GPU it runs at a usable speed. MLX decodes greedily
    (no beam search), with the standard temperature fallback when the output looks wrong.
    """

    REPOS = {"large-v3": "mlx-community/whisper-large-v3-mlx",
             "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
             "medium": "mlx-community/whisper-medium-mlx",
             "small": "mlx-community/whisper-small-mlx"}
    _lock = None

    def __init__(self, model_size: str | None = None, verbatim: bool | None = None, word_times: bool = True):
        import threading
        from . import config
        self.model_size = model_size or "large-v3"
        self.repo = self.REPOS.get(self.model_size, self.model_size)
        self.verbatim = config.WHISPER_VERBATIM if verbatim is None else verbatim
        # per-word timing gives per-word probabilities ("unsure" words) but costs time; without it the
        # confidence comes from the whole segment and no word is marked unsure
        self.word_times = word_times
        if MlxWhisperRecognizer._lock is None:
            MlxWhisperRecognizer._lock = threading.Lock()   # one GPU job at a time

    def describe(self, lang_code: str) -> str:
        return f"Whisper {self.model_size} on GPU" + (" (verbatim)" if self.verbatim and lang_code == "en-US" else "") \
            + ("" if self.word_times else " (fast)")

    @staticmethod
    def available() -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _transcribe(self, samples: np.ndarray, lang_code: str) -> dict:
        try:
            import mlx_whisper
        except Exception as exc:
            raise ASRServiceError(f"mlx-whisper is not installed (it needs a Mac with Apple Silicon): {exc}")
        prompt = VERBATIM_PROMPT if self.verbatim and lang_code == "en-US" else None
        with self._lock:
            try:
                return mlx_whisper.transcribe(samples.astype(np.float32), path_or_hf_repo=self.repo,
                                              language=WHISPER_CODE[lang_code], word_timestamps=self.word_times,
                                              condition_on_previous_text=False, initial_prompt=prompt,
                                              verbose=None)
            except ASRServiceError:
                raise
            except Exception as exc:
                raise ASRServiceError(f"Whisper on GPU failed ({self.repo}): {exc}")

    def preload(self):
        self._transcribe(np.zeros(TARGET_RATE, dtype=np.float32), "en-US")   # downloads + loads the model

    def recognize(self, samples: np.ndarray, lang_code: str):
        out = self._transcribe(samples, lang_code)
        segments = out.get("segments") or []
        text = " ".join(s.get("text", "").strip() for s in segments).strip() or out.get("text", "").strip()
        if not self.word_times:
            import math
            probs = [math.exp(s["avg_logprob"]) for s in segments if s.get("avg_logprob") is not None]
            return text, (round(sum(probs) / len(probs), 4) if probs and text else None), [], []
        return _from_words(text, [(w["word"], w["probability"]) for s in segments for w in s.get("words") or []])


class ParakeetRecognizer:
    """NVIDIA Parakeet TDT 0.6B (English) on the Apple Silicon GPU via parakeet-mlx. A transducer like
    IndicConformer: it reads the audio once instead of writing the text word by word, so it is fast."""

    REPOS = {"v2": "mlx-community/parakeet-tdt-0.6b-v2", "v3": "mlx-community/parakeet-tdt-0.6b-v3"}
    _models: dict = {}
    _lock = None

    def __init__(self, version: str | None = None):
        import threading
        self.version = version or "v2"
        self.repo = self.REPOS.get(self.version, self.version)
        if ParakeetRecognizer._lock is None:
            ParakeetRecognizer._lock = threading.Lock()

    def describe(self, lang_code: str) -> str:
        return f"Parakeet TDT 0.6B {self.version} on GPU"

    @staticmethod
    def available() -> bool:
        try:
            import parakeet_mlx  # noqa: F401
            return True
        except Exception:
            return False

    def _model(self):
        if self.repo not in self._models:
            try:
                from parakeet_mlx import from_pretrained
                self._models[self.repo] = from_pretrained(self.repo)
            except Exception as exc:
                raise ASRServiceError(f"Parakeet could not be loaded ({self.repo}): {exc}. "
                                      f"It needs an Apple Silicon Mac and: pip install parakeet-mlx")
        return self._models[self.repo]

    def preload(self):
        self.recognize(np.zeros(TARGET_RATE, dtype=np.float32), "en-US")

    def recognize(self, samples: np.ndarray, lang_code: str):
        if lang_code != "en-US":
            raise ASRServiceError(f"Parakeet is used for English only, not {lang_code}")
        with self._lock:
            model = self._model()
            try:
                import mlx.core as mx
                from parakeet_mlx.audio import get_logmel
                # float32, like parakeet-mlx's own loader: its spectrogram (STFT) breaks on bfloat16 input
                audio = mx.array(np.ascontiguousarray(samples, dtype=np.float32))
                result = model.generate(get_logmel(audio, model.preprocessor_config))[0]
            except Exception as exc:
                raise ASRServiceError(f"Parakeet failed: {exc}")
        return result.text.strip(), None, [], []


INDIC_REPO = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_CODE = {"hi-IN": "hi", "mr-IN": "mr", "bn-IN": "bn", "pa-IN": "pa"}


class IndicConformerRecognizer:
    """AI4Bharat IndicConformer (600M, MIT licence), trained on Indian-language speech, run locally.
    One model for 22 Indian languages; CTC decoding (fastest) by default, RNNT optional.
    The model is gated on Hugging Face: accept its terms once and run `hf auth login`."""

    _model = None
    _lock = None

    def __init__(self, decoding: str = "ctc"):
        import threading
        if decoding not in ("ctc", "rnnt"):
            raise ValueError("IndicConformer decoding must be 'ctc' or 'rnnt'")
        self.decoding = decoding
        if IndicConformerRecognizer._lock is None:
            IndicConformerRecognizer._lock = threading.Lock()

    def describe(self, lang_code: str) -> str:
        return f"IndicConformer ({self.decoding})"

    @classmethod
    def _load(cls):
        if cls._model is None:
            try:
                from transformers import AutoModel
                cls._model = AutoModel.from_pretrained(INDIC_REPO, trust_remote_code=True)
            except Exception as exc:
                raise ASRServiceError(f"IndicConformer could not be loaded ({exc}). Accept the model's terms on "
                                      f"huggingface.co/{INDIC_REPO} and run: hf auth login")
        return cls._model

    def preload(self):
        self.recognize(np.zeros(TARGET_RATE, dtype=np.float32), "hi-IN")

    def recognize(self, samples: np.ndarray, lang_code: str):
        if lang_code not in INDIC_CODE:
            raise ASRServiceError(f"IndicConformer does not support {lang_code}")
        import torch
        with self._lock:
            model = self._load()
            try:
                wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
                with torch.inference_mode():
                    out = model(wav, INDIC_CODE[lang_code], self.decoding)
            except Exception as exc:
                raise ASRServiceError(f"IndicConformer failed: {exc}")
        text = (out[0] if isinstance(out, (list, tuple)) else str(out)).strip()
        return text, None, [], []


TRANSCRIBE_REPO = "bodhan-ai/indic-transcribe-core"


class IndicTranscribeRecognizer:
    """Indic-Transcribe-core (1.2B, NVIDIA Canary architecture, Indic Open Model License), the successor that
    AI4Bharat's IndicConformer page recommends. Gated on Hugging Face like IndicConformer; runs locally."""

    _model = None
    _lock = None

    def __init__(self):
        import threading
        if IndicTranscribeRecognizer._lock is None:
            IndicTranscribeRecognizer._lock = threading.Lock()

    def describe(self, lang_code: str) -> str:
        return "Indic-Transcribe-core"

    @classmethod
    def _load(cls):
        if cls._model is None:
            try:
                import sys
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(TRANSCRIBE_REPO)
                if model_dir not in sys.path:
                    sys.path.insert(0, model_dir)      # the model code ships inside the download
                from indic_transcribe import IndicTranscribe
                cls._model = IndicTranscribe.from_pretrained(model_dir)
            except Exception as exc:
                raise ASRServiceError(f"Indic-Transcribe could not be loaded ({exc}). Accept the model's terms on "
                                      f"huggingface.co/{TRANSCRIBE_REPO} and run: hf auth login")
        return cls._model

    def preload(self):
        self.recognize(np.zeros(TARGET_RATE, dtype=np.float32), "hi-IN")

    def recognize(self, samples: np.ndarray, lang_code: str):
        import os
        import tempfile
        import soundfile as sf
        with self._lock:
            model = self._load()
            fd, path = tempfile.mkstemp(suffix=".wav")     # the model takes a file path
            os.close(fd)
            try:
                sf.write(path, np.clip(samples, -1.0, 1.0), TARGET_RATE, subtype="PCM_16")
                out = model(path, lang=lang_code[:2])
            except Exception as exc:
                raise ASRServiceError(f"Indic-Transcribe failed: {exc}")
            finally:
                os.remove(path)
        if isinstance(out, dict):
            out = out.get("text", "")
        elif isinstance(out, (list, tuple)):
            out = out[0] if out else ""
            out = out.get("text", "") if isinstance(out, dict) else out
        return str(out).strip(), None, [], []


class FallbackRecognizer:
    """Try the first engine; if it fails (not installed, model not downloaded), use the second."""

    def __init__(self, first, second):
        self.first, self.second, self.failed = first, second, None

    def describe(self, lang_code: str) -> str:
        if self.failed:   # the reason is in self.failed (shown by /api/status)
            return f"{getattr(self.second, 'describe', lambda c: 'backup')(lang_code)} (fallback)"
        return self.first.describe(lang_code)

    def preload(self):
        try:
            if hasattr(self.first, "preload"):
                self.first.preload()
        except ASRServiceError as exc:
            self.failed = str(exc)

    def recognize(self, samples: np.ndarray, lang_code: str):
        if not self.failed:
            try:
                return self.first.recognize(samples, lang_code)
            except ASRServiceError as exc:
                self.failed = str(exc)
        return self.second.recognize(samples, lang_code)


def make_engine(spec: str):
    """Build a recogniser from a spec: "google", "google:<locale>" (e.g. google:en-IN),
    "whisper[:<model size>[:verbatim]]" (faster-whisper on the CPU) or
    "mlx[:<model size>[:verbatim|:fast]]" (Whisper on the Apple Silicon GPU; fast = no per-word timing),
    "parakeet[:v2|v3]" (NVIDIA Parakeet on the Apple Silicon GPU, English),
    "indicconformer[:ctc|rnnt]" (AI4Bharat, local), "indictranscribe" (Indic-Transcribe-core, local), or "<first>|<fallback>", e.g. "indicconformer|google"."""
    if "|" in spec:
        first, _, second = spec.partition("|")
        return FallbackRecognizer(make_engine(first), make_engine(second))
    name, _, arg = spec.partition(":")
    if name == "indicconformer":
        return IndicConformerRecognizer(arg or "ctc")
    if name == "indictranscribe":
        return IndicTranscribeRecognizer()
    if name == "google":
        return GoogleRecognizer(locale=arg or None)
    if name == "whisper":
        size, _, mode = arg.partition(":")
        return WhisperRecognizer(size or None, verbatim=(mode == "verbatim") if mode else None)
    if name == "mlx":
        size, _, mode = arg.partition(":")
        if mode == "fast":
            return MlxWhisperRecognizer(size or None, word_times=False)
        return MlxWhisperRecognizer(size or None, verbatim=(mode == "verbatim") if mode else None)
    if name == "parakeet":
        return ParakeetRecognizer(arg or None)
    raise ValueError(f"Unknown speech recognition engine '{spec}'")


class RoutedRecognizer:
    """Send each language to the recogniser that measured best for it (eval/compare_asr.py, README)."""

    def __init__(self, routes: dict[str, str] | None = None):
        from . import config
        self.routes = routes or config.ASR_ROUTES
        self._engines: dict[str, object] = {}

    def engine_for(self, lang_code: str) -> str:
        return self.routes.get(lang_code, "google")

    def _engine(self, spec: str):
        if spec not in self._engines:
            self._engines[spec] = make_engine(spec)
        return self._engines[spec]

    def describe(self, lang_code: str) -> str:
        return self._engine(self.engine_for(lang_code)).describe(lang_code)

    def recognize(self, samples: np.ndarray, lang_code: str):
        return self._engine(self.engine_for(lang_code)).recognize(samples, lang_code)

    def preload(self):
        """Load every local model the routes use, so the first request doesn't wait for it."""
        for spec in set(self.routes.values()):
            engine = self._engine(spec)
            if hasattr(engine, "preload"):
                engine.preload()


def default_recognizer():
    from . import config
    if config.ASR_ENGINE == "google":
        return GoogleRecognizer()
    if config.ASR_ENGINE == "whisper":
        return WhisperRecognizer()
    return RoutedRecognizer()


def _recognize_one(recognizer, samples, lang_code: str) -> ASRResult:
    result = recognizer.recognize(samples, lang_code)
    text, conf, others = result[:3]
    uncertain = list(result[3]) if len(result) > 3 else []
    engine = getattr(recognizer, "describe", lambda c: "")(lang_code)
    return ASRResult(text=text, confidence=conf, language=lang_code, alternatives=others, uncertain=uncertain,
                     engine=engine)


def transcribe(samples: np.ndarray, lang_code: str, recognizer=None) -> ASRResult:
    """Transcribe 16 kHz samples in the given language. If nothing is heard, clean the audio and retry once."""
    recognizer = recognizer or default_recognizer()
    code = get_language(lang_code).code
    result = _recognize_one(recognizer, samples, code)
    if result.text:
        return result
    retry = _recognize_one(recognizer, clean(samples), code)
    retry.preprocessed = True
    return retry
