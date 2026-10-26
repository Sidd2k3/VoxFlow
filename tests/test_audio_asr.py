import numpy as np
import pytest

from app import asr
from app.audio import AudioError, TARGET_RATE, prepare, resample, voiced_only
from tests.conftest import FakeRecognizer, tone_wav


def test_prepare_resamples_any_rate_to_16k_mono():
    samples, seconds = prepare(tone_wav(1.0, 44100))
    assert abs(seconds - 1.0) < 0.01 and abs(samples.size - TARGET_RATE) <= 1


def test_resample_keeps_duration():
    x = np.zeros(48000, dtype=np.float32)
    assert resample(x, 48000).size == 16000


def test_bad_audio_is_rejected():
    with pytest.raises(AudioError):
        prepare(b"not a wav")
    with pytest.raises(AudioError):
        prepare(tone_wav(61))


def test_vad_drops_silence():
    rng = np.random.default_rng(0)
    speechlike = (0.3 * np.sin(2 * np.pi * 200 * np.arange(TARGET_RATE) / TARGET_RATE)
                  * (1 + 0.5 * rng.standard_normal(TARGET_RATE))).astype(np.float32)
    silence = np.zeros(TARGET_RATE, dtype=np.float32)
    kept = voiced_only(np.concatenate([silence, speechlike, silence]))
    assert kept.size < 3 * TARGET_RATE


def test_transcribe_returns_real_confidence():
    rec = FakeRecognizer({"hi-IN": ("मैं जाता है", 0.87)})
    result = asr.transcribe(np.zeros(16000, np.float32), "hi-IN", rec)
    assert result.text == "मैं जाता है" and result.confidence == 0.87 and not result.preprocessed


def test_empty_first_pass_retries_with_cleaned_audio():
    rec = FakeRecognizer({"en-US": ("hello", 0.9)}, silent_first_pass=True)
    samples, _ = prepare(tone_wav(1.0, 16000))
    result = asr.transcribe(samples, "en-US", rec)
    assert result.text == "hello" and result.preprocessed and rec.calls == 2


def _wav(tag, bits, channels, rate, payload, extensible=False):
    import struct
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * block, block, bits)
    if extensible:  # cbSize, valid bits, channel mask, sub-format GUID (first 2 bytes = real tag)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


@pytest.mark.parametrize("tag,bits,dtype,scale,extensible", [
    (3, 32, "<f4", 1.0, False),            # 32-bit float, as used by FLEURS
    (3, 32, "<f4", 1.0, True),             # float inside a WAVE_FORMAT_EXTENSIBLE header
    (1, 32, "<i4", 2 ** 31, False),
    (1, 16, "<i2", 2 ** 15, True),
])
def test_reads_float_and_extensible_wav(tag, bits, dtype, scale, extensible):
    from app.audio import read_wav
    values = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float64)
    payload = (values * (scale if scale != 1.0 else 1)).astype(dtype).tobytes()
    samples, rate = read_wav(_wav(tag, bits, 1, 16000, payload, extensible))
    assert rate == 16000 and np.allclose(samples, values, atol=1e-3)


def test_reads_24bit_stereo_wav():
    from app.audio import read_wav
    left, right = 0.5, -0.5
    def i24(x):
        v = int(round(x * (2 ** 23 - 1))) & 0xFFFFFF
        return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])
    payload = (i24(left) + i24(right)) * 4
    samples, _ = read_wav(_wav(1, 24, 2, 8000, payload))
    assert samples.size == 4 and np.allclose(samples, 0.0, atol=1e-3)   # stereo averaged to mono


def test_flac_is_encoded_in_process_not_by_bundled_binary():
    import io
    import soundfile as sf
    from app.asr import _flac_audio_data, encode_flac
    t = np.arange(TARGET_RATE) / TARGET_RATE
    samples = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    flac = _flac_audio_data(samples).get_flac_data(convert_rate=None, convert_width=2)
    assert flac[:4] == b"fLaC" and flac == encode_flac(samples)
    decoded, rate = sf.read(io.BytesIO(flac), dtype="float32")
    assert rate == TARGET_RATE and np.allclose(decoded, samples, atol=1e-3)


def test_network_drop_is_retried_then_reported_cleanly():
    from app.asr import ASRServiceError, GoogleRecognizer

    class Flaky(GoogleRecognizer):
        def __init__(self, failures):
            super().__init__(attempts=3, backoff_seconds=0)
            self.failures, self.calls = failures, 0

        def _call(self, audio, lang_code):
            self.calls += 1
            if self.calls <= self.failures:
                raise OSError(65, "No route to host")
            return {"alternative": [{"transcript": "hello", "confidence": 0.9}]}

    samples = np.zeros(TARGET_RATE, np.float32)
    ok = Flaky(failures=2)
    assert ok.recognize(samples, "en-US") == ("hello", 0.9, []) and ok.calls == 3
    with pytest.raises(ASRServiceError):
        Flaky(failures=3).recognize(samples, "en-US")



def test_nothing_heard_returns_empty_instead_of_crashing():
    import speech_recognition as sr
    from app.asr import GoogleRecognizer

    class Silent(GoogleRecognizer):
        def _call(self, audio, lang_code):
            raise sr.UnknownValueError()     # what SpeechRecognition 3.17 raises when nothing is heard
    assert Silent().recognize(np.zeros(TARGET_RATE, np.float32), "pa-IN") == ("", None, [])


def test_routed_recognizer_sends_each_language_to_its_engine():
    from app.asr import RoutedRecognizer
    r = RoutedRecognizer({"en-US": "whisper", "hi-IN": "google"})   # explicit: the default depends on the machine
    calls = []
    r._engines = {"google": type("G", (), {"recognize": lambda s, x, c: calls.append(("google", c)) or ("g", None, [])})(),
                  "whisper": type("W", (), {"recognize": lambda s, x, c: calls.append(("whisper", c)) or ("w", 0.9, [])})()}
    samples = np.zeros(TARGET_RATE, np.float32)
    assert r.recognize(samples, "en-US")[0] == "w" and r.recognize(samples, "hi-IN")[0] == "g"
    assert calls == [("whisper", "en-US"), ("google", "hi-IN")]


def test_google_locale_override_is_sent_to_google():
    from app.asr import GoogleRecognizer, RoutedRecognizer
    sent = []

    class Spy(GoogleRecognizer):
        def _call(self, audio, lang_code):
            sent.append(self.locale or lang_code)
            return {"alternative": [{"transcript": "hello there"}]}
    assert Spy(locale="en-IN").recognize(np.zeros(TARGET_RATE, np.float32), "en-US")[0] == "hello there"
    assert sent == ["en-IN"]
    r = RoutedRecognizer({"en-US": "google:en-IN"})
    assert isinstance(r._engine("google:en-IN"), GoogleRecognizer) and r._engine("google:en-IN").locale == "en-IN"


def test_mlx_engine_on_gpu_reads_words_and_uses_verbatim_prompt(monkeypatch):
    import sys
    import types
    from app import asr
    seen = {}

    def transcribe(audio, **kw):
        seen.update(kw)
        return {"text": " She knowed me", "segments": [{"text": " She knowed me", "words": [
            {"word": " She", "probability": 0.95}, {"word": " knowed", "probability": 0.4},
            {"word": " me", "probability": 0.9}]}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    rec = asr.make_engine("mlx:large-v3:verbatim")
    assert isinstance(rec, asr.MlxWhisperRecognizer) and rec.available()
    text, conf, alts, unsure = rec.recognize(np.zeros(16000, np.float32), "en-US")
    assert text == "She knowed me" and unsure == ["knowed"] and abs(conf - 0.75) < 1e-6
    assert seen["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert seen["language"] == "en" and seen["initial_prompt"] == asr.VERBATIM_PROMPT
    rec.recognize(np.zeros(16000, np.float32), "hi-IN")
    assert seen["language"] == "hi" and seen["initial_prompt"] is None


def test_mlx_engine_missing_is_a_service_error_not_a_crash(monkeypatch):
    import sys
    from app import asr
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)    # import fails, as on Linux/Intel
    rec = asr.make_engine("mlx:large-v3")
    assert not rec.available()
    with pytest.raises(asr.ASRServiceError):
        rec.recognize(np.zeros(16000, np.float32), "en-US")


def test_routed_preload_loads_each_local_engine_once():
    from app import asr
    loaded = []

    class Local:
        def preload(self):
            loaded.append(1)

    routed = asr.RoutedRecognizer({"en-US": "x", "hi-IN": "google"})
    routed._engines = {"x": Local(), "google": object()}
    routed.preload()
    assert loaded == [1]


def test_indicconformer_runs_the_model_in_the_selected_language(monkeypatch):
    seen = {}

    class Model:
        def __call__(self, wav, lang, decoding):
            seen.update(shape=tuple(wav.shape), lang=lang, decoding=decoding)
            return " मैं घर जाता हूँ "

    monkeypatch.setattr(asr.IndicConformerRecognizer, "_model", Model())
    rec = asr.make_engine("indicconformer")
    assert rec.recognize(np.zeros(16000, np.float32), "mr-IN") == ("मैं घर जाता हूँ", None, [], [])
    assert seen == {"shape": (1, 16000), "lang": "mr", "decoding": "ctc"}
    assert asr.make_engine("indicconformer:rnnt").decoding == "rnnt"
    with pytest.raises(asr.ASRServiceError):
        rec.recognize(np.zeros(16000, np.float32), "en-US")


def test_indicconformer_not_available_falls_back_to_google(monkeypatch):
    monkeypatch.setattr(asr.IndicConformerRecognizer, "_model", None)

    def fail(cls):
        raise asr.ASRServiceError("model is gated")
    monkeypatch.setattr(asr.IndicConformerRecognizer, "_load", classmethod(fail))
    rec = asr.make_engine("indicconformer|google")
    assert isinstance(rec, asr.FallbackRecognizer)
    rec.second = FakeRecognizer({"hi-IN": ("नमस्ते", 0.9)})
    assert rec.recognize(np.zeros(16000, np.float32), "hi-IN")[0] == "नमस्ते"
    assert "fallback" in rec.describe("hi-IN")


def test_indian_languages_route_to_indicconformer_by_default():
    from app import config
    assert config.ASR_ROUTES["hi-IN"] == config.ASR_ROUTES["mr-IN"] == "indicconformer:rnnt|google"
    assert config.ASR_ROUTES["pa-IN"] == "indicconformer:ctc|google"
    assert config.ASR_ROUTES["bn-IN"] == "google"       # live test: Google heard short Bengali sentences better
    assert asr.make_engine(config.ASR_ROUTES["mr-IN"]).first.decoding == "rnnt"


def test_transcribe_uses_only_the_selected_language():
    calls = []
    rec = FakeRecognizer({"pa-IN": ("ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ", None), "hi-IN": ("मैं", None)})
    orig = rec.recognize
    rec.recognize = lambda s, c: calls.append(c) or orig(s, c)
    result = asr.transcribe(np.zeros(16000, np.float32), "pa-IN", rec)
    assert result.language == "pa-IN" and result.text == "ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ" and calls == ["pa-IN"]
    with pytest.raises(ValueError):
        asr.transcribe(np.zeros(16000, np.float32), "auto", rec)


def test_parakeet_engine_reads_text_from_the_model(monkeypatch):
    import sys
    import types

    class Result:
        text = " She knows me "

    class Model:
        preprocessor_config = object()

        def generate(self, mel):
            return [Result()]

    fake_mx = types.SimpleNamespace(array=lambda a: a)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.SimpleNamespace(from_pretrained=lambda r: Model()))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", types.SimpleNamespace(get_logmel=lambda a, c: a))
    rec = asr.make_engine("parakeet:v2")
    monkeypatch.setattr(asr.ParakeetRecognizer, "_models", {})
    assert rec.recognize(np.zeros(16000, np.float32), "en-US") == ("She knows me", None, [], [])
    with pytest.raises(asr.ASRServiceError):
        rec.recognize(np.zeros(16000, np.float32), "hi-IN")


def test_mlx_fast_mode_skips_word_timing(monkeypatch):
    import sys
    import types
    seen = {}

    def transcribe(audio, **kw):
        seen.update(kw)
        return {"text": " She knows me", "segments": [{"text": " She knows me", "avg_logprob": 0.0}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    rec = asr.make_engine("mlx:large-v3-turbo:fast")
    assert rec.recognize(np.zeros(16000, np.float32), "en-US") == ("She knows me", 1.0, [], [])
    assert seen["word_timestamps"] is False and "fast" in rec.describe("en-US")


def test_indictranscribe_engine_passes_the_language_and_reads_the_text(monkeypatch):
    seen = {}

    class Model:
        def __call__(self, path, lang):
            import soundfile as sf
            seen.update(lang=lang, seconds=len(sf.read(path)[0]) / 16000)
            return " আমি বই পড়ি "

    monkeypatch.setattr(asr.IndicTranscribeRecognizer, "_model", Model())
    rec = asr.make_engine("indictranscribe")
    assert rec.recognize(np.zeros(16000, np.float32), "bn-IN") == ("আমি বই পড়ি", None, [], [])
    assert seen == {"lang": "bn", "seconds": 1.0}
