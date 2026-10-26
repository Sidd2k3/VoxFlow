"""One shared, lazily loaded copy of each Whisper model (used for language ID and for ASR)."""

import threading

_models: dict[str, object] = {}
_lock = threading.Lock()


class WhisperLoadError(RuntimeError):
    pass


def get(size: str):
    """Return the faster-whisper model of this size, loading (and downloading) it on first use."""
    with _lock:
        if size not in _models:
            try:
                from faster_whisper import WhisperModel
                _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
            except Exception as exc:
                raise WhisperLoadError(f"Could not load Whisper model '{size}': {exc}")
        return _models[size]


def is_loaded(size: str) -> bool:
    return size in _models
