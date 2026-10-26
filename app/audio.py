"""Audio handling: read WAV, convert to 16 kHz mono, and clean it up for a second ASR try.

Cleanup = noise reduction (noisereduce, spectral gating) + voice activity detection
(webrtcvad), which drops silent/noise-only frames so the recogniser only hears speech.
"""

import io
import struct
import wave

import numpy as np

TARGET_RATE = 16000          # Google ASR and webrtcvad both work well at 16 kHz
MAX_SECONDS = 60             # the free Google Web Speech endpoint handles about a minute


class AudioError(ValueError):
    """Raised for audio the app cannot use (wrong format, empty, too long)."""


PCM, IEEE_FLOAT, EXTENSIBLE = 1, 3, 0xFFFE


def _parse_wav(data: bytes) -> tuple[int, int, int, int, bytes]:
    """Read the RIFF chunks of a WAV file: (format, channels, rate, bits per sample, sample bytes).

    Python's `wave` module only reads integer PCM; this also handles 32-bit float WAV
    (used by many datasets, e.g. FLEURS) and the WAVE_FORMAT_EXTENSIBLE header.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("Not a WAV file. Record in the app or upload a .wav file.")
    fmt = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            if len(body) < 16:
                raise AudioError("Broken WAV header.")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == EXTENSIBLE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]   # real format is in the sub-format GUID
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise AudioError("Broken WAV file: audio data before the format header.")
            return (*fmt, body)
        pos += 8 + size + (size & 1)                          # chunks are padded to even sizes
    raise AudioError("Broken WAV file: no audio data found.")


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes (8/16/24/32-bit PCM or 32/64-bit float) into mono float32 samples in [-1, 1]."""
    tag, channels, rate, bits, frames = _parse_wav(data)
    if channels < 1 or rate <= 0:
        raise AudioError("Broken WAV header.")
    if tag == PCM and bits == 8:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif tag == PCM and bits == 16:
        samples = np.frombuffer(frames[: len(frames) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif tag == PCM and bits == 24:
        raw = np.frombuffer(frames[: len(frames) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (np.where(ints >= 1 << 23, ints - (1 << 24), ints) / float(1 << 23)).astype(np.float32)
    elif tag == PCM and bits == 32:
        samples = np.frombuffer(frames[: len(frames) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == IEEE_FLOAT and bits in (32, 64):
        dtype = "<f4" if bits == 32 else "<f8"
        size = bits // 8
        samples = np.clip(np.frombuffer(frames[: len(frames) // size * size], dtype=dtype), -1, 1).astype(np.float32)
    else:
        raise AudioError(f"Unsupported WAV encoding (format {tag}, {bits}-bit).")
    samples = samples[: samples.size // channels * channels]
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        raise AudioError("The recording is empty.")
    if samples.size / rate > MAX_SECONDS:
        raise AudioError(f"Recording is longer than {MAX_SECONDS} seconds.")
    return samples, rate


def resample(samples: np.ndarray, rate: int, target: int = TARGET_RATE) -> np.ndarray:
    """Linear-interpolation resampling (enough for speech recognition)."""
    if rate == target:
        return samples
    duration = samples.size / rate
    n_out = int(round(duration * target))
    t_in = np.arange(samples.size) / rate
    t_out = np.arange(n_out) / target
    return np.interp(t_out, t_in, samples).astype(np.float32)


def to_wav_bytes(samples: np.ndarray, rate: int = TARGET_RATE) -> bytes:
    """Encode float samples as 16-bit mono WAV bytes."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def prepare(data: bytes) -> tuple[np.ndarray, float]:
    """Read any 16-bit WAV and return 16 kHz mono samples plus duration in seconds."""
    samples, rate = read_wav(data)
    samples = resample(samples, rate)
    return samples, samples.size / TARGET_RATE


def reduce_noise(samples: np.ndarray) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=samples, sr=TARGET_RATE).astype(np.float32)


def voiced_only(samples: np.ndarray, aggressiveness: int = 2, frame_ms: int = 30,
                padding_frames: int = 3) -> np.ndarray:
    """Keep only frames webrtcvad classifies as speech, plus a little padding around them."""
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = TARGET_RATE * frame_ms // 1000
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    n_frames = pcm.size // frame_len
    if n_frames == 0:
        return samples
    is_speech = np.array([vad.is_speech(pcm[i * frame_len:(i + 1) * frame_len].tobytes(), TARGET_RATE)
                          for i in range(n_frames)])
    if not is_speech.any():
        return np.zeros(0, dtype=np.float32)
    keep = is_speech.copy()
    for shift in range(1, padding_frames + 1):      # pad so word onsets/endings aren't clipped
        keep[shift:] |= is_speech[:-shift]
        keep[:-shift] |= is_speech[shift:]
    chunks = [samples[i * frame_len:(i + 1) * frame_len] for i in range(n_frames) if keep[i]]
    return np.concatenate(chunks).astype(np.float32)


def clean(samples: np.ndarray) -> np.ndarray:
    """Noise reduction followed by VAD; returns the input unchanged if nothing usable is left."""
    reduced = reduce_noise(samples)
    voiced = voiced_only(reduced)
    return voiced if voiced.size >= TARGET_RATE // 4 else reduced
