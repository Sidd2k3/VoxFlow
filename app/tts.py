"""Speak Back: text-to-speech.

1. Microsoft Edge neural voice (edge-tts) for the language.
2. If that voice isn't available (Microsoft's free endpoint has no voice for some languages,
   e.g. Punjabi) or the call fails, Google Translate's voice (gTTS) is used instead.
Failure of both is an error, never an empty 200 response (the original's bug).
"""

import asyncio
import io

import edge_tts

from .languages import get_language


class TTSError(RuntimeError):
    pass


async def _edge(text: str, voice: str) -> bytes:
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise TTSError(f"no audio from voice {voice}")
    return bytes(audio)


def _gtts(text: str, lang: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    if not buf.getvalue():
        raise TTSError(f"no audio from Google TTS ({lang})")
    return buf.getvalue()


async def synthesize(text: str, lang_code: str) -> tuple[bytes, str]:
    """Return (MP3 bytes, name of the voice used)."""
    lang = get_language(lang_code)
    errors = []
    try:
        return await _edge(text, lang.voice), lang.voice
    except Exception as exc:
        errors.append(f"{lang.voice}: {exc}")
    try:
        return await asyncio.to_thread(_gtts, text, lang.gtts), f"Google TTS ({lang.gtts})"
    except Exception as exc:
        errors.append(f"Google TTS: {exc}")
    raise TTSError("Text-to-speech failed. " + " | ".join(errors))
