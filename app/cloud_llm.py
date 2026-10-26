"""Large free cloud LLMs for the grammar step (Stage 2), with automatic fallback.

    Gemini (Google AI Studio free tier)  ->  Groq (free tier, Llama 3.3 70B / gpt-oss)  ->  local Ollama

Why: the local 3B model (llama3.2) made Hindi and Marathi worse and is weak in Bengali and Punjabi
(README, Results 5, 7, 11). Large models know the grammar of all five languages and can fix several
mistakes across 2-3 sentences at once, and repair a misheard word from context.

Only the recognised TEXT is sent, never the audio. Keys are read from .env (GEMINI_API_KEY, GROQ_API_KEY);
without keys the app works as before (rules, plus the local LLM for English).
Each client has the same interface as stage2.OllamaClient: chat(messages, lang_code), status(),
warm_up(), supports(lang_code), model.
"""

import json
import re
import time

import httpx

from . import config

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
GROQ_URL = "https://api.groq.com/openai/v1"

# models with these words in the name are not chat models (speech, images, embeddings...)
_NOT_CHAT = re.compile(r"tts|live|transcribe|image|embedding|vision|audio|native|robotics|computer|guard|whisper|"
                       r"aqa|learnlm|gemma|nano|banana|veo|imagen", re.I)


class CloudLLMError(Exception):
    pass


class RateLimited(CloudLLMError):
    pass


def _parse_json_answer(text: str) -> str:
    """{"corrected": "..."} -> the text. Tolerates ```json fences and text around the object."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise CloudLLMError(f"no JSON in the answer: {text[:120]!r}")
    return str(json.loads(match.group(0)).get("corrected", "")).strip()


def _version(name: str) -> tuple:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def pick_gemini_model(names: list[str]) -> str | None:
    """Newest stable Flash-Lite (the free tier with the biggest daily quota), else newest stable Flash."""
    names = [n.removeprefix("models/") for n in names if "gemini" in n and not _NOT_CHAT.search(n)]
    for kind in ("flash-lite", "flash"):
        pool = [n for n in names if re.search(rf"-{kind}($|-\d{{3}}$|-latest$|-preview)", n)
                and (kind == "flash-lite" or "lite" not in n)]
        stable = [n for n in pool if "preview" not in n and "exp" not in n] or pool
        if stable:
            return max(stable, key=lambda n: (_version(n), "latest" not in n, -len(n)))
    return None


GROQ_PREFERENCE = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct",
                   "qwen/qwen3-32b", "meta-llama/llama-4-maverick-17b-128e-instruct"]


def pick_groq_model(names: list[str]) -> str | None:
    for wanted in GROQ_PREFERENCE:
        if wanted in names:
            return wanted
    big = [n for n in names if re.search(r"70b|120b|maverick|kimi|qwen3", n) and not _NOT_CHAT.search(n)]
    return big[0] if big else None


class _CloudClient:
    name = "cloud"
    strong = True          # big model: its answers get the light check, not the strict edit filter

    def __init__(self, key: str, model: str | None, timeout: float):
        self.key, self._model, self.timeout = key, model, timeout
        self.error = ""
        self.blocked_until = 0.0   # after a rate limit, skip this client for a while

    @property
    def model(self) -> str:
        return self._model or f"{self.name} (model not chosen yet)"

    def supports(self, lang_code: str) -> bool:
        return bool(self.key) and time.time() >= self.blocked_until

    def warm_up(self) -> None:
        self._ensure_model()

    def status(self) -> dict:
        try:
            self._ensure_model()
            return {"online": True, "model": self.model, "model_available": True}
        except Exception as exc:
            return {"online": False, "model": self.model, "model_available": False, "error": str(exc)}

    def _ensure_model(self) -> None:
        if not self._model:
            self._model = self._choose_model()
            if not self._model:
                raise CloudLLMError(f"no suitable {self.name} model found for this key")

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", "20") or 20)
            self.blocked_until = time.time() + min(wait, 300)
            raise RateLimited(f"{self.name} rate limit reached (free tier)")
        if r.status_code >= 400:
            raise CloudLLMError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")


class GeminiClient(_CloudClient):
    name = "Gemini"

    # request settings, from the most to the least specific; models differ in what they accept (newer ones
    # reject a thinking budget of 0, some reject a response schema), so on "400 invalid argument" the next
    # one is tried and the first that works is remembered
    VARIANTS = [{"thinking": True, "schema": True}, {"thinking": False, "schema": True},
                {"thinking": False, "schema": False, "mime": True}, {"thinking": False, "schema": False, "mime": False}]

    def __init__(self, key: str = config.GEMINI_API_KEY, model: str | None = config.GEMINI_MODEL or None,
                 timeout: float = 30):
        super().__init__(key, model, timeout)
        self._variant = 0

    def _choose_model(self) -> str | None:
        r = httpx.get(f"{GEMINI_URL}/models", params={"pageSize": 1000},
                      headers={"x-goog-api-key": self.key}, timeout=self.timeout)
        self._check(r)
        models = [m["name"] for m in r.json().get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])]
        return pick_gemini_model(models)

    def _body(self, messages: list[dict], variant: dict) -> dict:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        gen = {"temperature": 0}
        if variant.get("mime", True):
            gen["responseMimeType"] = "application/json"
        if variant["schema"]:
            gen["responseSchema"] = {"type": "OBJECT", "properties": {"corrected": {"type": "STRING"}},
                                     "required": ["corrected"]}
        if variant["thinking"]:   # no "thinking" for a one-line correction: faster
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        body = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def chat(self, messages: list[dict], lang_code: str | None = None) -> str:
        self._ensure_model()
        while True:
            r = httpx.post(f"{GEMINI_URL}/models/{self._model}:generateContent",
                           json=self._body(messages, self.VARIANTS[self._variant]),
                           headers={"x-goog-api-key": self.key}, timeout=self.timeout)
            if r.status_code == 400 and self._variant + 1 < len(self.VARIANTS):
                self._variant += 1     # this model rejects a setting: try simpler settings
                continue
            break
        self._check(r)
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise CloudLLMError(f"Gemini gave no answer: {str(data)[:200]}")
        return _parse_json_answer("".join(p.get("text", "") for p in parts if not p.get("thought")))


class GroqClient(_CloudClient):
    name = "Groq"

    def __init__(self, key: str = config.GROQ_API_KEY, model: str | None = config.GROQ_MODEL or None,
                 timeout: float = 30):
        super().__init__(key, model, timeout)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def _choose_model(self) -> str | None:
        r = httpx.get(f"{GROQ_URL}/models", headers=self._headers(), timeout=self.timeout)
        self._check(r)
        return pick_groq_model([m["id"] for m in r.json().get("data", [])])

    def chat(self, messages: list[dict], lang_code: str | None = None) -> str:
        self._ensure_model()
        body = {"model": self._model, "messages": messages, "temperature": 0,
                "response_format": {"type": "json_object"}}
        if "gpt-oss" in self._model:
            body["reasoning_effort"] = "low"
        r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout)
        self._check(r)
        return _parse_json_answer(r.json()["choices"][0]["message"]["content"])


class ChainClient:
    """Try each client in order; a failure or rate limit moves on to the next. The client that
    answered is remembered (last_model, last_strong), so the result says which model fixed the text."""

    def __init__(self, clients: list, wait_on_rate_limit: bool = False):
        self.clients = clients
        self.last_model, self.last_strong = "", False
        self.problems: list[str] = []
        # evaluations: wait for a rate-limited cloud model instead of falling back to a weaker one
        # (otherwise the score would silently measure the fallback)
        self.wait_on_rate_limit = wait_on_rate_limit

    @property
    def model(self) -> str:
        return " -> ".join(c.model for c in self.clients)

    def supports(self, lang_code: str) -> bool:
        if self.wait_on_rate_limit and any(getattr(c, "strong", False) and c.key for c in self.clients):
            return True
        return any(_supports(c, lang_code) for c in self.clients)

    def chat(self, messages: list[dict], lang_code: str | None = None) -> str:
        for _ in range(30):
            try:
                return self._chat_once(messages, lang_code)
            except CloudLLMError:
                blocked = [c.blocked_until for c in self.clients if getattr(c, "strong", False) and c.key]
                if not (self.wait_on_rate_limit and blocked and min(blocked) > time.time()):
                    raise
                wait = min(blocked) - time.time() + 1
                print(f"    (free-tier rate limit: waiting {wait:.0f} s)", flush=True)
                time.sleep(wait)
        raise CloudLLMError("still rate-limited after waiting")

    def _chat_once(self, messages: list[dict], lang_code: str | None) -> str:
        errors = []
        has_strong = any(getattr(c, "strong", False) for c in self.clients)
        for client in self.clients:
            if lang_code and not _supports(client, lang_code):
                continue
            if self.wait_on_rate_limit and has_strong and not getattr(client, "strong", False):
                continue
            try:
                answer = client.chat(messages, lang_code) if _takes_lang(client) else client.chat(messages)
            except Exception as exc:   # rate limit, network, bad answer: try the next one
                errors.append(f"{getattr(client, 'name', client.model)}: {exc}")
                continue
            self.last_model, self.last_strong = client.model, getattr(client, "strong", False)
            self.problems = errors
            return answer
        self.problems = errors
        raise CloudLLMError("; ".join(errors) or "no LLM available for this language")

    def warm_up(self) -> None:
        for c in self.clients:
            try:
                c.warm_up()
            except Exception:
                pass

    def status(self) -> dict:
        parts = [dict(c.status(), provider=getattr(c, "name", "Ollama")) for c in self.clients]
        online = [p for p in parts if p.get("online") and p.get("model_available")]
        return {"online": bool(online), "model_available": bool(online),
                "model": online[0]["model"] if online else self.model, "chain": parts}


def _supports(client, lang_code: str) -> bool:
    fn = getattr(client, "supports", None)
    return fn(lang_code) if fn else lang_code not in config.STAGE2_SKIP_LANGUAGES


def _takes_lang(client) -> bool:
    import inspect
    try:
        return "lang_code" in inspect.signature(client.chat).parameters
    except (TypeError, ValueError):
        return False
