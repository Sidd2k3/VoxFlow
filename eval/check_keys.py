"""Check the free cloud grammar models: is each key working, which model was chosen, how fast is it?

    python -m eval.check_keys

Sends one Hindi sentence with two mistakes to each provider that has a key in .env.
"""

import time

from app import config, stage2
from app.cloud_llm import GeminiClient, GroqClient

SENTENCE = "मेरी बहन कल बाजार गया था। उसने वहाँ से सब्जी खरीदा।"


def show_models(client) -> None:
    """On a failure: which models does this key offer? (to pick another one with GEMINI_MODEL / GROQ_MODEL)"""
    import httpx
    from app.cloud_llm import GEMINI_URL, GROQ_URL
    try:
        if isinstance(client, GeminiClient):
            r = httpx.get(f"{GEMINI_URL}/models", params={"pageSize": 1000}, headers={"x-goog-api-key": client.key})
            names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
        else:
            r = httpx.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {client.key}"})
            names = [m["id"] for m in r.json().get("data", [])]
        print("  models this key offers: " + ", ".join(sorted(names)) + "\n")
    except Exception as exc:
        print(f"  (could not list the models: {exc})\n")


def main():
    providers = [("Gemini", config.GEMINI_API_KEY, GeminiClient), ("Groq", config.GROQ_API_KEY, GroqClient)]
    if not any(key for _, key, _ in providers):
        raise SystemExit("No keys found. Put GEMINI_API_KEY=... and GROQ_API_KEY=... in the .env file "
                         "in the project folder.")
    print(f"Test sentence: {SENTENCE}\n")
    for name, key, cls in providers:
        if not key:
            print(f"{name}: no key in .env (skipped)\n")
            continue
        client = cls()
        try:
            client.warm_up()   # picks the model (not counted in the time)
            t0 = time.perf_counter()
            result = stage2.correct(SENTENCE, "hi-IN", 50, client)
            ms = round((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            print(f"{name}: FAILED - {exc}\n")
            continue
        if result.status in ("applied", "unchanged"):
            print(f"{name}: OK  model {client.model}, {ms} ms\n  -> {result.text}\n")
        else:
            print(f"{name}: FAILED ({result.status}) with model {client.model} - {result.reason}\n")
            show_models(client)


if __name__ == "__main__":
    main()
