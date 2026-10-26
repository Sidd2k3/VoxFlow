"""Download an Indian-accented English test set: clips from AI4Bharat's Svarah (117 speakers, 65 places).

    python -m eval.fetch_svarah --n 60

Svarah is on Hugging Face (ai4bharat/Svarah). If the dataset is gated, open
https://huggingface.co/datasets/ai4bharat/Svarah, accept its terms, run `hf auth login`, and retry.
Clips are spread across the first data shard and saved as 16 kHz WAV in eval/data_svarah/.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np

from app.audio import TARGET_RATE, resample, to_wav_bytes

REPO = "ai4bharat/Svarah"
OUT = Path(__file__).resolve().parent / "data_svarah"
TEXT_COLUMNS = ("text", "transcript", "transcription", "sentence", "normalized_text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    try:
        files = HfApi().list_repo_files(REPO, repo_type="dataset")
    except Exception as exc:
        raise SystemExit(f"Could not open {REPO}: {exc}\n"
                         f"If it is gated: accept the terms at https://huggingface.co/datasets/{REPO}, "
                         f"run `hf auth login`, then retry.")
    shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        raise SystemExit(f"No parquet files found. Files in the repo:\n" + "\n".join(files[:50]))
    try:
        path = hf_hub_download(REPO, shards[0], repo_type="dataset")
    except Exception as exc:
        if "gated" in type(exc).__name__.lower() or "403" in str(exc):
            raise SystemExit(
                f"Svarah is gated: your Hugging Face account needs access first.\n"
                f"  1. Open https://huggingface.co/datasets/{REPO}, log in and accept the terms\n"
                f"  2. Create a Read token at https://huggingface.co/settings/tokens\n"
                f"  3. Run: hf auth login   (paste the token)\n"
                f"  4. Run this command again")
        raise

    import pyarrow.parquet as pq
    import soundfile as sf
    table = pq.read_table(path)
    names = table.column_names
    text_col = next((c for c in TEXT_COLUMNS if c in names), None)
    audio_col = next((c for c in names if "audio" in c.lower()), None)
    if not text_col or not audio_col:
        raise SystemExit(f"Unexpected columns {names}; expected an audio column and one of {TEXT_COLUMNS}")

    # usable clips: 2-20 s long, with a transcript; spread across the speakers' native languages
    durations = table.column("duration").to_pylist() if "duration" in names else [None] * table.num_rows
    texts = table.column(text_col).to_pylist()
    native = table.column("primary_language").to_pylist() if "primary_language" in names else ["?"] * table.num_rows
    by_language: dict[str, list[int]] = {}
    for i, (d, t) in enumerate(zip(durations, texts)):
        if t and str(t).strip() and (d is None or 2.0 <= d <= 20.0):
            by_language.setdefault(str(native[i]), []).append(i)
    chosen: list[int] = []
    while len(chosen) < args.n and any(by_language.values()):     # round-robin over native languages
        for lang in sorted(by_language):
            if by_language[lang] and len(chosen) < args.n:
                chosen.append(by_language[lang].pop(0))

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in sorted(chosen):
        audio = table.column(audio_col)[i].as_py()
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            continue
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        samples = resample(samples.mean(axis=1).astype(np.float32), rate)
        name = f"svarah_{i:05d}.wav"
        (OUT / name).write_bytes(to_wav_bytes(samples, TARGET_RATE))
        manifest.append({"file": name, "reference": str(texts[i]).strip(), "native_language": str(native[i])})
    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
                                        encoding="utf-8")
    langs = sorted({m["native_language"] for m in manifest})
    print(f"Saved {len(manifest)} Indian-accented English clips from speakers of {len(langs)} native languages to {OUT}")


if __name__ == "__main__":
    main()
