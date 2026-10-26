"""Download a small test set: the first N clips of each language from Google's FLEURS test split.

FLEURS has real human recordings with reference transcripts for all five of our languages.
The audio archive is streamed and reading stops after N clips, so only a few MB are downloaded.

    python -m eval.fetch_fleurs --n 30
"""

import argparse
import csv
import json
import os
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
FLEURS_CODE = {"en-US": "en_us", "hi-IN": "hi_in", "mr-IN": "mr_in", "bn-IN": "bn_in", "pa-IN": "pa_in"}
DATA_DIR = Path(__file__).resolve().parent / "data"


def fetch_language(code: str, n: int) -> int:
    fleurs = FLEURS_CODE[code]
    out_dir = DATA_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True)

    # reference transcripts: id, file_name, raw_transcription, transcription, ...
    with urllib.request.urlopen(f"{BASE}/{fleurs}/test.tsv", timeout=60) as resp:
        rows = list(csv.reader(resp.read().decode("utf-8").splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE))
    reference = {row[1]: row[2] for row in rows if len(row) > 2}

    manifest, seen = [], set()
    with urllib.request.urlopen(f"{BASE}/{fleurs}/audio/test.tar.gz", timeout=60) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = os.path.basename(member.name)
                # FLEURS has several speakers per sentence: keep one recording per sentence
                if not member.isfile() or name not in reference or reference[name] in seen:
                    continue
                seen.add(reference[name])
                with open(out_dir / name, "wb") as f:
                    f.write(tar.extractfile(member).read())
                manifest.append({"file": name, "reference": reference[name]})
                if len(manifest) >= n:
                    break

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="clips per language")
    parser.add_argument("--langs", nargs="+", default=list(FLEURS_CODE))
    args = parser.parse_args()
    for code in args.langs:
        count = fetch_language(code, args.n)
        print(f"{code}: {count} clips saved to {DATA_DIR / code}")


if __name__ == "__main__":
    main()
