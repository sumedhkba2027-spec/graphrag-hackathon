import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download, list_repo_files

# ── Edit this to scale the corpus up or down ──────────────────────────────────
TOKEN_TARGET = 2_000_000
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REPO = "FINDA-FIT/Fin_Corpus_EarningCall"
OUTPUT_DIR = Path("data/raw/earnings_calls")

# Skip transcripts longer than this to avoid memory spikes on 333K-char outliers
MAX_CHARS = 50_000


def safe_filename(id_str: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", id_str).strip("_") + ".json"


def download():
    enc = tiktoken.get_encoding("cl100k_base")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing parquet files in {DATASET_REPO} …")
    parquet_files = sorted(
        f for f in list_repo_files(DATASET_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    )
    print(f"Found {len(parquet_files)} parquet file(s)\n")
    print(f"Target: {TOKEN_TARGET:,} tokens  |  Max chars per doc: {MAX_CHARS:,}\n")

    total_tokens = 0
    total_chars = 0
    saved = 0
    done = False

    for parquet_path in parquet_files:
        if done:
            break

        print(f"Downloading {parquet_path} …")
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=parquet_path,
            repo_type="dataset",
        )

        table = pq.read_table(local_path, columns=["ID", "CONTEXT"])

        for batch in table.to_batches(max_chunksize=256):
            ids = batch.column("ID").to_pylist()
            contexts = batch.column("CONTEXT").to_pylist()

            for id_val, context in zip(ids, contexts):
                context = (context or "").strip()
                if not context or len(context) > MAX_CHARS:
                    continue

                token_count = len(enc.encode(context))
                filename = safe_filename(str(id_val))

                with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
                    json.dump({
                        "id": id_val,
                        "context": context,
                        "token_count": token_count,
                        "char_count": len(context),
                    }, f, ensure_ascii=False)

                total_tokens += token_count
                total_chars += len(context)
                saved += 1

                print(
                    f"  [{saved:>4}]  +{token_count:>6,}  cumulative: {total_tokens:>10,}",
                    end="\r",
                )

                if total_tokens >= TOKEN_TARGET:
                    done = True
                    break

            if done:
                break

    print()
    print(f"\n{'='*55}")
    print(f"Files saved   : {saved:,}")
    print(f"Total chars   : {total_chars:,}")
    print(f"Total tokens  : {total_tokens:,}  (cl100k_base)")
    print(f"Output dir    : {OUTPUT_DIR}")
    print(f"{'='*55}")


def tokenize():
    files = sorted(OUTPUT_DIR.glob("*.json"))
    if not files:
        print("No files found — run download first.")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    total_chars = 0

    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        token_count = obj.get("token_count") or len(enc.encode(obj.get("context", "")))
        total_tokens += token_count
        total_chars += obj.get("char_count") or len(obj.get("context", ""))

    print(f"\n{'='*55}")
    print(f"Files         : {len(files):,}")
    print(f"Total chars   : {total_chars:,}")
    print(f"Total tokens  : {total_tokens:,}  (cl100k_base)")
    print(f"Token target  : {TOKEN_TARGET:,}")
    print(f"{'='*55}")


if __name__ == "__main__":
    if "--tokenize-only" in sys.argv:
        tokenize()
    elif any(OUTPUT_DIR.glob("*.json")):
        count = len(list(OUTPUT_DIR.glob("*.json")))
        print(f"Found {count} existing files — skipping download.")
        print("Run with --tokenize-only to recount, or delete output dir to re-download.")
        tokenize()
    else:
        download()
