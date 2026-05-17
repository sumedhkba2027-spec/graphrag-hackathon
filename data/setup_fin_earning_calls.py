import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import tiktoken
from huggingface_hub import HfApi, hf_hub_download


DATASET_REPO = "FINDA-FIT/Fin_Corpus_EarningCall"
DEFAULT_TARGET_TOKENS = 1_100_000
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_CHUNK_OVERLAP = 80
MIN_DOCUMENT_TOKENS = 100

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw" / "fin_corpus_earning_call"
BACKUP_DIR = DATA_DIR / "backups"
CHROMADB_PATH = ROOT_DIR / "chromadb"

ARTICLES_PATH = DATA_DIR / "articles.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download FINDA-FIT earning-call transcripts and prepare "
            "data/articles.json plus data/processed/chunks.json for the pipelines."
        )
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help="Stop after at least this many source text tokens are collected.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help="Maximum token length for each retrieval chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between consecutive chunks from the same document.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional hard cap on documents to keep.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up existing generated data files before replacing them.",
    )
    parser.add_argument(
        "--keep-vector-db",
        action="store_true",
        help="Keep the existing ChromaDB index instead of moving it aside for a rebuild.",
    )
    return parser.parse_args()


def clean_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def list_parquet_files():
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    return sorted(path for path in files if path.endswith(".parquet"))


def backup_existing_files(enabled, reset_vector_db):
    if not enabled:
        return []

    existing = [path for path in (ARTICLES_PATH, CHUNKS_PATH, MANIFEST_PATH) if path.exists()]
    has_vector_db = reset_vector_db and CHROMADB_PATH.exists()
    if not existing and not has_vector_db:
        return []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = BACKUP_DIR / f"fin_setup_{stamp}"
    target_dir.mkdir(parents=True, exist_ok=True)

    backups = []
    for source in existing:
        target = target_dir / source.relative_to(DATA_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        backups.append(str(target.relative_to(ROOT_DIR)))

    if has_vector_db:
        target = target_dir / "chromadb"
        shutil.move(str(CHROMADB_PATH), str(target))
        backups.append(str(target.relative_to(ROOT_DIR)))
    return backups


def encode(enc, text):
    return enc.encode(text, disallowed_special=())


def make_title(source_id, index):
    safe_id = clean_text(source_id)
    if safe_id:
        return f"Earning Call {safe_id}"
    return f"Earning Call {index}"


def collect_articles(target_tokens, max_documents):
    enc = tiktoken.get_encoding("cl100k_base")
    parquet_files = list_parquet_files()
    articles = []
    total_tokens = 0
    skipped_short = 0
    seen_ids = set()
    used_files = []

    for filename in parquet_files:
        print(f"Downloading/reading {filename}")
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
        )
        used_files.append(filename)

        frame = pd.read_parquet(local_path, columns=["ID", "CONTEXT"])
        for row in frame.itertuples(index=False):
            if max_documents is not None and len(articles) >= max_documents:
                return articles, total_tokens, skipped_short, used_files

            source_id = clean_text(row.ID)
            if source_id in seen_ids:
                continue

            text = clean_text(row.CONTEXT)
            token_count = len(encode(enc, text))
            if token_count < MIN_DOCUMENT_TOKENS:
                skipped_short += 1
                continue

            article_id = len(articles)
            articles.append(
                {
                    "id": article_id,
                    "title": make_title(source_id, article_id),
                    "text": text,
                    "source": DATASET_REPO,
                    "source_id": source_id,
                    "tokens": token_count,
                }
            )
            seen_ids.add(source_id)
            total_tokens += token_count

            if total_tokens >= target_tokens:
                return articles, total_tokens, skipped_short, used_files

    return articles, total_tokens, skipped_short, used_files


def token_windows(tokens, chunk_tokens, overlap):
    if chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("--chunk-overlap must be greater than or equal to 0 and less than --chunk-tokens")

    step = chunk_tokens - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_tokens, len(tokens))
        yield start, end, tokens[start:end]
        if end == len(tokens):
            break


def build_chunks(articles, chunk_tokens, chunk_overlap):
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for article in articles:
        tokens = encode(enc, article["text"])
        total_parts = max(1, math.ceil(max(1, len(tokens) - chunk_overlap) / (chunk_tokens - chunk_overlap)))

        for part_index, (start, end, window) in enumerate(
            token_windows(tokens, chunk_tokens, chunk_overlap),
            start=1,
        ):
            chunk_text = enc.decode(window).strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "article_id": article["id"],
                    "title": article["title"],
                    "text": chunk_text,
                    "source_id": article["source_id"],
                    "chunk_index": part_index,
                    "chunk_count": total_parts,
                    "token_start": start,
                    "token_end": end,
                    "tokens": len(window),
                }
            )

    return chunks


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    backups = backup_existing_files(
        enabled=not args.no_backup,
        reset_vector_db=not args.keep_vector_db,
    )
    articles, total_tokens, skipped_short, used_files = collect_articles(
        target_tokens=args.target_tokens,
        max_documents=args.max_documents,
    )

    if total_tokens < args.target_tokens:
        raise RuntimeError(
            f"Only collected {total_tokens:,} tokens, below target {args.target_tokens:,}."
        )

    chunks = build_chunks(
        articles=articles,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )

    manifest = {
        "dataset": DATASET_REPO,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_tokens": args.target_tokens,
        "total_tokens": total_tokens,
        "document_count": len(articles),
        "chunk_count": len(chunks),
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "min_document_tokens": MIN_DOCUMENT_TOKENS,
        "skipped_short_documents": skipped_short,
        "used_files": used_files,
        "output_files": [
            str(ARTICLES_PATH.relative_to(ROOT_DIR)),
            str(CHUNKS_PATH.relative_to(ROOT_DIR)),
            str(MANIFEST_PATH.relative_to(ROOT_DIR)),
        ],
        "backups": backups,
    }

    write_json(ARTICLES_PATH, articles)
    write_json(CHUNKS_PATH, chunks)
    write_json(MANIFEST_PATH, manifest)

    print("\nFINDA earning-call dataset is ready.")
    print(f"Documents: {len(articles):,}")
    print(f"Source tokens: {total_tokens:,}")
    print(f"Chunks: {len(chunks):,}")
    print(f"Wrote: {ARTICLES_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote: {CHUNKS_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote: {MANIFEST_PATH.relative_to(ROOT_DIR)}")
    if backups:
        print(f"Backups: {', '.join(backups)}")


if __name__ == "__main__":
    main()
