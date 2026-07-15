"""
One sentence job: Adds new chunks to an EXISTING FAISS index without
rebuilding from scratch — same index type (IndexFlatIP), same metadata
alignment, same embed_texts() call as indexer.py.

Understand before writing:
FAISS IndexFlatIP supports .add() incrementally — new vectors are simply
appended after the last existing vector. The metadata list mirrors this:
metadata[i] always corresponds to FAISS row i. So we load the existing
metadata list, extend it with new entries, and pickle it back. The
position alignment is preserved automatically because both operations
(index.add + metadata.extend) append in the same order.

When to use this vs indexer.py:
- indexer.py    → full rebuild (new embedding model, new chunk strategy)
- this file     → add more data to existing index (safe, fast, no rebuild)

Usage:
    python -m src.ingestion.incremental_indexer --source data/processed/new_chunks.jsonl
"""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.core.config import settings
from src.core.logging_config import logger
from src.ingestion.embedder import embed_texts


def incremental_index(new_chunks_path: str) -> None:

    # ── 1. Validate new chunks file ───────────────────────────────────
    path = Path(new_chunks_path)
    if not path.exists():
        logger.error(f"New chunks file not found: {path}")
        return

    new_chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not new_chunks:
        logger.warning("No chunks found in file — nothing to add")
        return
    logger.info(f"New chunks to add: {len(new_chunks)}")

    # ── 2. Load existing FAISS index ──────────────────────────────────
    index_path    = Path(settings.faiss_index_path)
    metadata_path = Path(settings.faiss_metadata_path)

    if not index_path.exists() or not metadata_path.exists():
        logger.error(
            "Existing index not found — run indexer.py first to build the base index"
        )
        return

    index    = faiss.read_index(str(index_path))
    metadata = pickle.load(open(metadata_path, "rb"))

    before = index.ntotal
    logger.info(f"Existing index loaded — {before} vectors, dim={index.d}")

    # ── 3. Verify alignment before touching anything ──────────────────
    if index.ntotal != len(metadata):
        logger.error(
            f"Index/metadata misaligned: {index.ntotal} vectors vs "
            f"{len(metadata)} metadata entries — fix before adding more"
        )
        return

    # ── 4. Embed new chunks — same call as indexer.py ────────────────
    texts      = [c["text"] for c in new_chunks]
    embeddings = embed_texts(texts, batch_size=32, show_progress=True)

    # ── 5. Verify dimension matches existing index ────────────────────
    if embeddings.shape[1] != index.d:
        logger.error(
            f"Dimension mismatch: new embeddings are {embeddings.shape[1]}-dim "
            f"but existing index is {index.d}-dim — cannot mix"
        )
        return

    # ── 6. Add to existing index — same metadata schema as indexer.py ─
    index.add(embeddings.astype("float32"))         # type: ignore[arg-type]

    new_metadata = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id":   c["doc_id"],
            "text":     c["text"],
            "source":   c["source"],
            "metadata": c.get("metadata", {}),
        }
        for c in new_chunks
    ]
    metadata.extend(new_metadata)

    # ── 7. Save updated index + metadata ─────────────────────────────
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    after = index.ntotal
    logger.info(f"Incremental index complete:")
    logger.info(f"  Before : {before} vectors")
    logger.info(f"  Added  : {len(new_chunks)} vectors")
    logger.info(f"  After  : {after} vectors")
    logger.info(f"  Aligned: {after == len(metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add new chunks to existing FAISS index without rebuilding"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to new chunks .jsonl file (e.g. data/processed/new_chunks.jsonl)",
    )
    args = parser.parse_args()
    incremental_index(args.source)