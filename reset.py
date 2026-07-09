"""
reset.py — wipe all derived/generated state before a fresh pipeline run.

Run this whenever you change the embedding model, reranker model,
chunking strategy, or anything else upstream of the FAISS index —
old vectors/chunks are NOT compatible with a new embedding model and
must be rebuilt from scratch.

Usage:
    python reset.py                # dry run — shows what WOULD be deleted
    python reset.py --yes          # actually deletes everything
    python reset.py --yes --keep-postgres   # skip wiping Postgres tables
    python reset.py --yes --keep-redis      # skip flushing Redis
"""

import argparse
import shutil
from pathlib import Path

from src.core.config import settings
from src.core.logging_config import logger


def _delete_file(path: Path, dry_run: bool) -> None:
    if not path.exists():
        logger.info(f"[skip]  {path} (not found)")
        return
    if dry_run:
        logger.info(f"[would delete] {path}")
    else:
        path.unlink()
        logger.info(f"[deleted] {path}")


def _delete_dir_contents(path: Path, dry_run: bool) -> None:
    if not path.exists():
        logger.info(f"[skip]  {path} (not found)")
        return
    for item in path.iterdir():
        if dry_run:
            logger.info(f"[would delete] {item}")
        else:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            logger.info(f"[deleted] {item}")


def reset_faiss(dry_run: bool) -> None:
    _delete_file(Path(settings.faiss_index_path), dry_run)
    _delete_file(Path(settings.faiss_metadata_path), dry_run)


def reset_processed_chunks(dry_run: bool) -> None:
    # NOTE: your chunker.py currently has a typo writing "chunks.json1"
    # (digit 1) instead of "chunks.jsonl" (letter l). Once you fix that
    # typo in chunker.py, change the line below to match.
    _delete_file(Path("data/processed/chunks.json1"), dry_run)
    _delete_file(Path("data/processed/benchmark_results.jsonl"), dry_run)
    _delete_file(Path("data/processed/benchmark_summary.json"), dry_run)
    # NOTE: benchmark_set.jsonl and eval_set.jsonl are your hand-authored
    # question sets, not generated output — intentionally left alone so
    # you can compare base vs domain-specific results on the same questions.


def reset_logs(dry_run: bool) -> None:
    _delete_dir_contents(Path("logs"), dry_run)


def reset_postgres(dry_run: bool) -> None:
    import sqlalchemy

    # NOTE: check scripts/init_db.sql for your real table names and
    # update this list to match — CREATE TABLE statements there are
    # the source of truth for your schema.
    tables = ["query_logs", "guardrail_events", "eval_runs"]

    if dry_run:
        logger.info(f"[would TRUNCATE] {tables} in {settings.postgres_url}")
        return

    engine = sqlalchemy.create_engine(settings.postgres_url)
    with engine.begin() as conn:
        for table in tables:
            try:
                conn.execute(sqlalchemy.text(f"TRUNCATE TABLE {table} CASCADE"))
                logger.info(f"[truncated] {table}")
            except Exception as e:
                logger.warning(f"[skip] {table}: {e}")


def reset_redis(dry_run: bool) -> None:
    import redis

    if dry_run:
        logger.info(f"[would FLUSHDB] {settings.redis_url}")
        return

    r = redis.from_url(settings.redis_url)
    r.flushdb()
    logger.info("[flushed] Redis DB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="actually delete (default is dry run)")
    parser.add_argument("--keep-postgres", action="store_true")
    parser.add_argument("--keep-redis", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args()

    dry_run = not args.yes
    if dry_run:
        logger.info("DRY RUN — nothing will be deleted. Pass --yes to actually reset.")

    reset_faiss(dry_run)
    reset_processed_chunks(dry_run)

    if not args.keep_logs:
        reset_logs(dry_run)
    if not args.keep_postgres:
        reset_postgres(dry_run)
    if not args.keep_redis:
        reset_redis(dry_run)

    if dry_run:
        logger.info("Dry run complete. Re-run with --yes to apply.")
    else:
        logger.info("Reset complete. Ready for a fresh pipeline run.")


if __name__ == "__main__":
    main()