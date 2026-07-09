# STAGE 2 — Build Knowldegde Base; Run Once 

"""
Loads medical knowledge base data from HuggingFace datasets.

WHY two datasets:
PubMedQA gives grounded, abstract-backed answers (high trust — these are
real PubMed abstracts). MedMCQA gives breadth (194k questions covering
many specialties) but is exam-style, not abstract-grounded. We use
PubMedQA as the primary retrieval corpus (it has real source text to
ground answers in) and MedMCQA as a secondary corpus + evaluation set.

WHAT this module does:
Downloads both datasets via the `datasets` library (free, no API key),
normalizes them into one schema: {id, text, source, metadata}, and saves
as JSONL so ingestion only needs to be re-run if the corpus changes —
not on every server restart.

HOW:
HuggingFace `datasets.load_dataset()` streams and caches automatically.
We flatten PubMedQA's structured abstract (background/methods/results/
conclusion) into a single text block since that's what gets embedded.
"""

# import json
# from pathlib import Path
# from datasets import load_dataset
# from src.core.logging_config import logger 



# RAW_DIR = Path("data/raw")
# RAW_DIR.mkdir(parents = True, exist_ok= True)

# def load_pubmedqa(limit: int = 1000) -> list[dict]:
#     logger.info('Loading PubMedQA.....')
    
#     ds = load_dataset(
#         "qiaojin/PubMedQA", "pqa_labeled", split='train'
#     )

#     ds = ds.select(range(min(limit, len(ds))))      # type: ignore[arg-type]#because the dataset type isn't always Sized

#     docs = []
#     for row in ds:
#         row = dict(row)
#         contexts = row.get('context', {})
#         labels = contexts.get('labels', [])
#         sections = contexts.get('contexts', [])

#         abstract = "\n".join(
#             f"{lbl} : {txt}"
#             for lbl, txt in zip(labels, sections)
#         )

#         full_text = (
#             f"Question : {row['question']}\n\n"
#             f"Abstract : \n {abstract} \n\n"
#             f"Conslusion : {row.get('long_answer', '')}"
#         )

#         docs.append({
#             "id" :   f"pubmedqa_{row['pubid']}",
#             "text" :    full_text,
#             "source": "PubMedQA",
#             "metadata" : {
#                 "answer_label": row.get('final_decision', ''),
#                 "pubid" : row['pubid'],
#             },
#         })

#     logger.info(f"Loaded {len(docs)} PubMedQA documents")
#     return docs


# def load_medmcqa(limit: int = 3000) -> list[dict]:
#     logger.info('Loading MedMCQA.....')

#     ds = load_dataset(
#         "openlifescienceai/medmcqa", split="train"
#     )

#     ds = ds.select(range(min(limit, len(ds))))      # type: ignore[arg-type]

#     opt_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
    
#     docs = []
#     for row in ds:
#         row = dict(row)
#         explanation = row.get("exp", "") or ""
#         if not explanation.strip():
#             continue

#         correct_key = opt_map.get(row.get("cop", -1), "")
#         correct_answer = row.get(correct_key, "")

#         full_text = (
#             f"Question: {row['question']}\n\n"
#             f"Correct answer: {correct_answer}\n\n"
#             f"Explanation: {explanation}"
#         )

#         docs.append({
#             "id":       f"medmcqa_{row['id']}",
#             "text":     full_text,
#             "source":   "MedMCQA",
#             "metadata": {
#                 "subject": row.get("subject_name", "unknown"),
#                 "topic":   row.get("topic_name",   "unknown"),
#             },
#         })

#     logger.info(f"Loaded {len(docs)} MedMCQA documents")
#     return docs


# def save_json1(docs: list[dict], filename : str) -> None:
#     path = RAW_DIR / filename
#     with open(path, "w", encoding='utf-8') as f:
#         for doc in docs:
#             f.write(json.dumps(doc, ensure_ascii=False) + "\n")
#     logger.info(f"Saved {len(docs)} docs -> {path}")

# def main():
#     pubmed = load_pubmedqa(limit=1000)
#     mcqa = load_medmcqa(limit=3000)

#     save_json1(pubmed, "pubmedqa.json1")
#     save_json1(mcqa, "medmcqa.json1")

#     combined = pubmed + mcqa
#     save_json1(combined, "combined_corpus.json1")
#     logger.info(f"Total Corpus: {len(combined)} Documents")


# if __name__ == "__main__":
#     main()

    





import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from src.core.logging_config import logger

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str, config: str | None, split: str, limit: int) -> list[Any]:
    """
    Loads a HuggingFace dataset and returns a plain Python list of rows.
    Handles datasets where len() may not be available (streaming configs).
    """
    if config:
        ds = load_dataset(name, config, split=split)  # type: ignore[call-overload]
    else:
        ds = load_dataset(name, split=split)           # type: ignore[call-overload]

    try:
        total = len(ds)                                # type: ignore[arg-type]
    except TypeError:
        total = limit

    ds = ds.select(range(min(limit, total)))           # type: ignore[union-attr]
    return list(ds)                                    # type: ignore[return-value]


def load_pubmedqa(limit: int = 5000) -> list[dict]:
    """
    PubMedQA labeled split — real PubMed abstracts with yes/no/maybe labels.
    Note: labeled split only has ~1000 samples total regardless of limit.
    """
    logger.info("Loading PubMedQA...")
    rows = _load("qiaojin/PubMedQA", "pqa_labeled", "train", limit)

    docs = []
    for row in rows:
        row_: dict[str, Any] = dict(row)
        contexts = row_.get("context") or {}
        labels   = contexts.get("labels")   or []
        sections = contexts.get("contexts") or []

        abstract = "\n".join(
            f"{lbl}: {txt}"
            for lbl, txt in zip(labels, sections)
        )
        full_text = (
            f"Question: {row_.get('question', '')}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Conclusion: {row_.get('long_answer', '')}"
        )
        docs.append({
            "id":       f"pubmedqa_{row_.get('pubid', '')}",
            "text":     full_text,
            "source":   "PubMedQA",
            "metadata": {
                "answer_label": str(row_.get("final_decision", "")),
                "pubid":        str(row_.get("pubid", "")),
            },
        })

    logger.info(f"Loaded {len(docs)} PubMedQA documents")
    return docs


def load_medmcqa(limit: int = 20000) -> list[dict]:
    """
    MedMCQA — Indian medical entrance exam MCQs with explanations.
    Rows without explanations are skipped — nothing useful to retrieve from them.
    """
    logger.info("Loading MedMCQA...")
    rows = _load("openlifescienceai/medmcqa", None, "train", limit)

    opt_map: dict[int, str] = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
    docs = []

    for row in rows:
        row_: dict[str, Any] = dict(row)
        explanation: str = str(row_.get("exp") or "").strip()
        if not explanation:
            continue                          # skip rows with no explanation

        correct_key    = opt_map.get(int(row_.get("cop") or -1), "")
        correct_answer = str(row_.get(correct_key) or "")

        full_text = (
            f"Question: {row_.get('question', '')}\n\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Explanation: {explanation}"
        )
        docs.append({
            "id":       f"medmcqa_{row_.get('id', '')}",
            "text":     full_text,
            "source":   "MedMCQA",
            "metadata": {
                "subject": str(row_.get("subject_name") or "unknown"),
                "topic":   str(row_.get("topic_name")   or "unknown"),
            },
        })

    logger.info(f"Loaded {len(docs)} MedMCQA documents")
    return docs


def load_medqa(limit: int = 10000) -> list[dict]:
    """
    MedQA — USMLE-style questions with answers.
    Higher clinical reasoning quality than MedMCQA.
    Catches ACE inhibitor, beta-blocker, electrolyte content missing from other datasets.
    """
    logger.info("Loading MedQA (USMLE)...")
    try:
        rows = _load("bigbio/med_qa", "med_qa_en_source", "train", limit)
    except Exception as e:
        logger.warning(f"MedQA load failed: {e} — skipping")
        return []

    docs = []
    for i, row in enumerate(rows):
        row_: dict[str, Any] = dict(row)
        question = str(row_.get("question") or "").strip()
        answer   = str(row_.get("answer")   or "").strip()
        if not question or not answer:
            continue

        full_text = (
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )
        docs.append({
            "id":       f"medqa_{row_.get('id') or i}",
            "text":     full_text,
            "source":   "MedQA",
            "metadata": {"type": "usmle"},
        })

    logger.info(f"Loaded {len(docs)} MedQA documents")
    return docs


def save_jsonl(docs: list[dict], filename: str) -> None:
    path = RAW_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(docs)} docs → {path}")


def main() -> None:
    combined_path = RAW_DIR / "combined_corpus.jsonl"

    if combined_path.exists():
        count = sum(1 for _ in open(combined_path, encoding="utf-8"))
        logger.info(f"Corpus exists: {count} docs. Delete to rebuild.")
        return

    total = 0
    with open(combined_path, "w", encoding="utf-8") as out:

        for doc in load_pubmedqa(limit=5000):
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

        for doc in load_medmcqa(limit=20000):
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

        for doc in load_medqa(limit=10000):
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

    logger.info(f"Total corpus: {total} documents → {combined_path}")