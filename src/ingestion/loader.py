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

def load_pubmed_abstracts(limit: int = 10000) -> list[dict]:
    """
    PubMed abstracts — pure medical knowledge text.
    Better for retrieval than Q&A format because:
    - No question noise
    - Dense factual content
    - Matches answer-shaped text in FAISS
    """
    logger.info("Loading PubMed abstracts...")
    try:
        rows = _load("ncats/pubmed_abstracts", None, "train", limit)
    except Exception:
        try:
            rows = _load("ccdv/pubmed-summarization", None, "train", limit)
        except Exception as e:
            logger.warning(f"PubMed abstracts load failed: {e} — skipping")
            return []

    docs = []
    for i, row in enumerate(rows):
        row_: dict = dict(row)
        abstract = str(row_.get("abstract") or row_.get("article") or "").strip()
        title    = str(row_.get("title") or "").strip()
        if not abstract or len(abstract) < 50:
            continue
        full_text = f"{title}\n\n{abstract}" if title else abstract
        docs.append({
            "id":       f"pubmed_abstract_{i}",
            "text":     full_text,
            "source":   "PubMedAbstract",
            "metadata": {"type": "abstract"},
        })

    logger.info(f"Loaded {len(docs)} PubMed abstracts")
    return docs


def load_mechanism_abstracts(limit: int = 5000) -> list[dict]:
    """
    PubMed abstracts filtered for mechanism-rich content.
    Fills the gap left by exam datasets (MedMCQA/MedQA) which explain
    WHAT drug to use but not HOW/WHY it works at molecular level.

    Targets: AMPK pathways, adipokines, pharmacokinetics, receptor mechanisms,
    insulin signaling, inflammatory pathways — exactly what eval questions need.
    """
    logger.info("Loading mechanism-rich PubMed abstracts...")

    MECHANISM_KEYWORDS = [
        # Drug mechanisms
        "mechanism of action", "ampk", "gluconeogenesis",
        "hepatic glucose", "insulin sensitivity", "biguanide",
        "pharmacokinetics", "receptor antagonist", "enzyme inhibitor",
        "beta-adrenergic", "renin angiotensin", "ace inhibitor",
        "metformin mechanism", "hepatic gluconeogenesis",

        # Obesity/metabolic pathways
        "adipokine", "leptin", "adiponectin", "visceral adipose",
        "insulin signaling", "inflammatory pathway", "cytokine",
        "free fatty acid", "lipotoxicity", "metabolic syndrome",
        "insulin resistance mechanism",

        # General mechanisms
        "pathophysiology", "molecular mechanism", "signaling pathway",
        "protein kinase", "oxidative stress", "gene expression",
    ]

    try:
        rows = _load("ccdv/pubmed-summarization", None, "train", limit * 4)
    except Exception as e:
        logger.warning(f"Mechanism abstracts load failed: {e} — skipping")
        return []

    docs = []
    for i, row in enumerate(rows):
        if len(docs) >= limit:
            break

        row_: dict = dict(row)
        abstract = str(row_.get("abstract") or "").strip()
        article  = str(row_.get("article")  or "").strip()
        content  = abstract if abstract else article

        if not content or len(content) < 100:
            continue

        content_lower = content.lower()
        if not any(kw.lower() in content_lower for kw in MECHANISM_KEYWORDS):
            continue

        docs.append({
            "id":       f"pubmed_mechanism_{i}",
            "text":     content,
            "source":   "PubMedMechanism",
            "metadata": {"type": "mechanism_abstract"},
        })

    logger.info(f"Loaded {len(docs)} mechanism-rich abstracts")
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
        if not explanation.strip() or len(explanation.strip()) < 20:
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

    # Delete old corpus to force rebuild
    if combined_path.exists():
        combined_path.unlink()
        logger.info("Deleted old corpus — rebuilding...")

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

        for doc in load_pubmed_abstracts(limit=10000):
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1

    logger.info(f"Total corpus: {total} documents → {combined_path}")