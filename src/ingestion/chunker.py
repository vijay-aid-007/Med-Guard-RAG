"""
Splits documents into retrieval-sized chunks.

WHY chunk at all:
Embedding models have a context limit and lose precision on long text —
a 2000-word abstract embedded as one vector blurs together multiple facts
into a single point in vector space, so similarity search gets fuzzy.
Smaller chunks (~512 tokens) keep each vector tightly about one idea,
so retrieval finds the most relevant passage, not just the most relevant
document.

WHY overlap (64 tokens):
Without overlap, a sentence that starts at the end of chunk N and
finishes in chunk N+1 gets cut in half — neither chunk fully contains the
fact. A 64-token overlap means context that spans a chunk boundary still
appears whole in at least one chunk.

HOW:
We use a simple recursive splitter: try to split on paragraph breaks
first, then sentences, then words — only falling back to a hard
character cut if nothing else fits. This keeps chunks semantically
coheren

"""



import json
from pathlib import Path
from src.core.config import settings
from src.core.logging_config import logger


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents= True, exist_ok=True)


# After ✅ — sentence-aware splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _get_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size * 5,  # convert words→chars (~5 chars/word)
        chunk_overlap = overlap * 5,
        separators    = ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function = len,
    )

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    splitter = _get_splitter(chunk_size, overlap)
    chunks   = splitter.split_text(text)
    return chunks if chunks else [text]



def chunk_documents(docs: list[dict]) -> list[dict]:
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    chunked = []
    for doc in docs:
        pieces = _split_text(doc["text"], size, overlap)
        for i, piece in enumerate(pieces):
            chunked.append({
                "chunk_id": f"{doc['id']}_chunk{i}",
                "doc_id": doc["id"],
                "text": piece,
                "source": doc["source"],
                "metadata" : doc.get("metadata", {}),
            })
    logger.info(
        f"Chunked {len(docs)} documents -> {len(chunked)} chunks"
    )
    return chunked


def main():
    print(">>> chunker main() started", flush=True)
    input_path = Path("data/raw/combined_corpus.jsonl")
    print(">>> looking for:", input_path.resolve(), "exists:", input_path.exists(), flush=True)
    if not input_path.exists():
        logger.error("Run loader.py first")
        return
    
    docs = [json.loads(i) for i in open(input_path, encoding='utf-8')]
    print(">>> loaded docs:", len(docs), flush=True)
    chunks = chunk_documents(docs)

    out = PROCESSED_DIR / "chunks.jsonl"
    with open(out, "w", encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(">>> saved to:", out.resolve(), flush=True)
    logger.info(f"saved {len(chunks)} chunks -> {out}")




if __name__ == "__main__":
    main()