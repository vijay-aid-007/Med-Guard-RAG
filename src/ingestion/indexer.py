"""
One sentence job: Reads all chunks, calls embed_texts(), builds a FAISS IndexFlatIP, and saves both 
the index and a metadata pickle to disk.
Understand before writing:
FAISS only stores vectors — it has no idea what text those vectors came from. You save metadata.
pkl as a plain Python list where metadata[i] is the chunk whose embedding is at position i in the FAISS index. 
This alignment (position in FAISS = position in metadata list) is what lets retriever.py look up indices[0] from FAISS and 
immediately find the corresponding text.

"""  

import json
import pickle
from pathlib import Path

import faiss

from src.core.config import settings
from src.core.logging_config import logger
from src.ingestion.embedder import embed_texts


def build_index() -> None:
    chunks_path = Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        logger.error('Run chunker.py First')
        return
    
    chunks = [
        json.loads(i)
        for i in open(chunks_path, encoding='utf-8')
    ]
    logger.info(f"Loaded {len(chunks)} chunks for indexing")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, batch_size= 32, show_progress= True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)               # exact inner-product search
    index.add(embeddings.astype("float32"))      # type: ignore[arg-type]
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    #_____Save Index____________________
    index_path = Path(settings.faiss_index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    #_____Save Metadata aligned to FAISS row positions_____________
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "text": c["text"],
            "source": c["source"],
            "metadata": c.get("metadata", {}),
        }
        for c in chunks
    ]
    with open(settings.faiss_metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Index -> {index_path}")
    logger.info(f"Metadata -> {settings.faiss_metadata_path}")


if __name__ == "__main__":
    build_index()
