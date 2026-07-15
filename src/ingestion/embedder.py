"""
In this we are going to convert chunks into vector embeddings using embedding pre-trained
embedding models from open source HuggingFace. The model is "SentenceTransformer"

This embeddr.py turns each chunk into a 384 dimensional vector using all_miniLM-L6-V2 
embedding model 

The Vector has 384 numbers because all-miniLM-L6-V2 produces 384-dimensional embedding 
that captures the semantic meaning as well 

Each chunk (512 tokens long) is passed into an embedding model, 
which outputs a fixed-length vector (e.g., 384 dimensions)

Chunk text: 512 tokens.
Embedding model: all-MiniLM-L6-v2 (SentenceTransformers).
Output: A 384-dimensional vector representing the semantic meaning of that chunk.
"""

"""
One sentence job: Turns text into 384-dimensional normalized vectors — shared by ingestion (batch) 
and retrieval (single query) so both always use the same model and normalization.
Understand before writing:

This file is imported by indexer.py, retriever.py, input_guard.py, 
and output_guard.py. They all need vectors from the same model — if each 
loaded its own SentenceTransformer instance, you'd have 4 copies of an 80MB model in RAM simultaneously.
The singleton pattern here (_embedding_model = None, loaded once) keeps only one copy alive for the whole process lifetime.
normalize_embeddings=True means every vector is L2-normalized to unit length — which makes inner 
product equal to cosine similarity. FAISS's IndexFlatIP (inner product) is only meaningful as 
cosine similarity if this is true.

"""


import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.config import settings
from src.core.logging_config import logger

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        # Verify actual output dimension matches config
        test_vec = np.asarray(_embedding_model.encode(["test"], convert_to_numpy=True))
        actual_dim = test_vec.shape[1]
        if actual_dim != settings.embedding_dim:
            logger.warning(
                f"Dim mismatch: model outputs {actual_dim}, "
                f"config says {settings.embedding_dim}. "
                f"Update EMBEDDING_DIM in .env"
            )
    return _embedding_model


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    model = get_embedding_model()
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec, dtype="float32")