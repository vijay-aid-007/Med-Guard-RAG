""" 
Thin wrapper around embedder.embed_query() — gives retrieval its own 
named entry point without duplicating embedding logic.

"""

import numpy as np
from src.ingestion.embedder import embed_query


def get_query_vector(text: str) -> np.ndarray:
    """
    Returns a normalized (1, 384) float32 array for the given query.
    Delegates to embedder.py — same model, same normalization as
    ingestion. Guaranteed identical by sharing one code path.
    
    """
    return embed_query(text)