"""
Loads the FAISS index, embeds the query, returns top-k chunks with their metadata and similarity scores.

Understand before writing:
FAISS's .search(query_vec, top_k) returns two arrays: scores (inner product values) and indices (row positions 
in the index). You use indices[0][i] to look up metadata[i] — that's the text-to-vector alignment from indexer.py 
paying off here. FAISS returns -1 as an index when fewer than top_k vectors exist — always guard against that.

"""

# import pickle 
# import faiss
# from src.core.config import settings
# from src.core.logging_config import logger 
# from src.retrieval.query_embed import get_query_vector 

# class Retriever:
#     def __init__(self):
#         logger.info(f"Loading FAISS index: {settings.faiss_index_path}")
#         self.index = faiss.read_index(settings.faiss_index_path)
#         self.metadata = pickle.load(
#             open(settings.faiss_metadata_path, 'rb')
#         )
#         logger.info(f"Retriever ready — {self.index.ntotal} vectors")

#     def retrieve(self, query: str, top_k : int = None) -> list[dict]:        # type: ignore
#         top_k = top_k or settings.retrieval_top_k
#         query_vec = get_query_vector(query)

#         scores, indices = self.index.search(query_vec, top_k)

#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx == -1:               # FAISS padding — skip
#                 continue
#             chunk = self.metadata[idx]
#             results.append({
#                 **chunk, 
#                 "similarity_score": float(score),
#             })
#         logger.debug(f"Retrieved {len(results)} chunks (top_k = {top_k})")
#         return results

# _instance: Retriever | None = None 

# def get_retriever() -> Retriever:
#     global _instance 
#     if _instance is None:
#         _instance = Retriever()
#     return _instance










import pickle
import faiss
from src.core.config import settings
from src.core.logging_config import logger
from src.retrieval.query_embed import get_query_vector


class Retriever:

    def __init__(self):
        logger.info(f"Loading FAISS index: {settings.faiss_index_path}")
        self.index    = faiss.read_index(settings.faiss_index_path)
        self.metadata = pickle.load(
            open(settings.faiss_metadata_path, "rb")
        )
        logger.info(f"Retriever ready — {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k     = top_k or settings.retrieval_top_k
        query_vec = get_query_vector(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({**chunk, "similarity_score": float(score)})

        logger.debug(f"Retrieved {len(results)} chunks (top_k={top_k})")
        return results

    def retrieve_with_hyde(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """
        HyDE — Hypothetical Document Embedding.

        WHY this works better than raw query retrieval:
        Your FAISS index contains ANSWER-shaped text (explanations,
        conclusions, facts). A raw question like "What treats diabetes?"
        is QUESTION-shaped — semantically different from answer text
        even when the topic matches.

        HyDE bridges this gap:
        1. Generate a short hypothetical answer using the LLM
        2. Embed THAT answer (answer-shaped)
        3. Search against answer-shaped chunks
        → Much higher cosine similarity because shapes match
        """
        from src.generation.llm_client import get_llm_client

        hyde_prompt = (
            "You are a medical expert. Write a precise 2-sentence "
            "answer to the following question using medical terminology. "
            "Be specific, factual, and concise.\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            hypothetical = get_llm_client().generate(hyde_prompt)
            logger.info(f"HyDE hypothesis: {hypothetical[:80]}")
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e} — falling back to raw query")
            return self.retrieve(query, top_k)

        hyde_vec = get_query_vector(hypothetical)
        top_k    = top_k or settings.retrieval_top_k
        scores, indices = self.index.search(hyde_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({
                **chunk,
                "similarity_score": float(score),
                "retrieval_method": "hyde",
            })

        logger.info(f"HyDE retrieved {len(results)} chunks")
        return results


_instance: Retriever | None = None


def get_retriever() -> Retriever:
    global _instance
    if _instance is None:
        _instance = Retriever()
    return _instance