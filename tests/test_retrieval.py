import pytest


@pytest.mark.integration
class TestRetriever:

    @pytest.fixture(scope="class")
    def retriever(self):
        from src.retrieval.retriever import get_retriever
        return get_retriever()

    def test_returns_results(self, retriever):
        results = retriever.retrieve("What are symptoms of diabetes?", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_results_have_required_fields(self, retriever):
        results = retriever.retrieve("What causes hypertension?", top_k=3)
        for chunk in results:
            assert "chunk_id"        in chunk
            assert "text"            in chunk
            assert "source"          in chunk
            assert "similarity_score" in chunk

    def test_results_sorted_descending(self, retriever):
        results = retriever.retrieve("What is insulin resistance?", top_k=10)
        scores  = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
class TestReranker:

    @pytest.fixture(scope="class")
    def retriever(self):
        from src.retrieval.retriever import get_retriever
        return get_retriever()

    @pytest.fixture(scope="class")
    def reranker(self):
        from src.retrieval.reranker import get_reranker
        return get_reranker()

    def test_rerank_returns_fewer_results(self, retriever, reranker):
        q          = "What are side effects of metformin?"
        candidates = retriever.retrieve(q, top_k=20)
        reranked   = reranker.rerank(q, candidates, top_k=5)
        assert len(reranked) <= 5

    def test_rerank_adds_score(self, retriever, reranker):
        q          = "What causes high cholesterol?"
        candidates = retriever.retrieve(q, top_k=10)
        reranked   = reranker.rerank(q, candidates, top_k=5)
        for chunk in reranked:
            assert "rerank_score" in chunk

    def test_rerank_sorted_descending(self, retriever, reranker):
        q          = "What is the mechanism of beta blockers?"
        candidates = retriever.retrieve(q, top_k=15)
        reranked   = reranker.rerank(q, candidates, top_k=8)
        scores     = [c["rerank_score"] for c in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_input(self, reranker):
        result = reranker.rerank("any query", [], top_k=5)
        assert result == []