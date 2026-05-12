import structlog
from typing import List, Optional
from .base import SearchResult

logger = structlog.get_logger(__name__)


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is required for CohereReranker. Install with 'pip install cohere'")
        
        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key)

    def rerank(self, query: str, results: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
        documents = [r.text for r in results]
        
        rerank_response = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=self.model,
            return_documents=False
        )
        
        reranked_results = []
        for item in rerank_response.results:
            original_result = results[item.index]
            new_metadata = original_result.metadata.copy()
            new_metadata["rerank_score"] = item.relevance_score
            reranked_results.append(
                SearchResult(
                    text=original_result.text,
                    score=item.relevance_score,
                    metadata=new_metadata
                )
            )
        
        cost = (1 / 1000) * 2.00
        logger.info(
            "Cohere rerank completed",
            model=self.model,
            top_n=top_n,
            cost_usd=cost
        )
        
        return reranked_results


class LocalReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers is required for LocalReranker. Install with 'pip install sentence-transformers'")
        
        self.model_name = model
        self.model = CrossEncoder(model)

    def rerank(self, query: str, results: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)
        
        scored_results = []
        for i, score in enumerate(scores):
            original_result = results[i]
            new_metadata = original_result.metadata.copy()
            new_metadata["rerank_score"] = float(score)
            scored_results.append(
                SearchResult(
                    text=original_result.text,
                    score=float(score),
                    metadata=new_metadata
                )
            )
        
        reranked_results = sorted(
            scored_results,
            key=lambda r: r.score,
            reverse=True
        )[:top_n]
        
        logger.info(
            "Local rerank completed",
            model=self.model_name,
            top_n=top_n
        )
        
        return reranked_results
