from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict


class RAGResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[SearchResult]
    retrieval_strategy: str = "naive"
    llm_model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    retrieved_from_dense: int = 0
    retrieved_from_sparse: int = 0
    reranked: bool = False
    rerank_score_top: float = 0.0
    hypothetical_document: str | None = None


class BaseRAGChain(ABC):
    @abstractmethod
    def query(self, question: str, collection_name: str, top_k: int = 5) -> RAGResponse:
        pass
