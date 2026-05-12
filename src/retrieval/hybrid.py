import structlog
import time
import os
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage

from .base import BaseRAGChain, RAGResponse, SearchResult
from .bm25_retrieval import BM25Index
from .reranker import CohereReranker, LocalReranker
from ..ingestion.vectorstore import VectorStoreManager, Settings

logger = structlog.get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60
) -> list[SearchResult]:
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result.metadata.get("chunk_id", str(hash(result.text)))
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
            fused_scores[chunk_id] += 1.0 / (k + rank)
    
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda cid: fused_scores[cid],
        reverse=True
    )
    
    fused_results = []
    for chunk_id in sorted_chunk_ids:
        result = chunk_map[chunk_id]
        new_metadata = result.metadata.copy()
        new_metadata["rrf_score"] = fused_scores[chunk_id]
        fused_results.append(
            SearchResult(
                text=result.text,
                score=fused_scores[chunk_id],
                metadata=new_metadata
            )
        )
    
    return fused_results


class HybridRAGChain(BaseRAGChain):
    retrieval_strategy = "hybrid"

    def __init__(self, vectorstore: VectorStoreManager, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        self.bm25_indexes = {}
        self.reranker: Optional[CohereReranker | LocalReranker] = None
        
        if settings.LLM_PROVIDER == "openai":
            self.llm = self._create_openai_llm(settings)
        elif settings.LLM_PROVIDER == "anthropic":
            self.llm = self._create_anthropic_llm(settings)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        
        if settings.enable_reranking:
            if settings.COHERE_API_KEY:
                self.reranker = CohereReranker(
                    api_key=settings.COHERE_API_KEY,
                    model=settings.rerank_model
                )
                logger.info("Using Cohere reranker", model=settings.rerank_model)
            else:
                self.reranker = LocalReranker()
                logger.info("Using local cross-encoder reranker")
        
        logger.info("HybridRAGChain initialized", llm_provider=settings.LLM_PROVIDER, llm_model=settings.LLM_MODEL)

    def _create_openai_llm(self, settings: Settings):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY
        )

    def _create_anthropic_llm(self, settings: Settings):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.ANTHROPIC_API_KEY
        )

    def _get_bm25_index(self, collection_name: str) -> BM25Index:
        if collection_name in self.bm25_indexes:
            return self.bm25_indexes[collection_name]
        
        index_path = os.path.join("data", "indexes", f"{collection_name}_bm25.pkl")
        if os.path.exists(index_path):
            index = BM25Index.load(index_path)
            self.bm25_indexes[collection_name] = index
            return index
        
        raise FileNotFoundError(f"BM25 index not found for collection {collection_name}")

    def query(
        self, 
        question: str, 
        collection_name: str, 
        top_k: int = 5,
        dense_weight: float = 0.7, 
        sparse_weight: float = 0.3
    ) -> RAGResponse:
        start_time = time.time()
        
        logger.info(
            "Starting hybrid RAG query",
            question=question,
            collection=collection_name,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        fetch_k = self.settings.rerank_fetch_k if self.settings.enable_reranking else top_k * 2
        
        dense_results = self.vectorstore.similarity_search(
            query=question,
            collection_name=collection_name,
            top_k=fetch_k
        )
        
        bm25_index = self._get_bm25_index(collection_name)
        sparse_results = bm25_index.search(
            query=question,
            top_k=fetch_k
        )
        
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
        
        reranked = False
        rerank_score_top = 0.0
        if self.settings.enable_reranking and self.reranker:
            top_fused = self.reranker.rerank(
                query=question,
                results=fused_results,
                top_n=top_k
            )
            reranked = True
            if top_fused:
                rerank_score_top = top_fused[0].metadata.get("rerank_score", 0.0)
        else:
            top_fused = fused_results[:top_k]
        
        dense_chunk_ids = {r.metadata.get("chunk_id", str(hash(r.text))) for r in dense_results}
        sparse_chunk_ids = {r.metadata.get("chunk_id", str(hash(r.text))) for r in sparse_results}
        
        retrieved_from_dense = 0
        retrieved_from_sparse = 0
        for result in top_fused:
            chunk_id = result.metadata.get("chunk_id", str(hash(result.text)))
            if chunk_id in dense_chunk_ids:
                retrieved_from_dense += 1
            if chunk_id in sparse_chunk_ids:
                retrieved_from_sparse += 1
        
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk.text}" for i, chunk in enumerate(top_fused)])
        
        prompt = f"""Use the following context to answer the question accurately.
If the answer is not in the context, say 'I don't know'.
Context: {context}
Question: {question}
Answer:"""
        
        llm_start = time.time()
        response = self.llm.invoke([HumanMessage(content=prompt)])
        llm_latency = time.time() - llm_start
        
        prompt_tokens = response.usage_metadata["input_tokens"]
        completion_tokens = response.usage_metadata["output_tokens"]
        estimated_cost = self._calculate_cost(prompt_tokens, completion_tokens)
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        
        rag_response = RAGResponse(
            question=question,
            answer=response.content,
            retrieved_chunks=top_fused,
            retrieval_strategy=self.retrieval_strategy,
            llm_model=self.settings.LLM_MODEL,
            latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            retrieved_from_dense=retrieved_from_dense,
            retrieved_from_sparse=retrieved_from_sparse,
            reranked=reranked,
            rerank_score_top=rerank_score_top
        )
        
        logger.info(
            "Hybrid RAG query complete",
            question=question,
            latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost
        )
        
        return rag_response

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if self.settings.LLM_PROVIDER == "openai":
            if self.settings.LLM_MODEL == "gpt-4o-mini":
                input_cost = (prompt_tokens / 1_000_000) * 0.15
                output_cost = (completion_tokens / 1_000_000) * 0.60
            else:
                input_cost = (prompt_tokens / 1_000_000) * 0.15
                output_cost = (completion_tokens / 1_000_000) * 0.60
        elif self.settings.LLM_PROVIDER == "anthropic":
            if "claude-haiku" in self.settings.LLM_MODEL:
                input_cost = (prompt_tokens / 1_000_000) * 0.80
                output_cost = (completion_tokens / 1_000_000) * 4.00
            else:
                input_cost = (prompt_tokens / 1_000_000) * 0.80
                output_cost = (completion_tokens / 1_000_000) * 4.00
        else:
            return 0.0
        
        return input_cost + output_cost
