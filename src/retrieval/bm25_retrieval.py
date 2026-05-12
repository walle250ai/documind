import structlog
import pickle
import re
import os
from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .base import BaseRAGChain, RAGResponse, SearchResult
from ..ingestion.vectorstore import VectorStoreManager, Settings

logger = structlog.get_logger(__name__)


class BM25Index:
    def __init__(self):
        self.index = None
        self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build(self, chunks: List[Document]) -> None:
        self.chunks = chunks
        tokenized_chunks = [self._tokenize(chunk.page_content) for chunk in chunks]
        self.index = BM25Okapi(tokenized_chunks)
        logger.info("BM25 index built", chunk_count=len(chunks))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        normalized_scores = []
        
        for idx in top_indices:
            if max_score == min_score:
                normalized_score = 1.0
            else:
                normalized_score = (scores[idx] - min_score) / (max_score - min_score)
            
            normalized_scores.append(SearchResult(
                text=self.chunks[idx].page_content,
                score=normalized_score,
                metadata=self.chunks[idx].metadata
            ))
        
        return normalized_scores

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'index': self.index
            }, f)
        logger.info("BM25 index saved", path=path)

    @classmethod
    def load(cls, path: str) -> 'BM25Index':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        index = cls()
        index.chunks = data['chunks']
        index.index = data['index']
        logger.info("BM25 index loaded", path=path)
        return index


class BM25RAGChain(BaseRAGChain):
    retrieval_strategy = "bm25"

    def __init__(self, vectorstore: VectorStoreManager, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        self.bm25_indexes = {}
        
        if settings.LLM_PROVIDER == "openai":
            self.llm = self._create_openai_llm(settings)
        elif settings.LLM_PROVIDER == "anthropic":
            self.llm = self._create_anthropic_llm(settings)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        
        logger.info("BM25RAGChain initialized", llm_provider=settings.LLM_PROVIDER, llm_model=settings.LLM_MODEL)

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

    def query(self, question: str, collection_name: str, top_k: int = 5) -> RAGResponse:
        import time
        from langchain_core.messages import HumanMessage
        
        start_time = time.time()
        
        logger.info("Starting BM25 RAG query", question=question, collection=collection_name, top_k=top_k)
        
        bm25_index = self._get_bm25_index(collection_name)
        retrieved_chunks = bm25_index.search(question, top_k)
        
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk.text}" for i, chunk in enumerate(retrieved_chunks)])
        
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
            retrieved_chunks=retrieved_chunks,
            retrieval_strategy=self.retrieval_strategy,
            llm_model=self.settings.LLM_MODEL,
            latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost
        )
        
        logger.info(
            "BM25 RAG query complete",
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
