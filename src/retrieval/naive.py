import structlog
import time
from typing import List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .base import BaseRAGChain, RAGResponse, SearchResult
from ..ingestion.vectorstore import VectorStoreManager, Settings

logger = structlog.get_logger(__name__)


class NaiveRAGChain(BaseRAGChain):
    def __init__(self, vectorstore: VectorStoreManager, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        
        if settings.LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
        elif settings.LLM_PROVIDER == "anthropic":
            self.llm = ChatAnthropic(
                model=settings.LLM_MODEL,
                api_key=settings.ANTHROPIC_API_KEY
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        
        logger.info("NaiveRAGChain initialized", llm_provider=settings.LLM_PROVIDER, llm_model=settings.LLM_MODEL)

    def query(self, question: str, collection_name: str, top_k: int = 5) -> RAGResponse:
        start_time = time.time()
        
        logger.info("Starting RAG query", question=question, collection=collection_name, top_k=top_k)
        
        # Step 1: Retrieve chunks
        retrieved_chunks = self.vectorstore.similarity_search(
            query=question,
            collection_name=collection_name,
            top_k=top_k
        )
        
        # Step 2: Format context
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk.text}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Step 3: Build prompt
        prompt = f"""Use the following context to answer the question accurately.
If the answer is not in the context, say 'I don't know'.
Context: {context}
Question: {question}
Answer:"""
        
        # Step 4: Call LLM
        llm_start = time.time()
        response = self.llm.invoke([HumanMessage(content=prompt)])
        llm_latency = time.time() - llm_start
        
        # Calculate costs
        prompt_tokens = response.usage_metadata["input_tokens"]
        completion_tokens = response.usage_metadata["output_tokens"]
        estimated_cost = self._calculate_cost(prompt_tokens, completion_tokens)
        
        # Calculate total latency
        total_latency_ms = int((time.time() - start_time) * 1000)
        
        # Build RAGResponse
        rag_response = RAGResponse(
            question=question,
            answer=response.content,
            retrieved_chunks=retrieved_chunks,
            retrieval_strategy="naive",
            llm_model=self.settings.LLM_MODEL,
            latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost
        )
        
        logger.info(
            "RAG query complete",
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
                # Default to gpt-4o-mini pricing if model unknown
                input_cost = (prompt_tokens / 1_000_000) * 0.15
                output_cost = (completion_tokens / 1_000_000) * 0.60
        elif self.settings.LLM_PROVIDER == "anthropic":
            if "claude-haiku" in self.settings.LLM_MODEL:
                input_cost = (prompt_tokens / 1_000_000) * 0.80
                output_cost = (completion_tokens / 1_000_000) * 4.00
            else:
                # Default to claude-haiku pricing if model unknown
                input_cost = (prompt_tokens / 1_000_000) * 0.80
                output_cost = (completion_tokens / 1_000_000) * 4.00
        else:
            return 0.0
        
        return input_cost + output_cost
