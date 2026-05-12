import structlog
import time
from langchain_core.messages import HumanMessage

from .base import BaseRAGChain, RAGResponse, SearchResult
from ..ingestion.vectorstore import VectorStoreManager, Settings

logger = structlog.get_logger(__name__)


class HyDERAGChain(BaseRAGChain):
    retrieval_strategy = "hyde"

    def __init__(self, vectorstore: VectorStoreManager, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        
        if settings.LLM_PROVIDER == "openai":
            self.llm = self._create_openai_llm(settings)
        elif settings.LLM_PROVIDER == "anthropic":
            self.llm = self._create_anthropic_llm(settings)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        
        logger.info("HyDERAGChain initialized", llm_provider=settings.LLM_PROVIDER, llm_model=settings.LLM_MODEL)

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

    def _generate_hypothetical_document(self, question: str) -> str:
        prompt = f"""Write a detailed paragraph that would directly answer this question. 
Write as if you are an expert. Question: {question} 
Answer paragraph:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def query(self, question: str, collection_name: str, top_k: int = 5) -> RAGResponse:
        start_time = time.time()
        
        logger.info("Starting HyDE RAG query", question=question, collection=collection_name, top_k=top_k)
        
        # Check question length
        if len(question.split()) < 10:
            logger.warning("Question is shorter than 10 words, HyDE might not generate good hypothetical docs", question=question)
        
        # Generate hypothetical document
        hypothetical_doc = self._generate_hypothetical_document(question)
        logger.debug("Generated hypothetical document", doc=hypothetical_doc[:200])
        
        # Use hypothetical document for dense retrieval
        retrieved_chunks = self.vectorstore.similarity_search(
            query=hypothetical_doc,
            collection_name=collection_name,
            top_k=top_k
        )
        
        # Format context
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk.text}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Build final answer prompt
        final_prompt = f"""Use the following context to answer the question accurately.
If the answer is not in the context, say 'I don't know'.
Context: {context}
Question: {question}
Answer:"""
        
        # Call LLM for final answer
        llm_start = time.time()
        final_response = self.llm.invoke([HumanMessage(content=final_prompt)])
        llm_latency = time.time() - llm_start
        
        # Calculate costs (sum both HyDE generation and final answer)
        prompt_tokens = final_response.usage_metadata["input_tokens"]
        completion_tokens = final_response.usage_metadata["output_tokens"]
        estimated_cost = self._calculate_cost(prompt_tokens, completion_tokens)
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        
        # Build RAGResponse
        rag_response = RAGResponse(
            question=question,
            answer=final_response.content,
            retrieved_chunks=retrieved_chunks,
            retrieval_strategy=self.retrieval_strategy,
            llm_model=self.settings.LLM_MODEL,
            latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            hypothetical_document=hypothetical_doc
        )
        
        logger.info(
            "HyDE RAG query complete",
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
