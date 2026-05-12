import structlog
import time
import tiktoken
import os
from typing import List, Dict, Any
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential
from ..retrieval.bm25_retrieval import BM25Index

logger = structlog.get_logger(__name__)


class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "documind"
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    COHERE_API_KEY: str = ""
    enable_reranking: bool = False
    rerank_model: str = "rerank-english-v3.0"
    rerank_top_n: int = 5
    rerank_fetch_k: int = 20

    class Config:
        env_file = ".env"


class IngestResult(BaseModel):
    total: int
    collection: str
    duration_seconds: float


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStoreManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.tokenizer = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
        logger.info("VectorStoreManager initialized", qdrant_url=settings.QDRANT_URL)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_collection(self, collection_name: str, vector_size: int = 1536) -> None:
        if self.client.collection_exists(collection_name=collection_name):
            logger.info("Collection already exists", collection=collection_name)
            return
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info("Collection created", collection=collection_name, vector_size=vector_size)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def ingest(self, chunks: List[Document], collection_name: str) -> IngestResult:
        start_time = time.time()
        total_chunks = len(chunks)
        
        logger.info("Starting ingestion", total_chunks=total_chunks, collection=collection_name)
        
        batch_size = 100
        all_points = []
        total_tokens = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            
            # Compute token count
            batch_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
            total_tokens += batch_tokens
            
            # Get embeddings
            batch_embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare points
            for j, (doc, embedding) in enumerate(zip(batch, batch_embeddings)):
                point = PointStruct(
                    id=doc.metadata.get("chunk_id", str(i + j)),
                    vector=embedding,
                    payload={
                        **doc.metadata,
                        "text": doc.page_content
                    }
                )
                all_points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=collection_name,
            points=all_points
        )
        
        # Calculate cost
        cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
        logger.info(
            "Ingestion complete",
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            estimated_cost=cost,
            collection=collection_name
        )
        
        duration = time.time() - start_time
        return IngestResult(
            total=total_chunks,
            collection=collection_name,
            duration_seconds=duration
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def similarity_search(self, query: str, collection_name: str, top_k: int = 5) -> List[SearchResult]:
        logger.info("Starting similarity search", query=query, collection=collection_name, top_k=top_k)
        
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                text=result.payload["text"],
                score=result.score,
                metadata={k: v for k, v in result.payload.items() if k != "text"}
            ))
        
        logger.info("Search complete", results_count=len(search_results))
        return search_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            logger.info("Collection deleted", collection=collection_name)
        else:
            logger.warning("Collection does not exist", collection=collection_name)

    def rebuild_index(self, chunks: List[Document], collection_name: str) -> None:
        bm25_index = BM25Index()
        bm25_index.build(chunks)
        index_path = os.path.join("data", "indexes", f"{collection_name}_bm25.pkl")
        bm25_index.save(index_path)
        logger.info("BM25 index rebuilt", collection=collection_name)
