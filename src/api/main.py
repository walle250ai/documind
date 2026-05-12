#!/usr/bin/env python3

import os
import time
import structlog
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .cost_tracker import CostTracker, CostSummary
from ..ingestion.vectorstore import VectorStoreManager, Settings, IngestResult
from ..ingestion.loader import DocumentLoader, DocumentChunker, ChunkingStrategy
from ..retrieval.base import RAGResponse, BaseRAGChain
from ..retrieval.naive import NaiveRAGChain
from ..retrieval.hybrid import HybridRAGChain
from ..retrieval.hyde import HyDERAGChain
from ..retrieval.bm25_retrieval import BM25Index

logger = structlog.get_logger(__name__)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)
query_latency_seconds = Histogram(
    "query_latency_seconds",
    "Latency of query requests",
    ["strategy"]
)
query_cost_usd_total = Counter(
    "query_cost_usd_total",
    "Total cost of query requests in USD",
    ["strategy"]
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    collection_name: str
    retrieval_strategy: str
    top_k: int = 5

class CollectionInfo(BaseModel):
    name: str
    document_count: int

# Global variables
settings: Settings = None
vectorstore: VectorStoreManager = None
document_loader: DocumentLoader = None
document_chunker: DocumentChunker = None
naive_chain: BaseRAGChain = None
hybrid_chain: BaseRAGChain = None
hyde_chain: BaseRAGChain = None
hybrid_rerank_chain: BaseRAGChain = None
bm25_indexes: Dict[str, BM25Index] = {}
cost_tracker: CostTracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, vectorstore, document_loader, document_chunker
    global naive_chain, hybrid_chain, hyde_chain, hybrid_rerank_chain, bm25_indexes, cost_tracker
    
    # Initialize CostTracker
    cost_tracker = CostTracker()
    logger.info("CostTracker initialized")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize vectorstore
    vectorstore = VectorStoreManager(settings)
    
    # Initialize loader and chunker
    document_loader = DocumentLoader()
    document_chunker = DocumentChunker(vectorstore.embeddings)
    
    # Initialize all chains
    naive_chain = NaiveRAGChain(vectorstore, settings)
    
    # Hybrid chain without reranking
    hybrid_settings = Settings(**settings.model_dump())
    hybrid_settings.enable_reranking = False
    hybrid_chain = HybridRAGChain(vectorstore, hybrid_settings)
    
    # HyDE chain
    hyde_chain = HyDERAGChain(vectorstore, settings)
    
    # Hybrid chain with reranking
    rerank_settings = Settings(**settings.model_dump())
    rerank_settings.enable_reranking = True
    hybrid_rerank_chain = HybridRAGChain(vectorstore, rerank_settings)
    
    # Load BM25 indexes from disk if they exist
    indexes_dir = Path("data") / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    for index_file in indexes_dir.glob("*_bm25.pkl"):
        collection_name = index_file.stem.replace("_bm25", "")
        try:
            index = BM25Index.load(str(index_file))
            bm25_indexes[collection_name] = index
            logger.info("Loaded BM25 index", collection=collection_name)
        except Exception as e:
            logger.error("Failed to load BM25 index", collection=collection_name, error=str(e))
    
    logger.info("DocuMind API startup complete")
    yield
    logger.info("DocuMind API shutdown complete")

# Initialize app
app = FastAPI(title="DocuMind API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(process_time, 2)
    )
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    return response

@app.get("/health")
async def health_check():
    try:
        # Check Qdrant connection
        vectorstore.client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        qdrant_status = "error"
    return {"status": "ok", "qdrant": qdrant_status, "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/ingest", response_model=IngestResult)
async def ingest(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    strategy: ChunkingStrategy = Form(...)
):
    start_time = time.time()
    
    # Save uploaded file temporarily
    temp_dir = Path("data") / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Load document
        docs = document_loader.load(str(temp_path))
        
        # Chunk document
        chunks = document_chunker.chunk(docs, strategy)
        
        # Ingest to Qdrant
        result = vectorstore.ingest(chunks, collection_name)
        
        # Rebuild BM25 index
        vectorstore.rebuild_index(chunks, collection_name)
        
        # Load the new BM25 index into memory
        index_path = Path("data") / "indexes" / f"{collection_name}_bm25.pkl"
        if index_path.exists():
            bm25_indexes[collection_name] = BM25Index.load(str(index_path))
        
        duration = time.time() - start_time
        logger.info("Ingestion complete", collection=collection_name, duration=duration)
        return result
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

@app.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    
    # Route to correct chain
    if request.retrieval_strategy == "naive":
        chain = naive_chain
    elif request.retrieval_strategy == "hybrid":
        chain = hybrid_chain
    elif request.retrieval_strategy == "hyde":
        chain = hyde_chain
    elif request.retrieval_strategy == "hybrid_rerank":
        chain = hybrid_rerank_chain
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.retrieval_strategy}")
    
    # Execute query
    with query_latency_seconds.labels(strategy=request.retrieval_strategy).time():
        response = chain.query(request.question, request.collection_name, request.top_k)
    
    # Update cost metric
    query_cost_usd_total.labels(strategy=request.retrieval_strategy).inc(response.estimated_cost_usd)
    
    # Log query with CostTracker
    cost_tracker.log_query(response)
    
    logger.info("Query complete", strategy=request.retrieval_strategy, cost=response.estimated_cost_usd)
    return response

@app.get("/collections", response_model=List[CollectionInfo])
async def get_collections():
    collections = vectorstore.client.get_collections().collections
    collection_info = []
    for collection in collections:
        count = vectorstore.client.count(collection.name).count
        collection_info.append(CollectionInfo(name=collection.name, document_count=count))
    return collection_info

@app.get("/cost-summary", response_model=CostSummary)
async def get_cost_summary(since_days: int = 30):
    return cost_tracker.get_summary(since_days=since_days)
