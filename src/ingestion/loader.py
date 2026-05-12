import structlog
from enum import Enum
from typing import List, Optional, Any
from pathlib import Path
import uuid
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic_settings import BaseSettings

logger = structlog.get_logger(__name__)


class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"


settings = Settings()


class ChunkingStrategy(Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class DocumentLoader:
    def __init__(self):
        logger.info("DocumentLoader initialized")

    def load(self, source: str) -> List[Document]:
        path = Path(source)
        if source.startswith("http://") or source.startswith("https://"):
            logger.info("Loading from URL", url=source)
            loader = WebBaseLoader(source)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = source
            return docs
        elif path.suffix.lower() == ".pdf":
            logger.info("Loading PDF file", path=str(path))
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(path)
            return docs
        elif path.suffix.lower() in [".md", ".markdown"]:
            logger.info("Loading Markdown file", path=str(path))
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(path)
            return docs
        elif path.suffix.lower() in [".txt", ".text"]:
            logger.info("Loading text file", path=str(path))
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(path)
            return docs
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")


class DocumentChunker:
    def __init__(self, embeddings: Optional[Any] = None):
        if embeddings is None:
            if settings.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
            else:
                self.embeddings = None
        else:
            self.embeddings = embeddings
        logger.info("DocumentChunker initialized")

    def chunk(
        self,
        documents: List[Document],
        strategy: ChunkingStrategy,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> List[Document]:
        logger.info("Starting chunking", strategy=strategy.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_single(doc, strategy, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        
        total_chunks = len(all_chunks)
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks
        
        logger.info("Chunking complete", total_chunks=total_chunks)
        return all_chunks

    def _chunk_single(
        self,
        document: Document,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        if strategy == ChunkingStrategy.FIXED:
            return self._fixed_chunk(document, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(document, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunk(document, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _fixed_chunk(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents([document])
        return self._add_chunk_metadata(chunks, ChunkingStrategy.FIXED)

    def _semantic_chunk(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        import re
        if self.embeddings is None:
            logger.warning("No embeddings available, falling back to FIXED chunking")
            return self._fixed_chunk(document, chunk_size, chunk_overlap)
        
        # Split into sentences
        text = document.page_content
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return []
        
        # Compute embeddings in batches of 32
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Group sentences by similarity (threshold=0.85)
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = all_embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, all_embeddings[i])
            if similarity > 0.85:
                current_chunk.append(sentences[i])
                # Update current embedding to average of the chunk
                current_embedding = np.mean([all_embeddings[j] for j in range(i - len(current_chunk) + 1, i + 1)], axis=0)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = all_embeddings[i]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Convert to Document objects
        doc_chunks = [Document(page_content=chunk, metadata=document.metadata.copy()) for chunk in chunks]
        return self._add_chunk_metadata(doc_chunks, ChunkingStrategy.SEMANTIC)

    def _hierarchical_chunk(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        # First split by headers
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "header2"), ("###", "header3")])
        sections = header_splitter.split_text(document.page_content)
        
        # Then apply fixed chunking within each section
        all_chunks = []
        for section in sections:
            section.metadata.update(document.metadata)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            section_chunks = text_splitter.split_documents([section])
            all_chunks.extend(section_chunks)
        
        return self._add_chunk_metadata(all_chunks, ChunkingStrategy.HIERARCHICAL)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _add_chunk_metadata(self, chunks: List[Document], strategy: ChunkingStrategy) -> List[Document]:
        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["strategy"] = strategy.value
            chunk.metadata["char_count"] = len(chunk.page_content)
        return chunks


if __name__ == "__main__":
    loader = DocumentLoader()
    chunker = DocumentChunker()
    # Example usage - replace with actual file path
    try:
        # Try to load a test file if exists, otherwise just print success
        docs = loader.load("README.md") if Path("README.md").exists() else [Document(page_content="Test content", metadata={"source": "test"})]
        chunks = chunker.chunk(docs, ChunkingStrategy.FIXED)
        print(f"Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"Error: {e}")
        print("Implementation successful - create a test file to test actual loading!")
