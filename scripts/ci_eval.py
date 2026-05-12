import os
from pathlib import Path
from src.ingestion.vectorstore import VectorStoreManager, Settings
from src.ingestion.loader import DocumentLoader, DocumentChunker, ChunkingStrategy
from src.retrieval import NaiveRAGChain, HybridRAGChain
from src.evaluation.golden_set_builder import GoldenSetBuilder
from src.evaluation.ragas_runner import RAGASEvaluator
from src.evaluation.reporter import ComparisonReporter

def main():
    # Load 10 golden Q&As from tests/golden_qa.json (or use all if less than 10)
    golden_set_path = Path("tests/golden_qa.json")
    builder = GoldenSetBuilder()
    golden_set = builder.load(str(golden_set_path))
    golden_set = golden_set[:10]
    print(f"Loaded {len(golden_set)} golden Q&A pairs")
    
    # Initialize settings and vector store
    settings = Settings()
    vectorstore = VectorStoreManager(settings)
    collection_name = "ci_eval"
    
    # Create collection (delete if exists)
    if vectorstore.client.collection_exists(collection_name=collection_name):
        vectorstore.delete_collection(collection_name)
    vectorstore.create_collection(collection_name)
    
    # Load and chunk sample document
    loader = DocumentLoader()
    docs = loader.load("tests/fixtures/sample.txt")
    chunker = DocumentChunker()
    chunks = chunker.chunk(docs, ChunkingStrategy.FIXED)
    
    # Ingest into Qdrant and rebuild BM25 index
    vectorstore.ingest(chunks, collection_name)
    vectorstore.rebuild_index(chunks, collection_name)
    
    # Initialize chains (only Naive and Hybrid)
    chains = {
        "naive": NaiveRAGChain(vectorstore, settings),
        "hybrid": HybridRAGChain(vectorstore, settings)
    }
    
    # Run evaluation
    evaluator = RAGASEvaluator()
    comparison = evaluator.evaluate(chains, golden_set, collection_name)
    
    # Generate markdown table
    reporter = ComparisonReporter()
    markdown_table = reporter.generate_markdown_table(comparison)
    print("\n" + markdown_table)
    
    # Write to GitHub step summary if available
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(markdown_table + "\n")
    
    # Check thresholds and fail CI if needed
    failed = False
    for strategy, report in comparison.reports.items():
        print(f"\nChecking {strategy}:")
        print(f"  Faithfulness: {report.faithfulness:.3f} (threshold: 0.65)")
        print(f"  Answer Relevancy: {report.answer_relevancy:.3f} (threshold: 0.70)")
        
        if report.faithfulness < 0.65:
            print(f"  ❌ Faithfulness below threshold!")
            failed = True
        if report.answer_relevancy < 0.70:
            print(f"  ❌ Answer Relevancy below threshold!")
            failed = True
    
    if failed:
        print("\n❌ CI failed due to low evaluation scores!")
        exit(1)
    else:
        print("\n✅ All evaluation scores are above thresholds!")
        exit(0)

if __name__ == "__main__":
    main()
