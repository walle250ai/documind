#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.ingestion.vectorstore import VectorStoreManager, Settings
from src.retrieval import NaiveRAGChain, BM25RAGChain, HybridRAGChain, HyDERAGChain
from src.evaluation.golden_set_builder import GoldenSetBuilder
from src.evaluation.ragas_runner import RAGASEvaluator
from src.evaluation.reporter import ComparisonReporter


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on multiple RAG strategies")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--golden", required=True, help="Path to golden Q&A JSON file")
    parser.add_argument("--output-dir", default="data/eval_results", help="Output directory for evaluation results")

    args = parser.parse_args()

    # Initialize settings and vector store
    settings = Settings()
    vectorstore = VectorStoreManager(settings)

    # Load golden set
    builder = GoldenSetBuilder()
    golden_set = builder.load(args.golden)
    print(f"Loaded {len(golden_set)} golden Q&A pairs")

    # Initialize chains
    chains = {
        "naive": NaiveRAGChain(vectorstore, settings),
        "bm25": BM25RAGChain(vectorstore, settings),
        "hybrid": HybridRAGChain(vectorstore, settings),
        "hyde": HyDERAGChain(vectorstore, settings)
    }

    # Run evaluation
    evaluator = RAGASEvaluator()
    comparison = evaluator.evaluate(chains, golden_set, args.collection)

    # Save results
    output_path = evaluator.save_comparison(comparison, args.output_dir)
    print(f"Evaluation complete! Results saved to: {output_path}")

    # Generate and save HTML report
    reporter = ComparisonReporter()
    html_path = Path(args.output_dir) / "latest_report.html"
    reporter.generate_html_report(comparison, str(html_path))
    print(f"HTML report saved to: {html_path}")

    # Print markdown table
    print("\n=== Markdown Table ===")
    print(reporter.generate_markdown_table(comparison))

    # Print rich summary
    reporter.print_summary(comparison)

    # Print best overall strategy with mean score
    best_overall_report = comparison.reports[comparison.best_strategy_overall]
    mean_score = (
        best_overall_report.faithfulness +
        best_overall_report.answer_relevancy +
        best_overall_report.context_recall +
        best_overall_report.context_precision
    ) / 4
    print(f"\nBest overall strategy: {comparison.best_strategy_overall} (mean score: {mean_score:.3f})")


if __name__ == "__main__":
    main()
