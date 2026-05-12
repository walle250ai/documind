import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from src.retrieval.base import BaseRAGChain, RAGResponse
from src.evaluation.golden_set_builder import GoldenQA


class EvaluationReport(BaseModel):
    strategy: str
    n_questions: int
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    avg_latency_ms: float
    avg_cost_usd: float
    evaluated_at: datetime
    raw_results: List[Dict]


class StrategyComparison(BaseModel):
    reports: Dict[str, EvaluationReport]
    best_strategy_faithfulness: str
    best_strategy_overall: str
    generated_at: datetime


class RAGASEvaluator:
    def evaluate(
        self,
        chains: Dict[str, BaseRAGChain],
        golden_set: List[GoldenQA],
        collection_name: str
    ) -> StrategyComparison:
        reports: Dict[str, EvaluationReport] = {}
        all_strategy_results = {}

        for strategy, chain in chains.items():
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            raw_results = []
            latencies = []
            costs = []

            for qa in golden_set:
                response: RAGResponse = chain.query(qa.question, collection_name)
                questions.append(qa.question)
                answers.append(response.answer)
                contexts.append([chunk.text for chunk in response.retrieved_chunks])
                ground_truths.append(qa.ground_truth)
                latencies.append(response.latency_ms)
                costs.append(response.estimated_cost_usd)

                raw_results.append({
                    "question": qa.question,
                    "answer": response.answer,
                    "retrieved_contexts": [chunk.text for chunk in response.retrieved_chunks],
                    "ground_truth": qa.ground_truth
                })

            # Build ragas dataset
            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths
            })

            # Run ragas evaluation
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
            )

            # Create EvaluationReport
            report = EvaluationReport(
                strategy=strategy,
                n_questions=len(golden_set),
                faithfulness=float(result["faithfulness"]),
                answer_relevancy=float(result["answer_relevancy"]),
                context_recall=float(result["context_recall"]),
                context_precision=float(result["context_precision"]),
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                avg_cost_usd=sum(costs) / len(costs) if costs else 0.0,
                evaluated_at=datetime.now(),
                raw_results=raw_results
            )

            reports[strategy] = report
            all_strategy_results[strategy] = result

        # Determine best strategies
        best_faithfulness = max(reports.keys(), key=lambda k: reports[k].faithfulness)
        best_overall = max(
            reports.keys(),
            key=lambda k: (
                reports[k].faithfulness +
                reports[k].answer_relevancy +
                reports[k].context_recall +
                reports[k].context_precision
            ) / 4
        )

        comparison = StrategyComparison(
            reports=reports,
            best_strategy_faithfulness=best_faithfulness,
            best_strategy_overall=best_overall,
            generated_at=datetime.now()
        )

        return comparison

    def save_comparison(self, comparison: StrategyComparison, output_dir: str = "data/eval_results") -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"{timestamp}_comparison.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(comparison.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        return str(file_path)
