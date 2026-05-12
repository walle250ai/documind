#!/usr/bin/env python3

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel

from ..retrieval.base import RAGResponse


class CostSummary(BaseModel):
    total_queries: int
    total_cost_usd: float
    cost_by_strategy: Dict[str, float]
    avg_cost_per_query: float
    most_expensive_strategy: str
    cheapest_strategy: str
    daily_costs: List[Dict[str, Any]]


class CostTracker:
    TOKEN_COSTS = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
        "text-embedding-3-small": {"embedding": 0.02},
    }

    def __init__(self, log_file_path: str = "data/cost_log.jsonl"):
        self.log_file_path = Path(log_file_path)
        # Ensure parent directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        if not self.log_file_path.exists():
            self.log_file_path.touch()

    def calculate_llm_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Try to find model in TOKEN_COSTS, or use default based on model name
        if model in self.TOKEN_COSTS:
            costs = self.TOKEN_COSTS[model]
        elif "gpt-4o-mini" in model:
            costs = self.TOKEN_COSTS["gpt-4o-mini"]
        elif "gpt-4o" in model:
            costs = self.TOKEN_COSTS["gpt-4o"]
        elif "claude-haiku" in model:
            costs = self.TOKEN_COSTS["claude-haiku-3-5"]
        else:
            # Default to gpt-4o-mini pricing
            costs = self.TOKEN_COSTS["gpt-4o-mini"]

        input_cost = (prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (completion_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def calculate_embedding_cost(self, model: str, token_count: int) -> float:
        if model in self.TOKEN_COSTS and "embedding" in self.TOKEN_COSTS[model]:
            cost_per_million = self.TOKEN_COSTS[model]["embedding"]
        else:
            # Default to text-embedding-3-small pricing
            cost_per_million = self.TOKEN_COSTS["text-embedding-3-small"]["embedding"]
        return (token_count / 1_000_000) * cost_per_million

    def log_query(self, response: RAGResponse) -> None:
        log_entry = {
            "timestamp": time.time(),
            "question": response.question,
            "retrieval_strategy": response.retrieval_strategy,
            "llm_model": response.llm_model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "estimated_cost_usd": response.estimated_cost_usd,
            "latency_ms": response.latency_ms
        }
        # Append as JSON line
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_summary(self, since_days: int = 30) -> CostSummary:
        cutoff_time = time.time() - (since_days * 24 * 3600)
        entries = []

        with open(self.log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("timestamp", 0) >= cutoff_time:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

        total_queries = len(entries)
        total_cost_usd = sum(e.get("estimated_cost_usd", 0.0) for e in entries)
        cost_by_strategy = {}
        daily_data = {}

        for entry in entries:
            strategy = entry.get("retrieval_strategy", "unknown")
            cost = entry.get("estimated_cost_usd", 0.0)
            cost_by_strategy[strategy] = cost_by_strategy.get(strategy, 0.0) + cost

            # Calculate daily costs
            timestamp = entry.get("timestamp", time.time())
            date_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            if date_str not in daily_data:
                daily_data[date_str] = {"cost": 0.0, "queries": 0}
            daily_data[date_str]["cost"] += cost
            daily_data[date_str]["queries"] += 1

        avg_cost_per_query = total_cost_usd / total_queries if total_queries > 0 else 0.0

        # Find most expensive and cheapest strategies
        most_expensive_strategy = "none"
        cheapest_strategy = "none"
        if cost_by_strategy:
            sorted_strategies = sorted(cost_by_strategy.items(), key=lambda x: x[1], reverse=True)
            most_expensive_strategy = sorted_strategies[0][0]
            cheapest_strategy = sorted_strategies[-1][0]

        # Format daily costs
        daily_costs = [
            {"date": date, "cost": data["cost"], "queries": data["queries"]}
            for date, data in sorted(daily_data.items())
        ]

        return CostSummary(
            total_queries=total_queries,
            total_cost_usd=total_cost_usd,
            cost_by_strategy=cost_by_strategy,
            avg_cost_per_query=avg_cost_per_query,
            most_expensive_strategy=most_expensive_strategy,
            cheapest_strategy=cheapest_strategy,
            daily_costs=daily_costs
        )
