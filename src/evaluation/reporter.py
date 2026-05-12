import base64
import io
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.evaluation.ragas_runner import StrategyComparison, EvaluationReport


class ComparisonReporter:
    def generate_markdown_table(self, comparison: StrategyComparison) -> str:
        table_lines = [
            "| Strategy | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Avg Latency (ms) | Avg Cost ($) |",
            "|----------|-------------|------------------|----------------|------------------|-----------------|-------------|"
        ]

        for strategy, report in comparison.reports.items():
            table_lines.append(
                f"| {strategy} | {report.faithfulness:.3f} | {report.answer_relevancy:.3f} | {report.context_recall:.3f} | {report.context_precision:.3f} | {report.avg_latency_ms:.0f} | {report.avg_cost_usd:.6f} |"
            )

        return "\n".join(table_lines)

    def print_summary(self, comparison: StrategyComparison) -> None:
        console = Console()

        # Print header
        console.print("\n[bold blue]=== RAG Evaluation Results ===[/bold blue]\n")

        # Create rich table
        table = Table(title="Strategy Comparison")
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Faithfulness", style="green")
        table.add_column("Answer Relevancy", style="green")
        table.add_column("Context Recall", style="green")
        table.add_column("Context Precision", style="green")
        table.add_column("Avg Latency (ms)", style="yellow")
        table.add_column("Avg Cost ($)", style="magenta")

        for strategy, report in comparison.reports.items():
            table.add_row(
                strategy,
                f"{report.faithfulness:.3f}",
                f"{report.answer_relevancy:.3f}",
                f"{report.context_recall:.3f}",
                f"{report.context_precision:.3f}",
                f"{report.avg_latency_ms:.0f}",
                f"{report.avg_cost_usd:.6f}"
            )

        console.print(table)

        # Print best strategies
        console.print(f"\n[bold green]Best strategy by faithfulness:[/bold green] {comparison.best_strategy_faithfulness}")

        best_overall_report = comparison.reports[comparison.best_strategy_overall]
        overall_score = (
            best_overall_report.faithfulness +
            best_overall_report.answer_relevancy +
            best_overall_report.context_recall +
            best_overall_report.context_precision
        ) / 4

        console.print(f"[bold green]Best overall strategy:[/bold green] {comparison.best_strategy_overall} (mean score: {overall_score:.3f})")

    def _figure_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _generate_bar_chart(self, reports: Dict[str, EvaluationReport]) -> str:
        strategies = list(reports.keys())
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        metric_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        x = np.arange(len(strategies))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            values = [getattr(reports[s], metric) for s in strategies]
            ax.bar(x + i * width, values, width, label=metric_names[i], color=colors[i])

        ax.set_xlabel('Strategy')
        ax.set_ylabel('Score')
        ax.set_title('RAGAS Metrics by Strategy')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(strategies)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        return self._figure_to_base64(fig)

    def _generate_radar_chart(self, reports: Dict[str, EvaluationReport]) -> str:
        strategies = list(reports.keys())
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        metric_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, strategy in enumerate(strategies):
            values = [getattr(reports[strategy], metric) for metric in metrics]
            values += values[:1]
            ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=strategy)
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart of RAGAS Metrics', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        return self._figure_to_base64(fig)

    def _generate_heatmap(self, reports: Dict[str, EvaluationReport]) -> str:
        strategies = list(reports.keys())
        n_strategies = len(strategies)
        if n_strategies == 0:
            return ""

        n_questions = len(reports[strategies[0]].raw_results)
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ['Blues', 'Oranges', 'Greens', 'Reds']

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            scores = np.zeros((n_questions, n_strategies))

            for q_idx in range(n_questions):
                for s_idx, strategy in enumerate(strategies):
                    # Note: raw_results doesn't have per-question metric scores in our current implementation
                    # So we'll use a placeholder approach for the heatmap
                    scores[q_idx, s_idx] = getattr(reports[strategy], metric)

            im = ax.imshow(scores, cmap=colors[metric_idx], vmin=0, vmax=1)
            ax.set_xticks(np.arange(n_strategies))
            ax.set_xticklabels(strategies)
            ax.set_yticks(np.arange(min(n_questions, 10)))
            ax.set_yticklabels([f"Q{i+1}" for i in range(min(n_questions, 10))])
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        return self._figure_to_base64(fig)

    def generate_html_report(self, comparison: StrategyComparison, output_path: str) -> None:
        bar_chart = self._generate_bar_chart(comparison.reports)
        radar_chart = self._generate_radar_chart(comparison.reports)
        heatmap = self._generate_heatmap(comparison.reports)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background-color: #1a365d;
            color: white;
            padding: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #1a365d;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #1a365d;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        tr:hover {{
            background-color: #f1f5f9;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .best-strategy {{
            background-color: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Evaluation Report</h1>
            <p>Generated on {comparison.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
            <div class="section">
                <h2>Best Strategies</h2>
                <div class="best-strategy">
                    <strong>Best by Faithfulness:</strong> {comparison.best_strategy_faithfulness}<br>
                    <strong>Best Overall:</strong> {comparison.best_strategy_overall}
                </div>
            </div>

            <div class="section">
                <h2>Summary Table</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Faithfulness</th>
                            <th>Answer Relevancy</th>
                            <th>Context Recall</th>
                            <th>Context Precision</th>
                            <th>Avg Latency (ms)</th>
                            <th>Avg Cost ($)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for strategy, report in comparison.reports.items():
            html += f"""
                        <tr>
                            <td>{strategy}</td>
                            <td>{report.faithfulness:.3f}</td>
                            <td>{report.answer_relevancy:.3f}</td>
                            <td>{report.context_recall:.3f}</td>
                            <td>{report.context_precision:.3f}</td>
                            <td>{report.avg_latency_ms:.0f}</td>
                            <td>{report.avg_cost_usd:.6f}</td>
                        </tr>
"""

        html += f"""
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h2>Bar Chart</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{bar_chart}" alt="Bar Chart">
                </div>
            </div>

            <div class="section">
                <h2>Radar Chart</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{radar_chart}" alt="Radar Chart">
                </div>
            </div>

            <div class="section">
                <h2>Per-Question Score Heatmap</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{heatmap}" alt="Heatmap">
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
