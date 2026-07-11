"""
csv_qa_agent/evaluation/benchmark.py
Benchmark suite for evaluating the CSV QA Agent.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import CSVQAAgent


def run_benchmark(csv_path: str = None):
    """Run the full evaluation benchmark."""

    if csv_path is None:
        csv_path = str(PROJECT_ROOT / "data" / "sales.csv")

    print("=" * 70)
    print("  CSV QA AGENT — EVALUATION BENCHMARK")
    print("=" * 70)

    agent = CSVQAAgent(csv_path)

    # Define benchmark questions with expected answers
    benchmark = [
        {
            "question": "What is the total revenue?",
            "expected_answer_contains": "275,743",
            "expected_viz_type": "number"
        },
        {
            "question": "Which region had the highest revenue?",
            "expected_answer_contains": "North",
            "expected_viz_type": "bar"
        },
        {
            "question": "Which product made the most revenue?",
            "expected_answer_contains": "Epsilon AI",
            "expected_viz_type": "bar"
        },
        {
            "question": "What is the average revenue by customer segment?",
            "expected_answer_contains": "Enterprise",
            "expected_viz_type": "table"
        },
        {
            "question": "What is the month over month growth?",
            "expected_answer_contains": "2024-02",
            "expected_viz_type": "line"
        },
        {
            "question": "Compare Enterprise vs SMB sales",
            "expected_answer_contains": "breakdown",
            "expected_viz_type": "table"
        },
        {
            "question": "What is the correlation between units and unit price?",
            "expected_answer_contains": "-0.76",
            "expected_viz_type": "number"
        },
        {
            "question": "How many unique customer segments does each sales rep handle?",
            "expected_answer_contains": "rep",
            "expected_viz_type": "table"
        }
    ]

    results = agent.evaluate(benchmark)

    print(f"\n📊 RESULTS")
    print(f"   Total Questions: {results['total_questions']}")
    print(f"   Correct: {results['correct']}/{results['total_questions']}")
    print(f"   Accuracy: {results['accuracy']}%")
    print(f"   Success Rate: {results['success_rate']}%")
    print(f"   Avg Latency: {results['avg_latency']}s")
    print(f"\n📋 DETAILED RESULTS:")

    for i, r in enumerate(results['results'], 1):
        status = "✅" if r.get('correct') else "❌"
        mode = r.get('mode', 'unknown')
        print(f"\n   {status} Q{i}: {r['question']}")
        print(f"      Mode: {mode} | Latency: {r.get('latency', 0):.2f}s")
        if 'error' in r:
            print(f"      Error: {r['error']}")
        else:
            print(f"      Answer: {r['answer'][:80]}...")
            print(f"      Viz: {r.get('viz_type', 'none')}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()
