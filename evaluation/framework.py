"""
csv_qa_agent/evaluation/framework.py
Evaluation framework with benchmark suite, metrics tracking, and experiment comparison.
"""
import json
import csv
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from core.models import EvaluationResult, BenchmarkMetrics, PromptExperiment
from core.orchestrator import CSVQAAgent


class Evaluator:
    """Evaluates the CSV QA Agent against a test suite."""

    def __init__(
        self,
        csv_path: str,
        questions_file: Optional[str] = None,
        output_dir: str = "evaluation"
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.questions: List[Dict[str, Any]] = []
        self.results: List[EvaluationResult] = []

        if questions_file and os.path.exists(questions_file):
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)

    def add_question(
        self, 
        question: str, 
        expected: Any, 
        category: str = "general",
        tolerance: Optional[float] = None
    ):
        """Add a test question to the suite."""
        self.questions.append({
            'question': question,
            'expected': expected,
            'category': category,
            'tolerance': tolerance
        })

    def save_questions(self, filepath: Optional[str] = None):
        """Save questions to JSON file."""
        filepath = filepath or os.path.join(self.output_dir, 'questions.json')
        with open(filepath, 'w') as f:
            json.dump(self.questions, f, indent=2)

    def run_evaluation(
        self,
        model: str = "gpt-4o-mini",
        enable_critic: bool = True,
        max_retries: int = 3
    ) -> BenchmarkMetrics:
        """Run full evaluation and return metrics."""
        agent = CSVQAAgent(
            csv_path=self.csv_path,
            model=model,
            enable_critic=enable_critic,
            max_retries=max_retries
        )

        self.results = []
        total_latency = 0
        correct = 0
        failed = 0
        total_confidence = 0

        for q in self.questions:
            start = time.time()
            try:
                response = agent.answer(q['question'])
                latency = (time.time() - start) * 1000

                # Check correctness
                is_correct = self._check_correctness(
                    response.answer, 
                    q['expected'],
                    q.get('tolerance')
                )

                if is_correct:
                    correct += 1

                result = EvaluationResult(
                    question=q['question'],
                    expected=q['expected'],
                    actual=response.answer,
                    correct=is_correct,
                    latency_ms=latency,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    error=None
                )

                total_confidence += response.confidence

            except Exception as e:
                latency = (time.time() - start) * 1000
                failed += 1
                result = EvaluationResult(
                    question=q['question'],
                    expected=q['expected'],
                    actual=None,
                    correct=False,
                    latency_ms=latency,
                    confidence=0.0,
                    reasoning="",
                    error=str(e)
                )

            total_latency += latency
            self.results.append(result)

        n = len(self.questions)
        metrics = BenchmarkMetrics(
            accuracy=(correct / n * 100) if n > 0 else 0,
            avg_latency_ms=total_latency / n if n > 0 else 0,
            success_rate=((n - failed) / n * 100) if n > 0 else 0,
            total_questions=n,
            correct_answers=correct,
            failed_executions=failed,
            avg_confidence=total_confidence / n if n > 0 else 0
        )

        return metrics

    def _check_correctness(
        self, 
        actual: Any, 
        expected: Any, 
        tolerance: Optional[float] = None
    ) -> bool:
        """Check if actual answer matches expected."""
        try:
            # Try numeric comparison
            actual_num = float(str(actual).replace(',', '').replace('$', ''))
            expected_num = float(str(expected).replace(',', '').replace('$', ''))

            if tolerance:
                return abs(actual_num - expected_num) <= tolerance
            return abs(actual_num - expected_num) < 0.01
        except (ValueError, TypeError):
            # String comparison
            return str(actual).strip().lower() == str(expected).strip().lower()

    def save_results(self, metrics: BenchmarkMetrics, filepath: Optional[str] = None):
        """Save evaluation results to CSV."""
        filepath = filepath or os.path.join(self.output_dir, 'results.csv')

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'question', 'expected', 'actual', 'correct',
                'latency_ms', 'confidence', 'reasoning', 'error'
            ])

            for r in self.results:
                writer.writerow([
                    r.question, r.expected, r.actual, r.correct,
                    f"{r.latency_ms:.2f}", f"{r.confidence:.2f}",
                    r.reasoning[:200], r.error or ''
                ])

        # Also save metrics summary
        metrics_file = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

    def generate_report(self, metrics: BenchmarkMetrics) -> str:
        """Generate a text report of evaluation results."""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           CSV QA AGENT - EVALUATION REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
║ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<45} ║
╠══════════════════════════════════════════════════════════════╣
║ METRICS                                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Accuracy:        {metrics.accuracy:>10.2f}%                              ║
║  Avg Latency:     {metrics.avg_latency_ms:>10.2f}ms                            ║
║  Success Rate:     {metrics.success_rate:>10.2f}%                              ║
║  Avg Confidence:   {metrics.avg_confidence:>10.2f}%                              ║
║  Total Questions:  {metrics.total_questions:>10}                                ║
║  Correct Answers:  {metrics.correct_answers:>10}                                ║
║  Failed:           {metrics.failed_executions:>10}                                ║
╠══════════════════════════════════════════════════════════════╣
║ DETAILED RESULTS                                             ║
╠══════════════════════════════════════════════════════════════╣
"""
        for i, r in enumerate(self.results, 1):
            status = "✓" if r.correct else "✗"
            report += f"║ {i:3}. [{status}] {r.question[:50]:<50} ║\n"
            report += f"║      Expected: {str(r.expected)[:30]:<30} Actual: {str(r.actual)[:30]:<30} ║\n"
            report += f"║      Confidence: {r.confidence:.1f}%  Latency: {r.latency_ms:.1f}ms\n"
            if r.error:
                report += f"║      Error: {r.error[:60]}\n"
            report += "║\n"

        report += "╚══════════════════════════════════════════════════════════════╝"
        return report


class PromptExperimentRunner:
    """Run experiments comparing different prompt strategies."""

    def __init__(self, csv_path: str, questions_file: str):
        self.csv_path = csv_path
        self.evaluator = Evaluator(csv_path, questions_file)
        self.experiments: List[PromptExperiment] = []

    def add_experiment(self, experiment: PromptExperiment):
        """Add a prompt experiment to compare."""
        self.experiments.append(experiment)

    def run_all(self) -> Dict[str, BenchmarkMetrics]:
        """Run all experiments and return comparison."""
        results = {}

        for exp in self.experiments:
            print(f"\nRunning experiment: {exp.name}")
            metrics = self.evaluator.run_evaluation(
                model=exp.model,
                enable_critic=True
            )
            exp.metrics = metrics
            results[exp.name] = metrics

            print(f"  Accuracy: {metrics.accuracy:.2f}%")
            print(f"  Avg Latency: {metrics.avg_latency_ms:.2f}ms")
            print(f"  Success Rate: {metrics.success_rate:.2f}%")

        return results

    def comparison_table(self) -> str:
        """Generate a comparison table of all experiments."""
        table = "\n" + "="*80 + "\n"
        table += "PROMPT EXPERIMENT COMPARISON\n"
        table += "="*80 + "\n"
        table += f"{'Experiment':<20} {'Accuracy':<12} {'Latency':<12} {'Success':<12} {'Confidence':<12}\n"
        table += "-"*80 + "\n"

        for exp in self.experiments:
            if exp.metrics:
                table += f"{exp.name:<20} "
                table += f"{exp.metrics.accuracy:<12.2f}"
                table += f"{exp.metrics.avg_latency_ms:<12.2f}"
                table += f"{exp.metrics.success_rate:<12.2f}"
                table += f"{exp.metrics.avg_confidence:<12.2f}\n"

        table += "="*80 + "\n"
        return table
