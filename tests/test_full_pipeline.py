"""
tests/test_full_pipeline.py
Full pipeline tests for the multi-agent system.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import CSVQAAgent
from src.agents.planner import PlannerAgent
from src.agents.critic import CriticAgent
from src.sandbox.executor import SecureSandbox
import pandas as pd
import pytest


class TestPlannerAgent:
    """Test query planning."""

    def test_aggregation_detection(self):
        planner = PlannerAgent()
        plan = planner.plan("What is the total revenue?")
        assert plan.strategy == 'aggregation'
        assert plan.confidence > 0.9

    def test_top_k_detection(self):
        planner = PlannerAgent()
        plan = planner.plan("Which region has the highest revenue?")
        assert plan.strategy == 'top_k'

    def test_growth_detection(self):
        planner = PlannerAgent()
        plan = planner.plan("What is the month over month growth?")
        assert plan.strategy == 'growth'

    def test_comparison_detection(self):
        planner = PlannerAgent()
        plan = planner.plan("Compare Enterprise vs SMB sales")
        assert plan.strategy == 'comparison'

    def test_entity_extraction(self):
        planner = PlannerAgent()
        entities = planner.extract_entities("Average revenue by region")
        assert 'region' in entities['columns']
        assert 'average' in entities['operations']


class TestCriticAgent:
    """Test self-reflection."""

    def test_detects_unsafe_code(self):
        critic = CriticAgent()
        review = critic.review("open('file.txt')\nresult = {}", "test", {})
        assert review.passed is False
        assert review.safety_score == 0.0
        assert any('SAFETY' in issue for issue in review.issues)

    def test_detects_missing_result(self):
        critic = CriticAgent()
        review = critic.review("x = 5", "test", {})
        assert review.passed is False
        assert review.correctness_score < 100

    def test_passes_safe_code(self):
        critic = CriticAgent()
        code = "result = {'answer': 'safe', 'data': {}, 'viz_type': 'number'}"
        review = critic.review(code, "test", {})
        assert review.passed is True
        assert review.safety_score == 100.0
        assert review.correctness_score == 100.0

    def test_overall_score_calculation(self):
        critic = CriticAgent()
        review = critic.review(
            "result = {'answer': 'test', 'data': {}, 'viz_type': 'number'}",
            "test", {}
        )
        assert 0 <= review.overall_score <= 100
        assert review.overall_score == 100.0


class TestSecureSandbox:
    """Test sandbox security."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'units': [10, 20, 30],
            'unit_price': [100.0, 200.0, 300.0]
        })

    def test_valid_execution(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "result = {'answer': 'Test', 'data': {'count': len(df)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True
        assert result['data']['count'] == 3

    def test_blocks_imports(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "import os\nresult = {}"
        success, _, error = sandbox.execute(code)
        assert success is False
        assert "Imports not allowed" in error

    def test_blocks_eval(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "eval('1+1')\nresult = {}"
        success, _, error = sandbox.execute(code)
        assert success is False

    def test_blocks_open(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "open('test.txt')\nresult = {}"
        success, _, error = sandbox.execute(code)
        assert success is False

    def test_requires_result(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "x = len(df)"
        success, _, error = sandbox.execute(code)
        assert success is False
        assert "result" in error

    def test_timeout(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "while True: pass\nresult = {}"
        success, _, error = sandbox.execute(code)
        assert success is False
        assert "timed out" in error.lower()


class TestFullPipeline:
    """Test end-to-end pipeline."""

    @pytest.fixture
    def agent(self):
        return CSVQAAgent(str(PROJECT_ROOT / "data" / "sales.csv"))

    def test_total_revenue(self, agent):
        resp = agent.answer("What is the total revenue?")
        assert "$275,743.18" in resp.answer
        assert resp.viz_type == "number"
        assert resp.metadata['critic_safety'] == 100.0

    def test_top_region(self, agent):
        resp = agent.answer("Which region had the highest revenue?")
        assert "North" in resp.answer
        assert resp.viz_type == "bar"

    def test_top_product(self, agent):
        resp = agent.answer("Which product made the most revenue?")
        assert "Epsilon AI" in resp.answer
        assert resp.viz_type == "bar"

    def test_average_by_segment(self, agent):
        resp = agent.answer("What is the average revenue by customer segment?")
        assert "Enterprise" in resp.answer
        assert resp.viz_type == "table"

    def test_month_over_month_growth(self, agent):
        resp = agent.answer("What is the month over month growth?")
        assert "2024-02" in resp.answer
        assert resp.viz_type == "line"

    def test_correlation(self, agent):
        resp = agent.answer("What is the correlation between units and unit price?")
        assert "-0.76" in resp.answer
        assert resp.viz_type == "number"

    def test_compare_segments(self, agent):
        resp = agent.answer("Compare Enterprise vs SMB sales")
        assert resp.viz_type == "table"
        assert "Enterprise" in resp.data

    def test_execution_trace(self, agent):
        resp = agent.answer("Total revenue?")
        assert len(resp.execution_trace) >= 4  # planner, generator, critic, executor
        steps = [t['step'] for t in resp.execution_trace]
        assert 'planner' in steps
        assert 'critic' in steps
        assert 'executor' in steps


class TestEvaluation:
    """Test benchmark evaluation."""

    def test_benchmark_runs(self):
        agent = CSVQAAgent(str(PROJECT_ROOT / "data" / "sales.csv"))
        benchmark = [
            {"question": "What is the total revenue?", "expected_answer_contains": "275,743"},
            {"question": "Which region had the highest revenue?", "expected_answer_contains": "North"},
        ]
        results = agent.evaluate(benchmark)
        assert results['total_questions'] == 2
        assert results['accuracy'] == 100.0
        assert results['success_rate'] == 100.0
        assert results['avg_latency'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
