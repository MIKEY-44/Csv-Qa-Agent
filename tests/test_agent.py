"""
csv_qa_agent/tests/test_agent.py
Comprehensive test suite for the CSV QA Agent.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import CSVQAAgent, SecureSandbox, CriticAgent
from core.models import AgentResponse


@pytest.fixture
def agent():
    csv_path = str(PROJECT_ROOT / "data" / "sales.csv")
    return CSVQAAgent(csv_path)


@pytest.fixture
def sample_df():
    return pd.read_csv(str(PROJECT_ROOT / "data" / "sales.csv"))


class TestSecureSandbox:
    """Test the secure code execution environment."""

    def test_valid_code_execution(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = """result = {'answer': 'Test', 'data': {'count': len(df)}, 'viz_type': 'number'}"""
        success, result, output = sandbox.execute(code)
        assert success is True
        assert result['data']['count'] == 50

    def test_blocks_imports(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "import os\nresult = {'answer': 'bad'}"
        success, result, error = sandbox.execute(code)
        assert success is False
        assert "Imports not allowed" in error

    def test_blocks_open_function(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "open('test.txt')\nresult = {'answer': 'bad'}"
        success, result, error = sandbox.execute(code)
        assert success is False
        assert "not allowed" in error

    def test_blocks_eval(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "eval('1+1')\nresult = {'answer': 'bad'}"
        success, result, error = sandbox.execute(code)
        assert success is False

    def test_requires_result_variable(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "x = len(df)"
        success, result, error = sandbox.execute(code)
        assert success is False
        assert "result" in error

    def test_timeout_not_needed_for_small_code(self, sample_df):
        sandbox = SecureSandbox(sample_df)
        code = "result = {'answer': 'fast', 'data': {}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True


class TestCriticAgent:
    """Test the self-reflection Critic Agent."""

    def test_detects_missing_result(self):
        critic = CriticAgent()
        review = critic.review("x = 5", "test", {})
        assert review.passed is False
        assert any("result" in issue for issue in review.issues)

    def test_detects_unsafe_code(self):
        critic = CriticAgent()
        review = critic.review("open('file.txt')\nresult = {}", "test", {})
        assert review.passed is False
        assert any("unsafe" in issue.lower() for issue in review.issues)

    def test_passes_safe_code(self):
        critic = CriticAgent()
        code = "result = {'answer': 'safe', 'data': {}, 'viz_type': 'number'}"
        review = critic.review(code, "test", {})
        assert review.passed is True
        assert review.safety_score == 100.0

    def test_suggest_fix_includes_error(self):
        critic = CriticAgent()
        fix = critic.suggest_fix("bad code", "SyntaxError", "test question")
        assert "SyntaxError" in fix
        assert "test question" in fix


class TestRuleBasedAgent:
    """Test rule-based query patterns."""

    def test_total_revenue(self, agent):
        resp = agent.answer("What is the total revenue?")
        assert "$275,743.18" in resp.answer
        assert resp.viz_type == "number"
        assert resp.data is not None
        assert abs(resp.data['total'] - 275743.18) < 0.01

    def test_top_region(self, agent):
        resp = agent.answer("Which region had the highest revenue?")
        assert "North" in resp.answer
        assert resp.viz_type == "bar"
        assert "North" in resp.data

    def test_top_product(self, agent):
        resp = agent.answer("Which product made the most revenue?")
        assert "Epsilon AI" in resp.answer
        assert resp.viz_type == "bar"

    def test_average_by_segment(self, agent):
        resp = agent.answer("What is the average revenue by customer segment?")
        assert "Enterprise" in resp.answer
        assert resp.viz_type == "table"
        assert abs(resp.data['Enterprise'] - 6066.56) < 0.01

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
        assert "SMB" in resp.data

    def test_default_response(self, agent):
        resp = agent.answer("something random")
        assert len(resp.answer) > 0
        assert resp.viz_type in ["number", "table", "bar", "line"]


class TestEvaluationSuite:
    """Test the benchmark evaluation."""

    def test_benchmark_runs(self, agent):
        benchmark = [
            {"question": "What is the total revenue?", "expected_answer_contains": "275,743", "expected_viz_type": "number"},
            {"question": "Which region had the highest revenue?", "expected_answer_contains": "North", "expected_viz_type": "bar"},
        ]
        results = agent.evaluate(benchmark)
        assert results['total_questions'] == 2
        assert results['accuracy'] == 100.0
        assert results['success_rate'] == 100.0
        assert results['avg_latency'] >= 0
        assert len(results['results']) == 2

    def test_benchmark_detects_failures(self, agent):
        benchmark = [
            {"question": "What is the total revenue?", "expected_answer_contains": "WRONG_ANSWER"},
        ]
        results = agent.evaluate(benchmark)
        assert results['correct'] == 0
        assert results['accuracy'] == 0.0


class TestResponseStructure:
    """Test that responses have correct structure."""

    def test_response_has_all_fields(self, agent):
        resp = agent.answer("Total revenue?")
        assert isinstance(resp.answer, str)
        assert isinstance(resp.confidence, float)
        assert isinstance(resp.execution_trace, list)
        assert isinstance(resp.metadata, dict)
        assert resp.code is not None
        assert resp.data is not None
        assert resp.viz_type is not None

    def test_trace_contains_steps(self, agent):
        resp = agent.answer("Total revenue?")
        assert len(resp.execution_trace) > 0
        assert all('step' in trace for trace in resp.execution_trace)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
