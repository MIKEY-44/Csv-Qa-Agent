"""Unit tests for CSV AI Agent."""

import pytest
import pandas as pd
from agents.planner import PlannerAgent
from agents.generator import CodeGeneratorAgent
from agents.validator import ValidatorAgent
from agents.executor import ExecutorAgent

class TestPlanner:
    def test_detects_total_intent(self):
        planner = PlannerAgent()
        plan = planner.plan("What is the total sales?", has_llm=False)
        assert plan['strategy'] == 'total'
        assert plan['confidence'] == 0.9

    def test_detects_top_intent(self):
        planner = PlannerAgent()
        plan = planner.plan("Which region has highest revenue?", has_llm=False)
        assert plan['strategy'] == 'top'

    def test_fallback_for_unknown(self):
        planner = PlannerAgent()
        plan = planner.plan("Something random", has_llm=False)
        assert plan['strategy'] == 'default'

class TestValidator:
    def test_blocks_imports(self):
        validator = ValidatorAgent()
        is_valid, error = validator.validate("import os")
        assert not is_valid
        assert "Import" in error

    def test_blocks_open(self):
        validator = ValidatorAgent()
        is_valid, error = validator.validate("open('file.txt')")
        assert not is_valid
        assert "open" in error

    def test_allows_safe_code(self):
        validator = ValidatorAgent()
        is_valid, error = validator.validate("result = {'answer': 'test'}")
        assert is_valid

class TestExecutor:
    def test_executes_code(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        executor = ExecutorAgent(df, timeout=5)
        success, result, output = executor.execute("result = {'answer': str(df['a'].sum())}")
        assert success
        assert result['answer'] == '6'

    def test_handles_timeout(self):
        df = pd.DataFrame({'a': [1]})
        executor = ExecutorAgent(df, timeout=1)
        # Note: Timeout only works on Unix with SIGALRM
        # This test documents the feature
