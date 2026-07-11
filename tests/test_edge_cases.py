"""
csv_qa_agent/tests/test_edge_cases.py
Edge case tests for robustness.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import SecureSandbox


class TestEmptyCSV:
    """Test behavior with empty or minimal CSVs."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['order_id', 'date', 'region', 'product', 'units', 'unit_price'])
        sandbox = SecureSandbox(df)
        code = "result = {'answer': 'Empty', 'data': {'rows': len(df)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True
        assert result['data']['rows'] == 0

    def test_single_row(self):
        df = pd.DataFrame({
            'order_id': [1],
            'date': ['2024-01-01'],
            'region': ['North'],
            'product': ['Alpha'],
            'units': [10],
            'unit_price': [100.0]
        })
        sandbox = SecureSandbox(df)
        code = "result = {'answer': 'Single', 'data': {'rows': len(df)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True
        assert result['data']['rows'] == 1


class TestMissingColumns:
    """Test behavior when expected columns are missing."""

    def test_missing_revenue_columns(self):
        df = pd.DataFrame({
            'name': ['A', 'B'],
            'value': [1, 2]
        })
        sandbox = SecureSandbox(df)
        code = "df['revenue'] = df['units'] * df['unit_price']\nresult = {'answer': 'test'}"
        success, result, error = sandbox.execute(code)
        assert success is False
        assert "KeyError" in error or "units" in error

    def test_adapts_to_available_columns(self):
        df = pd.DataFrame({
            'region': ['North', 'South'],
            'sales': [100, 200]
        })
        sandbox = SecureSandbox(df)
        code = "total = df['sales'].sum()\nresult = {'answer': f'Total: {total}', 'data': {'total': float(total)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True
        assert result['data']['total'] == 300


class TestInvalidData:
    """Test behavior with malformed data."""

    def test_nan_values(self):
        df = pd.DataFrame({
            'units': [10, None, 30],
            'unit_price': [100.0, 200.0, None]
        })
        sandbox = SecureSandbox(df)
        code = "total = df['units'].sum()\nresult = {'answer': f'Total units: {total}', 'data': {'total': float(total)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True

    def test_string_in_numeric_column(self):
        df = pd.DataFrame({
            'units': ['10', '20', 'abc'],
            'unit_price': [100.0, 200.0, 300.0]
        })
        sandbox = SecureSandbox(df)
        code = "df['units'] = pd.to_numeric(df['units'], errors='coerce')\ntotal = df['units'].sum()\nresult = {'answer': f'Total: {total}', 'data': {'total': float(total)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True


class TestTimeout:
    """Test execution timeout."""

    def test_infinite_loop_timeout(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        sandbox = SecureSandbox(df)
        code = "while True: pass\nresult = {'answer': 'never'}"
        success, result, error = sandbox.execute(code)
        assert success is False
        assert "timed out" in error.lower() or "Timeout" in error


class TestLargeCSV:
    """Test performance with larger datasets."""

    def test_1000_rows(self):
        df = pd.DataFrame({
            'order_id': range(1000),
            'units': [10] * 1000,
            'unit_price': [100.0] * 1000
        })
        sandbox = SecureSandbox(df)
        code = "total = (df['units'] * df['unit_price']).sum()\nresult = {'answer': f'Total: {total}', 'data': {'total': float(total)}, 'viz_type': 'number'}"
        success, result, _ = sandbox.execute(code)
        assert success is True
        assert result['data']['total'] == 1000000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
