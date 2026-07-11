"""
core/orchestrator.py
Main CSV QA Agent orchestration - Windows compatible, fixed keyword matching.
"""
import os
import re
import json
import time
from typing import Dict, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from core.models import AgentResponse
from core.sandbox import SecureSandbox

class CSVQAAgent:
    """Orchestrator that routes queries to rule-based or LLM agents."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path)
        self.schema = self._get_schema()
        self.sandbox = SecureSandbox(self.df, timeout=5.0)
        self.llm_available = False
        self._init_llm()

    def _get_schema(self):
        return {
            "columns": [
                {"name": c, "dtype": str(self.df[c].dtype), "unique": int(self.df[c].nunique())}
                for c in self.df.columns
            ],
            "shape": [int(self.df.shape[0]), int(self.df.shape[1])]
        }

    def _init_llm(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return
        try:
            os.environ["OPENAI_TIMEOUT"] = "30"
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.models.list()
            self.llm_available = True
            self.openai_client = client
        except Exception:
            self.llm_available = False

    def _generate_rule_code(self, question: str) -> str:
        """Generate rule-based code with robust keyword matching."""
        q = question.lower()

        # Helper: check if any word from a list is in the query
        def has_any(words): return any(w in q for w in words)

        # Helper: check if ALL words from a list are in the query
        def has_all(words): return all(w in q for w in words)

        # TOTAL REVENUE / TOTAL SALES
        if has_any(['total revenue', 'total sales', 'sum of revenue', 'all revenue', 'overall revenue', 'revenue total']):
            return """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Total revenue: ${total:,.2f}', 'data': {'total': round(float(total),2)}, 'viz_type': 'number'}"""

        # TOP REGION / HIGHEST REGION / BEST REGION
        # Trigger: "region" + any of [top, highest, best, most, max]
        if has_all(['region']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max', 'biggest', 'largest']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        # TOP PRODUCT / HIGHEST PRODUCT / BEST PRODUCT
        if has_all(['product']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max', 'biggest', 'largest']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        # TOP CATEGORY
        if has_all(['category']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        # TOP SALES REP
        if has_any(['sales rep', 'rep', 'representative']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('sales_rep')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        # AVERAGE BY SEGMENT
        if has_any(['average', 'mean', 'avg']) and has_any(['segment', 'customer_segment']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('customer_segment')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by segment: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        # AVERAGE BY REGION
        if has_any(['average', 'mean', 'avg']) and has_all(['region']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('region')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by region: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        # AVERAGE BY PRODUCT
        if has_any(['average', 'mean', 'avg']) and has_all(['product']):
            return """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('product')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by product: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        # MoM GROWTH / MONTHLY GROWTH
        if has_any(['growth', 'mom', 'month over month', 'monthly', 'month to month', 'trend']):
            return """df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
df['revenue'] = df['units'] * df['unit_price']
monthly = df.groupby('month')['revenue'].sum()
growth = monthly.pct_change() * 100
g = {str(k): round(float(v),1) for k,v in growth.dropna().items()}
result = {'answer': 'Month-over-month growth: ' + ', '.join([f"{k}={v}%" for k,v in g.items()]), 'data': g, 'viz_type': 'line'}"""

        # COMPARE / PIVOT / BREAKDOWN
        if has_any(['compare', 'vs', 'versus', 'pivot', 'breakdown', 'cross', 'by region and', 'by segment and']):
            return """df['revenue'] = df['units'] * df['unit_price']
pivot = df.pivot_table(values='revenue', index='region', columns='customer_segment', aggfunc='sum', fill_value=0)
result = {'answer': 'Enterprise vs SMB sales by region', 'data': {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in pivot.to_dict().items()}, 'viz_type': 'table'}"""

        # CORRELATION
        if has_any(['correlation', 'correlate', 'relationship between']):
            return """corr = df['units'].corr(df['unit_price'])
result = {'answer': f'Correlation between units and unit_price: {corr:.4f}', 'data': {'correlation': round(float(corr),4)}, 'viz_type': 'number'}"""

        # COUNT / UNIQUE
        if has_any(['how many', 'count', 'unique', 'number of', 'distinct']):
            return """counts = df.groupby('sales_rep')['customer_segment'].nunique()
result = {'answer': 'Unique customer segments handled by each rep', 'data': {str(k): int(v) for k,v in counts.to_dict().items()}, 'viz_type': 'table'}"""

        # DEFAULT FALLBACK
        return """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Dataset: {len(df)} rows. Total revenue: ${total:,.2f}. Try: total revenue, top region, avg by segment, growth, compare, correlation', 'data': {'total': round(float(total),2)}, 'viz_type': 'number'}"""

    def _llm_generate(self, question: str) -> str:
        if not self.llm_available:
            raise Exception("LLM not available")

        schema_str = json.dumps(self.schema, default=str)
        prompt = f"""Generate Python/Pandas code to answer: "{question}"

Dataset: {schema_str}
Rules:
1. Write ONLY Python code. No markdown, no explanations.
2. df is pre-loaded. Store answer in `result` = {{'answer': str, 'data': any, 'viz_type': 'table'|'bar'|'line'|'number'|None}}
3. Compute actual values. No hardcoded numbers.
4. No imports, no file I/O.
5. Calculate revenue as df['revenue'] = df['units'] * df['unit_price'] first, then use df['revenue'] for groupby operations.

Example: result = {{'answer': 'Total: $45,230', 'data': {{'total': 45230}}, 'viz_type': 'number'}}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        code = response.choices[0].message.content.strip()
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        return code

    def answer(self, question: str) -> AgentResponse:
        start = time.time()
        trace = []

        if self.llm_available:
            try:
                trace.append("Attempting LLM generation...")
                code = self._llm_generate(question)
                trace.append("LLM code generated")

                success, result, msg = self.sandbox.run(code)
                if success:
                    trace.append("LLM code executed successfully")
                    latency = round((time.time() - start) * 1000, 2)
                    return AgentResponse(
                        answer=result.get('answer', ''),
                        confidence=0.95,
                        viz_type=result.get('viz_type', ''),
                        data=result.get('data'),
                        execution_trace=trace,
                        code=code,
                        mode="llm",
                        latency_ms=latency
                    )
                else:
                    trace.append(f"LLM execution failed: {msg}")
            except Exception as e:
                trace.append(f"LLM failed: {e}")

        trace.append("Using rule-based fallback...")
        code = self._generate_rule_code(question)
        trace.append("Rule-based code generated")

        success, result, msg = self.sandbox.run(code)
        latency = round((time.time() - start) * 1000, 2)

        if success:
            trace.append("Rule-based code executed successfully")
            return AgentResponse(
                answer=result.get('answer', ''),
                confidence=0.90,
                viz_type=result.get('viz_type', ''),
                data=result.get('data'),
                execution_trace=trace,
                code=code,
                mode="rule-based",
                latency_ms=latency
            )
        else:
            trace.append(f"Rule-based execution failed: {msg}")
            return AgentResponse(
                answer=f"Error: {msg}",
                confidence=0.0,
                viz_type="error",
                data=None,
                execution_trace=trace,
                code=code,
                mode="error",
                latency_ms=latency
            )
