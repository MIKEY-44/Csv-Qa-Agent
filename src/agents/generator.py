"""
src/agents/generator.py
Code Generator: Produces safe Python/Pandas code.
"""
import json
from typing import Dict
import pandas as pd


class CodeGenerator:
    """Generates executable Python code for CSV queries."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.schema = self._get_schema()

    def _get_schema(self) -> Dict:
        cols = []
        for c in self.df.columns:
            sample = self.df[c].dropna().head(3).tolist()
            cols.append({
                "name": c,
                "dtype": str(self.df[c].dtype),
                "unique": int(self.df[c].nunique()),
                "sample": [str(s) for s in sample]
            })
        return {"columns": cols, "shape": list(self.df.shape)}

    def generate(self, question: str, strategy: str, entities: Dict = None) -> str:
        """Generate code based on strategy."""
        generators = {
            'aggregation': self._gen_aggregation,
            'top_k': self._gen_top_k,
            'average': self._gen_average,
            'correlation': self._gen_correlation,
            'growth': self._gen_growth,
            'comparison': self._gen_comparison,
            'count': self._gen_count,
            'filter': self._gen_filter,
            'general': self._gen_general
        }

        gen_func = generators.get(strategy, self._gen_general)
        return gen_func(question, entities)

    def _gen_aggregation(self, q: str, entities: Dict) -> str:
        return """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Total revenue: ${total:,.2f}', 'data': {'total': round(total,2)}, 'viz_type': 'number'}"""

    def _gen_top_k(self, q: str, entities: Dict) -> str:
        group_col = 'region' if 'region' in q else 'product' if 'product' in q else 'sales_rep' if 'rep' in q else 'region'
        return f"""df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('{group_col}')['revenue'].sum().sort_values(ascending=False)
result = {{'answer': f'{{grouped.index[0]}} has the highest revenue at ${{grouped.iloc[0]:,.2f}}', 'data': {{str(k): round(v, 2) for k, v in grouped.to_dict().items()}}, 'viz_type': 'bar'}}"""

    def _gen_average(self, q: str, entities: Dict) -> str:
        group_col = 'customer_segment' if 'segment' in q else 'region' if 'region' in q else 'customer_segment'
        return f"""df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('{group_col}')['revenue'].mean().round(2)
result = {{'answer': 'Average revenue by {group_col}: ' + ', '.join([f"{{k}}=${{v}}" for k,v in grouped.items()]), 'data': {{str(k): float(v) for k,v in grouped.to_dict().items()}}, 'viz_type': 'table'}}"""

    def _gen_correlation(self, q: str, entities: Dict) -> str:
        return """corr = df['units'].corr(df['unit_price'])
result = {'answer': f'Correlation between units and unit_price: {corr:.4f}', 'data': {'correlation': round(corr,4)}, 'viz_type': 'number'}"""

    def _gen_growth(self, q: str, entities: Dict) -> str:
        return """df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
df['revenue'] = df['units'] * df['unit_price']
monthly = df.groupby('month')['revenue'].sum()
growth = monthly.pct_change() * 100
g = {str(k): round(v,1) for k,v in growth.dropna().items()}
result = {'answer': 'Month-over-month growth: ' + ', '.join([f"{k}={v}%" for k,v in g.items()]), 'data': g, 'viz_type': 'line'}"""

    def _gen_comparison(self, q: str, entities: Dict) -> str:
        return """df['revenue'] = df['units'] * df['unit_price']
pivot = df.pivot_table(values='revenue', index='region', columns='customer_segment', aggfunc='sum', fill_value=0)
result = {'answer': 'Revenue breakdown by region and customer segment', 'data': {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in pivot.to_dict().items()}, 'viz_type': 'table'}"""

    def _gen_count(self, q: str, entities: Dict) -> str:
        return """counts = df.groupby('sales_rep')['customer_segment'].nunique()
result = {'answer': 'Unique customer segments handled by each rep', 'data': {str(k): int(v) for k,v in counts.to_dict().items()}, 'viz_type': 'table'}"""

    def _gen_filter(self, q: str, entities: Dict) -> str:
        return """df['revenue'] = df['units'] * df['unit_price']
filtered = df[df['revenue'] > 1000]
result = {'answer': f'{len(filtered)} orders above $1000', 'data': {'count': len(filtered)}, 'viz_type': 'number'}"""

    def _gen_general(self, q: str, entities: Dict) -> str:
        return """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Dataset: {len(df)} rows. Total revenue: ${total:,.2f}.', 'data': {'total': round(total,2), 'rows': len(df)}, 'viz_type': 'number'}"""

    def generate_llm_prompt(self, question: str) -> str:
        """Build prompt for LLM code generation."""
        schema = json.dumps(self.schema, default=str, indent=2)
        return f"""Generate Python/Pandas code to answer this question about a CSV dataset:

Question: "{question}"

Dataset Schema:
{schema}

Rules:
1. Write ONLY Python code. No markdown, no explanations, no backticks.
2. The dataframe is pre-loaded as `df`. Store the final answer in a variable called `result`.
3. `result` must be a dictionary: {{'answer': str, 'data': any, 'viz_type': 'table'|'bar'|'line'|'number'|None}}
4. Compute actual values from the data. No hardcoded numbers.
5. No imports, no file I/O, no external API calls.
6. Calculate revenue as df['revenue'] = df['units'] * df['unit_price'] first if those columns exist.

Example:
result = {{'answer': 'Total revenue is $45,230.50', 'data': {{'total': 45230.50}}, 'viz_type': 'number'}}"""
