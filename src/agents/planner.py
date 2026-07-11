"""
src/agents/planner.py
Planner Agent: Analyzes queries and determines execution strategy.
"""
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ExecutionPlan:
    strategy: str
    confidence: float
    reasoning: str
    needs_aggregation: bool = False
    needs_grouping: bool = False
    needs_visualization: bool = False
    suggested_viz_type: str = ""


class PlannerAgent:
    """Analyzes natural language queries and plans execution strategy."""

    STRATEGIES = {
        'aggregation': {
            'keywords': ['total', 'sum', 'all', 'overall', 'combined', 'aggregate', 'cumulative', 'grand'],
            'needs_aggregation': True,
            'needs_grouping': False,
            'suggested_viz': 'number'
        },
        'top_k': {
            'keywords': ['highest', 'maximum', 'top', 'most', 'best', 'largest', 'biggest', 'worst', 'lowest', 'minimum'],
            'needs_aggregation': True,
            'needs_grouping': True,
            'suggested_viz': 'bar'
        },
        'average': {
            'keywords': ['average', 'mean', 'avg', 'median', 'typical', 'normal'],
            'needs_aggregation': True,
            'needs_grouping': True,
            'suggested_viz': 'table'
        },
        'correlation': {
            'keywords': ['correlation', 'correlate', 'relationship', 'related', 'association'],
            'needs_aggregation': True,
            'needs_grouping': False,
            'suggested_viz': 'number'
        },
        'growth': {
            'keywords': ['growth', 'trend', 'over time', 'monthly', 'weekly', 'yearly', 'progression', 'momentum'],
            'needs_aggregation': True,
            'needs_grouping': True,
            'suggested_viz': 'line'
        },
        'comparison': {
            'keywords': ['compare', 'vs', 'versus', 'difference', 'between', 'against', 'relative', 'pivot', 'breakdown'],
            'needs_aggregation': True,
            'needs_grouping': True,
            'suggested_viz': 'table'
        },
        'count': {
            'keywords': ['how many', 'count', 'unique', 'distinct', 'number of', 'frequency'],
            'needs_aggregation': True,
            'needs_grouping': True,
            'suggested_viz': 'table'
        },
        'filter': {
            'keywords': ['filter', 'only', 'where', 'greater than', 'less than', 'above', 'below'],
            'needs_aggregation': False,
            'needs_grouping': False,
            'suggested_viz': 'table'
        },
        'general': {
            'keywords': [],
            'needs_aggregation': False,
            'needs_grouping': False,
            'suggested_viz': 'number'
        }
    }

    def plan(self, question: str) -> ExecutionPlan:
        q = question.lower()
        best_strategy = 'general'
        best_score = 0

        for strategy, config in self.STRATEGIES.items():
            if strategy == 'general':
                continue
            score = sum(1 for kw in config['keywords'] if kw in q)
            if score > best_score:
                best_score = score
                best_strategy = strategy

        config = self.STRATEGIES[best_strategy]
        confidence = min(0.5 + best_score * 0.15, 0.98)

        return ExecutionPlan(
            strategy=best_strategy,
            confidence=confidence,
            reasoning=f"Detected '{best_strategy}' pattern with {best_score} keyword matches",
            needs_aggregation=config['needs_aggregation'],
            needs_grouping=config['needs_grouping'],
            needs_visualization=True,
            suggested_viz_type=config['suggested_viz']
        )

    def extract_entities(self, question: str) -> Dict:
        q = question.lower()
        entities = {'columns': [], 'values': [], 'operations': []}

        common_cols = ['region', 'product', 'category', 'segment', 'customer', 'sales', 'revenue', 'price', 'units']
        for col in common_cols:
            if col in q:
                entities['columns'].append(col)

        ops = ['sum', 'average', 'mean', 'count', 'max', 'min', 'correlation']
        for op in ops:
            if op in q:
                entities['operations'].append(op)

        return entities
