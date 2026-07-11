"""
csv_qa_agent/agents/planner.py
Planner Agent: Determines execution strategy based on query type.
"""
from typing import Dict, List


class PlannerAgent:
    """
    Analyzes the user query and determines the best execution strategy.
    """

    QUERY_PATTERNS = {
        'total': ['total', 'sum', 'all sales', 'overall', 'combined'],
        'top': ['highest', 'maximum', 'top', 'most', 'best', 'largest'],
        'average': ['average', 'mean', 'avg'],
        'correlation': ['correlation', 'correlate', 'relationship'],
        'growth': ['growth', 'month', 'mom', 'trend', 'over time'],
        'compare': ['compare', 'vs', 'versus', 'pivot', 'breakdown'],
        'count': ['how many', 'count', 'unique', 'number of', 'distinct']
    }

    def plan(self, question: str) -> Dict:
        """
        Analyze question and return execution plan.

        Returns:
            Dict with 'strategy', 'confidence', 'reasoning'
        """
        q = question.lower()

        for strategy, keywords in self.QUERY_PATTERNS.items():
            if any(kw in q for kw in keywords):
                return {
                    'strategy': strategy,
                    'confidence': 0.95,
                    'reasoning': f"Matched '{strategy}' pattern"
                }

        return {
            'strategy': 'general',
            'confidence': 0.7,
            'reasoning': 'No specific pattern matched, using general approach'
        }
