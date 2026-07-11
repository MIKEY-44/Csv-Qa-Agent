"""
src/agents/visualizer.py
Visualizer Agent: Automatically selects and renders visualizations.
"""
from typing import Dict, Any


class VisualizerAgent:
    """
    Automatically selects visualization type based on data characteristics.
    """

    VIZ_TYPES = {
        'number': {
            'description': 'Single numeric value',
            'conditions': lambda data: isinstance(data, dict) and len(data) == 1
        },
        'bar': {
            'description': 'Categorical comparison',
            'conditions': lambda data: isinstance(data, dict) and len(data) > 1 and len(data) <= 15
        },
        'line': {
            'description': 'Time series or sequential data',
            'conditions': lambda data: isinstance(data, dict) and any('-' in str(k) for k in data.keys())
        },
        'table': {
            'description': 'Multi-dimensional data',
            'conditions': lambda data: isinstance(data, dict) and any(isinstance(v, dict) for v in data.values())
        },
        'pie': {
            'description': 'Part-to-whole relationship',
            'conditions': lambda data: isinstance(data, dict) and len(data) <= 8
        }
    }

    def select_viz_type(self, data: Any, suggested: str = None) -> str:
        """Select best visualization type for the data."""
        if suggested and suggested in self.VIZ_TYPES:
            return suggested

        if not isinstance(data, dict):
            return 'number'

        # Check each viz type in priority order
        for viz_type, config in self.VIZ_TYPES.items():
            if config['conditions'](data):
                return viz_type

        return 'table'

    def render_config(self, viz_type: str, data: Any) -> Dict:
        """Generate rendering configuration for the visualization."""
        configs = {
            'number': {
                'title': 'Key Metric',
                'format': 'currency' if any(isinstance(v, (int, float)) and v > 100 for v in (data.values() if isinstance(data, dict) else [])) else 'number'
            },
            'bar': {
                'title': 'Comparison',
                'orientation': 'vertical',
                'sort': True
            },
            'line': {
                'title': 'Trend Over Time',
                'smooth': True,
                'show_points': True
            },
            'table': {
                'title': 'Detailed Breakdown',
                'sortable': True,
                'pagination': len(data) > 10 if isinstance(data, dict) else False
            },
            'pie': {
                'title': 'Distribution',
                'show_legend': True,
                'show_percentages': True
            }
        }

        return configs.get(viz_type, {'title': 'Data'})
