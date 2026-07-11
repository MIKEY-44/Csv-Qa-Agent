"""Visualization Service - Auto-generates charts."""

import json
from typing import Optional, Dict, Any

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class VisualizationService:
    """Generate Plotly charts from data."""

    def generate(self, data: Any, viz_type: str, title: str = "") -> Optional[Dict]:
        if not HAS_PLOTLY or not data or not viz_type:
            return None

        try:
            if viz_type == "bar":
                return self._bar_chart(data, title)
            elif viz_type == "line":
                return self._line_chart(data, title)
            elif viz_type == "pie":
                return self._pie_chart(data, title)
            elif viz_type == "number":
                return self._number_card(data, title)
            else:
                return None
        except Exception:
            return None

    def _bar_chart(self, data: Dict, title: str) -> Dict:
        import pandas as pd
        df = pd.DataFrame(list(data.items()), columns=["Category", "Value"])
        fig = px.bar(df, x="Category", y="Value", title=title or "Chart")
        return json.loads(fig.to_json())

    def _line_chart(self, data: Dict, title: str) -> Dict:
        import pandas as pd
        df = pd.DataFrame(list(data.items()), columns=["X", "Y"])
        fig = px.line(df, x="X", y="Y", title=title or "Trend")
        return json.loads(fig.to_json())

    def _pie_chart(self, data: Dict, title: str) -> Dict:
        fig = px.pie(values=list(data.values()), names=list(data.keys()), title=title or "Distribution")
        return json.loads(fig.to_json())

    def _number_card(self, data: Dict, title: str) -> Dict:
        if isinstance(data, dict) and len(data) == 1:
            value = list(data.values())[0]
            fig = go.Figure(go.Indicator(mode="number", value=value, title={"text": title}))
            return json.loads(fig.to_json())
        return None
