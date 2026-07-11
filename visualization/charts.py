"""
csv_qa_agent/visualization/charts.py
Auto-generate Plotly charts based on query intent and data.
"""
import base64
import io
from typing import Optional, Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ChartGenerator:
    """Automatically generates appropriate charts for CSV data."""

    CHART_TYPES = {
        'bar': 'Bar chart for comparing categories',
        'line': 'Line chart for trends over time',
        'pie': 'Pie chart for proportions',
        'scatter': 'Scatter plot for correlations',
        'histogram': 'Histogram for distributions',
        'heatmap': 'Heatmap for correlations',
        'box': 'Box plot for distributions by category'
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    def suggest_chart_type(self, query: str) -> str:
        """Suggest the best chart type based on query intent."""
        query_lower = query.lower()

        if any(w in query_lower for w in ['trend', 'over time', 'monthly', 'yearly', 'daily']):
            return 'line'
        if any(w in query_lower for w in ['proportion', 'percentage', 'share', 'distribution']):
            return 'pie'
        if any(w in query_lower for w in ['correlation', 'relationship', 'vs', 'versus']):
            return 'scatter'
        if any(w in query_lower for w in ['distribution', 'frequency', 'histogram']):
            return 'histogram'
        if any(w in query_lower for w in ['compare', 'comparison', 'by category', 'group']):
            return 'bar'
        if any(w in query_lower for w in ['heatmap', 'correlation matrix']):
            return 'heatmap'
        if any(w in query_lower for w in ['box', 'outlier', 'quartile']):
            return 'box'

        # Default based on data
        if len(self.numeric_cols) >= 2:
            return 'scatter'
        return 'bar'

    def generate(
        self,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        title: str = "Chart"
    ) -> str:
        """Generate a chart and return as base64 encoded HTML."""
        fig = None

        if chart_type == 'bar':
            fig = self._create_bar_chart(x_col, y_col, color_col, title)
        elif chart_type == 'line':
            fig = self._create_line_chart(x_col, y_col, color_col, title)
        elif chart_type == 'pie':
            fig = self._create_pie_chart(x_col, y_col, title)
        elif chart_type == 'scatter':
            fig = self._create_scatter_chart(x_col, y_col, color_col, title)
        elif chart_type == 'histogram':
            fig = self._create_histogram(x_col, title)
        elif chart_type == 'heatmap':
            fig = self._create_heatmap(title)
        elif chart_type == 'box':
            fig = self._create_box_plot(x_col, y_col, title)
        else:
            fig = self._create_bar_chart(x_col, y_col, color_col, title)

        # Convert to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)

        # Encode as base64 for embedding
        html_bytes = html.encode('utf-8')
        return base64.b64encode(html_bytes).decode('utf-8')

    def _create_bar_chart(self, x_col, y_col, color_col, title):
        """Create a bar chart."""
        if not x_col and self.categorical_cols:
            x_col = self.categorical_cols[0]
        if not y_col and self.numeric_cols:
            y_col = self.numeric_cols[0]

        if color_col:
            fig = px.bar(self.df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.bar(self.df, x=x_col, y=y_col, title=title)

        fig.update_layout(
            template='plotly_white',
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        return fig

    def _create_line_chart(self, x_col, y_col, color_col, title):
        """Create a line chart."""
        if not x_col:
            if self.datetime_cols:
                x_col = self.datetime_cols[0]
            elif self.categorical_cols:
                x_col = self.categorical_cols[0]
        if not y_col and self.numeric_cols:
            y_col = self.numeric_cols[0]

        if color_col:
            fig = px.line(self.df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.line(self.df, x=x_col, y=y_col, title=title)

        fig.update_layout(template='plotly_white')
        return fig

    def _create_pie_chart(self, names_col, values_col, title):
        """Create a pie chart."""
        if not names_col and self.categorical_cols:
            names_col = self.categorical_cols[0]
        if not values_col and self.numeric_cols:
            values_col = self.numeric_cols[0]

        fig = px.pie(self.df, names=names_col, values=values_col, title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def _create_scatter_chart(self, x_col, y_col, color_col, title):
        """Create a scatter plot."""
        if not x_col and len(self.numeric_cols) > 0:
            x_col = self.numeric_cols[0]
        if not y_col and len(self.numeric_cols) > 1:
            y_col = self.numeric_cols[1]
        elif not y_col:
            y_col = self.numeric_cols[0]

        if color_col:
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.scatter(self.df, x=x_col, y=y_col, title=title)

        fig.update_layout(template='plotly_white')
        return fig

    def _create_histogram(self, x_col, title):
        """Create a histogram."""
        if not x_col and self.numeric_cols:
            x_col = self.numeric_cols[0]

        fig = px.histogram(self.df, x=x_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    def _create_heatmap(self, title):
        """Create a correlation heatmap."""
        corr = self.df[self.numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=title,
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(template='plotly_white')
        return fig

    def _create_box_plot(self, x_col, y_col, title):
        """Create a box plot."""
        if not y_col and self.numeric_cols:
            y_col = self.numeric_cols[0]

        if x_col:
            fig = px.box(self.df, x=x_col, y=y_col, title=title)
        else:
            fig = px.box(self.df, y=y_col, title=title)

        fig.update_layout(template='plotly_white')
        return fig

    def auto_generate(self, query: str) -> str:
        """Automatically generate the best chart for a query."""
        chart_type = self.suggest_chart_type(query)
        return self.generate(chart_type, title=f"Result for: {query}")
