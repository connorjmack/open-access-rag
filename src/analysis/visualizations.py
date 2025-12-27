"""
Visualization utilities for article corpus analysis.
"""

from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
import pandas as pd


class Visualizer:
    """Creates visualizations for article corpus analysis."""

    def __init__(self, theme: str = "plotly"):
        """
        Initialize visualizer.

        Args:
            theme: Plotly theme to use
        """
        self.theme = theme

    def plot_publication_timeline(
        self,
        timeline_data: Dict[str, int],
        title: str = "Publications Over Time",
        return_figure: bool = True,
    ):
        """
        Create a timeline plot of publications.

        Args:
            timeline_data: Dictionary mapping time periods to counts
            title: Plot title
            return_figure: Whether to return figure object (True) or show plot (False)

        Returns:
            Plotly figure if return_figure=True
        """
        if not timeline_data:
            logger.warning("No timeline data to plot")
            return None

        periods = list(timeline_data.keys())
        counts = list(timeline_data.values())

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=periods,
                    y=counts,
                    mode="lines+markers",
                    name="Publications",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=8),
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Number of Publications",
            template=self.theme,
            hovermode="x unified",
        )

        if return_figure:
            return fig
        else:
            fig.show()

    def plot_keyword_trends(
        self,
        keyword_data: Dict[str, int],
        title: str = "Top Keywords",
        top_n: int = 20,
        return_figure: bool = True,
    ):
        """
        Create a bar chart of keyword frequencies.

        Args:
            keyword_data: Dictionary mapping keywords to frequencies
            title: Plot title
            top_n: Number of top keywords to show
            return_figure: Whether to return figure object

        Returns:
            Plotly figure if return_figure=True
        """
        if not keyword_data:
            logger.warning("No keyword data to plot")
            return None

        # Get top N keywords
        sorted_keywords = sorted(
            keyword_data.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        keywords = [kw for kw, _ in sorted_keywords]
        counts = [cnt for _, cnt in sorted_keywords]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts,
                    y=keywords,
                    orientation="h",
                    marker=dict(color="#2ca02c"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Keyword",
            template=self.theme,
            yaxis=dict(autorange="reversed"),
            height=max(400, len(keywords) * 25),
        )

        if return_figure:
            return fig
        else:
            fig.show()

    def plot_topic_evolution(
        self,
        topic_timeline: Dict[str, Dict[str, int]],
        title: str = "Topic Evolution Over Time",
        return_figure: bool = True,
    ):
        """
        Create a multi-line plot showing topic evolution.

        Args:
            topic_timeline: Dictionary mapping topics to timeline data
            title: Plot title
            return_figure: Whether to return figure object

        Returns:
            Plotly figure if return_figure=True
        """
        if not topic_timeline:
            logger.warning("No topic timeline data to plot")
            return None

        fig = go.Figure()

        for topic, timeline in topic_timeline.items():
            periods = list(timeline.keys())
            counts = list(timeline.values())

            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=counts,
                    mode="lines+markers",
                    name=topic,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Number of Mentions",
            template=self.theme,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        if return_figure:
            return fig
        else:
            fig.show()

    def plot_article_type_distribution(
        self,
        type_distribution: Dict[str, int],
        title: str = "Article Type Distribution",
        return_figure: bool = True,
    ):
        """
        Create a pie chart of article types.

        Args:
            type_distribution: Dictionary mapping article types to counts
            title: Plot title
            return_figure: Whether to return figure object

        Returns:
            Plotly figure if return_figure=True
        """
        if not type_distribution:
            logger.warning("No article type data to plot")
            return None

        labels = list(type_distribution.keys())
        values = list(type_distribution.values())

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                )
            ]
        )

        fig.update_layout(
            title=title,
            template=self.theme,
        )

        if return_figure:
            return fig
        else:
            fig.show()

    def plot_author_network(
        self,
        top_authors: List[tuple],
        title: str = "Top Authors by Publication Count",
        return_figure: bool = True,
    ):
        """
        Create a bar chart of top authors.

        Args:
            top_authors: List of (author, count) tuples
            title: Plot title
            return_figure: Whether to return figure object

        Returns:
            Plotly figure if return_figure=True
        """
        if not top_authors:
            logger.warning("No author data to plot")
            return None

        authors = [author for author, _ in top_authors]
        counts = [count for _, count in top_authors]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts,
                    y=authors,
                    orientation="h",
                    marker=dict(color="#ff7f0e"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Number of Publications",
            yaxis_title="Author",
            template=self.theme,
            yaxis=dict(autorange="reversed"),
            height=max(400, len(authors) * 30),
        )

        if return_figure:
            return fig
        else:
            fig.show()

    def create_dashboard_summary(
        self, stats: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create a collection of summary visualizations.

        Args:
            stats: Dictionary with various statistics

        Returns:
            Dictionary mapping plot names to figures
        """
        figures = {}

        # Publication timeline
        if "timeline" in stats:
            figures["timeline"] = self.plot_publication_timeline(
                stats["timeline"], return_figure=True
            )

        # Keyword trends
        if "keywords" in stats:
            figures["keywords"] = self.plot_keyword_trends(
                stats["keywords"], return_figure=True
            )

        # Topic evolution
        if "topic_evolution" in stats:
            figures["topic_evolution"] = self.plot_topic_evolution(
                stats["topic_evolution"], return_figure=True
            )

        # Article types
        if "article_types" in stats:
            figures["article_types"] = self.plot_article_type_distribution(
                stats["article_types"], return_figure=True
            )

        # Top authors
        if "top_authors" in stats:
            figures["top_authors"] = self.plot_author_network(
                stats["top_authors"], return_figure=True
            )

        logger.info(f"Created {len(figures)} dashboard visualizations")
        return figures
