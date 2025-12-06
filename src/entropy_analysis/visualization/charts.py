"""
Interactive visualization charts using Plotly.

Provides high-quality, interactive charts for entropy analysis.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from entropy_analysis.core.analyze import (
    BatchAnalysisResult,
    TextAnalysisResult,
)
from entropy_analysis.core.metrics import RollingEntropyResult

# Color palette - high contrast for better readability
COLORS = {
    "primary": "#2E86AB",  # Steel blue
    "secondary": "#A23B72",  # Deep pink
    "accent": "#F18F01",  # Amber
    "success": "#C73E1D",  # Rust red
    "neutral": "#0D1B2A",  # Dark blue-black (HIGH CONTRAST)
    "background": "#FFFFFF",  # Pure white
    "grid": "#D0D0D0",  # Medium grey
    "text_dark": "#1B263B",  # Dark blue
}

# Alternative palette for multiple series
SERIES_COLORS = [
    "#2E86AB",  # Steel blue
    "#A23B72",  # Deep pink
    "#F18F01",  # Amber
    "#C73E1D",  # Rust red
    "#6B4E71",  # Purple
    "#4A7C59",  # Forest green
    "#E8871E",  # Orange
    "#1B4965",  # Dark blue
]


def _create_base_layout(title: str, **kwargs: Any) -> dict[str, Any]:
    """Create consistent base layout for all charts with high contrast."""
    return {
        "title": {
            "text": title,
            "font": {
                "size": 22,
                "family": "JetBrains Mono, monospace",
                "color": COLORS["neutral"],
                "weight": 700,
            },
            "x": 0.5,
            "xanchor": "center",
        },
        "font": {
            "family": "JetBrains Mono, monospace",
            "color": COLORS["neutral"],
            "size": 14,
        },
        "paper_bgcolor": COLORS["background"],
        "plot_bgcolor": "white",
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "gridwidth": 1,
            "zeroline": False,
            "title": {"font": {"size": 16, "color": COLORS["text_dark"], "weight": 600}},
            "tickfont": {"size": 14, "color": COLORS["neutral"]},
        },
        "yaxis": {
            "gridcolor": COLORS["grid"],
            "gridwidth": 1,
            "zeroline": False,
            "title": {"font": {"size": 16, "color": COLORS["text_dark"], "weight": 600}},
            "tickfont": {"size": 14, "color": COLORS["neutral"]},
        },
        "hoverlabel": {
            "bgcolor": "white",
            "bordercolor": COLORS["neutral"],
            "font": {
                "size": 13,
                "family": "JetBrains Mono, monospace",
                "color": COLORS["neutral"],
            },
        },
        "legend": {
            "font": {"size": 13, "color": COLORS["neutral"]},
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": COLORS["grid"],
            "borderwidth": 1,
        },
        **kwargs,
    }


def create_letter_distribution_chart(
    result: TextAnalysisResult,
    sort_by: str = "alphabet",  # "alphabet" or "frequency"
    show_cumulative: bool = False,
) -> go.Figure:
    """
    Create an interactive bar chart of letter distribution.

    Args:
        result: Analysis result
        sort_by: How to sort letters ("alphabet" or "frequency")
        show_cumulative: Whether to show cumulative distribution line

    Returns:
        Plotly Figure
    """
    letters = [s.letter for s in result.letter_stats]
    probs = [s.probability for s in result.letter_stats]
    counts = [s.count for s in result.letter_stats]

    if sort_by == "frequency":
        # Sort by probability descending
        sorted_data = sorted(zip(letters, probs, counts, strict=False), key=lambda x: x[1], reverse=True)
        letters, probs, counts = zip(*sorted_data, strict=False) if sorted_data else ([], [], [])
        letters, probs, counts = list(letters), list(probs), list(counts)

    fig = go.Figure()

    # Main bars
    fig.add_trace(
        go.Bar(
            x=letters,
            y=probs,
            name="Вероятность",
            marker_color=COLORS["primary"],
            hovertemplate="<b>%{x}</b><br>Вероятность: %{y:.4f}<br>Количество: %{customdata}<extra></extra>",
            customdata=counts,
        )
    )

    if show_cumulative:
        cumulative = np.cumsum(probs)
        fig.add_trace(
            go.Scatter(
                x=letters,
                y=cumulative,
                name="Накопленная",
                mode="lines+markers",
                line={"color": COLORS["secondary"], "width": 2},
                marker={"size": 6},
                yaxis="y2",
            )
        )

    title = "Распределение начальных букв"
    if result.source_name:
        title += f" — {result.source_name}"

    layout = _create_base_layout(
        title,
        xaxis_title="Буква",
        yaxis_title="Вероятность (p)",
        showlegend=show_cumulative,
    )

    if show_cumulative:
        layout["yaxis2"] = {
            "title": "Накопленная вероятность",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1.05],
        }

    fig.update_layout(**layout)

    # Add annotation with metrics (high contrast)
    metrics_text = f"N={result.n_words}"
    if result.shannon_entropy:
        metrics_text += f"  H={result.shannon_entropy:.3f} бит"
    if result.mean_rank:
        metrics_text += f"  x̄={result.mean_rank:.2f}"
    if result.std_rank:
        metrics_text += f"  σ={result.std_rank:.2f}"

    fig.add_annotation(
        text=metrics_text,
        xref="paper",
        yref="paper",
        x=0.99,
        y=1.02,
        showarrow=False,
        font={"size": 13, "color": COLORS["neutral"], "weight": 600},
        xanchor="right",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        borderpad=4,
    )

    return fig


def create_entropy_histogram(
    batch: BatchAnalysisResult,
    nbins: int = 20,
    show_kde: bool = True,
) -> go.Figure:
    """
    Create histogram of entropy values from batch analysis.

    Args:
        batch: Batch analysis result
        nbins: Number of histogram bins
        show_kde: Whether to show kernel density estimate

    Returns:
        Plotly Figure
    """
    entropies = [r.shannon_entropy for _, r in batch.results if r.shannon_entropy is not None]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=entropies,
            nbinsx=nbins,
            name="Распределение H",
            marker_color=COLORS["primary"],
            opacity=0.7,
            hovertemplate="H: %{x:.3f}<br>Количество: %{y}<extra></extra>",
        )
    )

    if show_kde and len(entropies) > 5:
        # Add KDE line
        from scipy import stats as sp_stats

        kde = sp_stats.gaussian_kde(entropies)
        x_range = np.linspace(min(entropies), max(entropies), 100)
        kde_values = kde(x_range)
        # Scale to match histogram
        kde_scaled = kde_values * len(entropies) * (max(entropies) - min(entropies)) / nbins

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_scaled,
                name="KDE",
                mode="lines",
                line={"color": COLORS["secondary"], "width": 2},
            )
        )

    # Add vertical lines for mean and median
    if batch.extended_stats:
        fig.add_vline(
            x=batch.extended_stats.mean,
            line_dash="dash",
            line_color=COLORS["accent"],
            line_width=3,
            annotation_text=f"Среднее: {batch.extended_stats.mean:.3f}",
            annotation_position="top",
            annotation_font={"size": 14, "color": COLORS["neutral"], "weight": 600},
        )
        fig.add_vline(
            x=batch.extended_stats.median,
            line_dash="dot",
            line_color=COLORS["success"],
            line_width=3,
            annotation_text=f"Медиана: {batch.extended_stats.median:.3f}",
            annotation_position="bottom",
            annotation_font={"size": 14, "color": COLORS["neutral"], "weight": 600},
        )

    layout = _create_base_layout(
        "Распределение энтропии текстов",
        xaxis_title="Энтропия H (бит)",
        yaxis_title="Количество текстов",
        showlegend=show_kde,
    )

    fig.update_layout(**layout)

    return fig


def create_correlation_scatter(
    batch: BatchAnalysisResult,
    show_trendline: bool = True,
    highlight_outliers: bool = True,
    label_top_n: int = 0,
) -> go.Figure:
    """
    Create scatter plot of N vs H with correlation analysis.

    Args:
        batch: Batch analysis result
        show_trendline: Whether to show linear regression line
        highlight_outliers: Whether to highlight outliers
        label_top_n: Number of highest-entropy points to label

    Returns:
        Plotly Figure
    """
    # Extract data
    names = []
    n_values = []
    h_values = []

    for name, result in batch.results:
        if result.shannon_entropy is not None:
            names.append(name)
            n_values.append(result.n_words)
            h_values.append(result.shannon_entropy)

    fig = go.Figure()

    # Determine outliers if requested
    outlier_mask = [False] * len(h_values)
    if highlight_outliers and batch.extended_stats:
        q1 = batch.extended_stats.q1
        q3 = batch.extended_stats.q3
        iqr = batch.extended_stats.iqr
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = [h < lower or h > upper for h in h_values]

    # Determine labels
    label_indices = set()
    if label_top_n > 0:
        sorted_indices = sorted(range(len(h_values)), key=lambda i: h_values[i], reverse=True)
        label_indices = set(sorted_indices[:label_top_n])

    # Regular points
    regular_n = [n for i, n in enumerate(n_values) if not outlier_mask[i]]
    regular_h = [h for i, h in enumerate(h_values) if not outlier_mask[i]]
    regular_names = [name for i, name in enumerate(names) if not outlier_mask[i]]

    fig.add_trace(
        go.Scatter(
            x=regular_n,
            y=regular_h,
            mode="markers",
            name="Тексты",
            marker={
                "color": COLORS["primary"],
                "size": 8,
                "line": {"width": 1, "color": "white"},
            },
            text=regular_names,
            hovertemplate="<b>%{text}</b><br>N=%{x}<br>H=%{y:.4f} бит<extra></extra>",
        )
    )

    # Outlier points
    if highlight_outliers and any(outlier_mask):
        outlier_n = [n for i, n in enumerate(n_values) if outlier_mask[i]]
        outlier_h = [h for i, h in enumerate(h_values) if outlier_mask[i]]
        outlier_names = [name for i, name in enumerate(names) if outlier_mask[i]]

        fig.add_trace(
            go.Scatter(
                x=outlier_n,
                y=outlier_h,
                mode="markers",
                name="Выбросы",
                marker={
                    "color": COLORS["secondary"],
                    "size": 10,
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "white"},
                },
                text=outlier_names,
                hovertemplate="<b>%{text}</b> (выброс)<br>N=%{x}<br>H=%{y:.4f} бит<extra></extra>",
            )
        )

    # Trendline
    if show_trendline and batch.correlation_slope is not None:
        x_min, x_max = min(n_values), max(n_values)
        trend_x = [x_min, x_max]
        trend_y = [
            batch.correlation_slope * x_min + batch.correlation_intercept,
            batch.correlation_slope * x_max + batch.correlation_intercept,
        ]

        fig.add_trace(
            go.Scatter(
                x=trend_x,
                y=trend_y,
                mode="lines",
                name=f"Тренд (r={batch.correlation:.3f})",
                line={"color": COLORS["accent"], "width": 2, "dash": "dash"},
            )
        )

    # Labels for top N
    if label_indices:
        for i in label_indices:
            short_name = names[i].split("/")[-1].split("\\")[-1][:20]
            fig.add_annotation(
                x=n_values[i],
                y=h_values[i],
                text=short_name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLORS["neutral"],
                font={"size": 12, "color": COLORS["neutral"], "weight": 600},
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=COLORS["neutral"],
                borderwidth=1,
                borderpad=3,
                ax=25,
                ay=-25,
            )

    # Title with correlation info
    title = "Корреляция: H vs N"
    if batch.correlation is not None:
        title += f" (r={batch.correlation:.4f}, R²={batch.correlation_r_squared:.4f})"

    layout = _create_base_layout(
        title,
        xaxis_title="N (количество слов)",
        yaxis_title="H (энтропия, бит)",
    )

    fig.update_layout(**layout)

    # Add equation annotation (high contrast)
    if batch.correlation_slope is not None:
        eq_text = f"y = {batch.correlation_slope:.4f}x + {batch.correlation_intercept:.4f}"
        fig.add_annotation(
            text=eq_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font={
                "size": 14,
                "family": "JetBrains Mono",
                "color": COLORS["neutral"],
                "weight": 600,
            },
            bgcolor="white",
            bordercolor=COLORS["neutral"],
            borderwidth=2,
            borderpad=6,
        )

    return fig


def create_dual_author_comparison(
    batch1: BatchAnalysisResult,
    batch2: BatchAnalysisResult,
    author1_name: str = "Автор 1",
    author2_name: str = "Автор 2",
) -> go.Figure:
    """
    Create comparison scatter plot for two authors.

    Args:
        batch1: First author's analysis
        batch2: Second author's analysis
        author1_name: Name for first author
        author2_name: Name for second author

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Author 1 data
    n1 = [r.n_words for _, r in batch1.results if r.shannon_entropy is not None]
    h1 = [r.shannon_entropy for _, r in batch1.results if r.shannon_entropy is not None]

    fig.add_trace(
        go.Scatter(
            x=n1,
            y=h1,
            mode="markers",
            name=author1_name,
            marker={"color": COLORS["primary"], "size": 8},
        )
    )

    # Author 1 trendline
    if batch1.correlation_slope is not None and n1:
        x_range = [min(n1), max(n1)]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[batch1.correlation_slope * x + batch1.correlation_intercept for x in x_range],
                mode="lines",
                name=f"{author1_name} (r={batch1.correlation:.3f})",
                line={"color": COLORS["primary"], "width": 2, "dash": "dash"},
            )
        )

    # Author 2 data
    n2 = [r.n_words for _, r in batch2.results if r.shannon_entropy is not None]
    h2 = [r.shannon_entropy for _, r in batch2.results if r.shannon_entropy is not None]

    fig.add_trace(
        go.Scatter(
            x=n2,
            y=h2,
            mode="markers",
            name=author2_name,
            marker={"color": COLORS["secondary"], "size": 8},
        )
    )

    # Author 2 trendline
    if batch2.correlation_slope is not None and n2:
        x_range = [min(n2), max(n2)]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[batch2.correlation_slope * x + batch2.correlation_intercept for x in x_range],
                mode="lines",
                name=f"{author2_name} (r={batch2.correlation:.3f})",
                line={"color": COLORS["secondary"], "width": 2, "dash": "dash"},
            )
        )

    layout = _create_base_layout(
        f"Сравнение корреляций: {author1_name} vs {author2_name}",
        xaxis_title="N (количество слов)",
        yaxis_title="H (энтропия, бит)",
    )

    fig.update_layout(**layout)

    return fig


def create_rolling_entropy_chart(
    result: RollingEntropyResult,
    title: str = "Скользящая энтропия",
) -> go.Figure:
    """
    Create line chart for rolling entropy analysis.

    Args:
        result: Rolling entropy result
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result.positions,
            y=result.entropies,
            mode="lines+markers",
            name="Энтропия",
            line={"color": COLORS["primary"], "width": 2},
            marker={"size": 4},
            hovertemplate="Позиция: %{x}<br>H: %{y:.4f} бит<extra></extra>",
        )
    )

    # Add mean line (high contrast)
    fig.add_hline(
        y=result.mean_entropy,
        line_dash="dash",
        line_color=COLORS["accent"],
        line_width=3,
        annotation_text=f"Среднее: {result.mean_entropy:.3f}",
        annotation_position="right",
        annotation_font={"size": 14, "color": COLORS["neutral"], "weight": 600},
    )

    # Add confidence band (mean ± std)
    fig.add_hrect(
        y0=result.mean_entropy - result.std_entropy,
        y1=result.mean_entropy + result.std_entropy,
        fillcolor=COLORS["primary"],
        opacity=0.15,
        line_width=0,
        annotation_text=f"±σ ({result.std_entropy:.3f})",
        annotation_position="right",
        annotation_font={"size": 13, "color": COLORS["neutral"], "weight": 600},
    )

    layout = _create_base_layout(
        f"{title} (окно={result.window_size} слов)",
        xaxis_title="Позиция (номер слова)",
        yaxis_title="H (энтропия, бит)",
    )

    fig.update_layout(**layout)

    return fig


def create_zipf_plot(
    result: TextAnalysisResult,
) -> go.Figure:
    """
    Create Zipf's law plot (log-log rank-frequency).

    Args:
        result: Analysis result

    Returns:
        Plotly Figure
    """
    # Sort by frequency
    sorted_stats = sorted(result.letter_stats, key=lambda s: s.count, reverse=True)
    sorted_stats = [s for s in sorted_stats if s.count > 0]

    ranks = list(range(1, len(sorted_stats) + 1))
    counts = [s.count for s in sorted_stats]
    letters = [s.letter for s in sorted_stats]

    fig = go.Figure()

    # Actual data
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=counts,
            mode="markers",
            name="Данные",
            marker={"color": COLORS["primary"], "size": 10},
            text=letters,
            hovertemplate="<b>%{text}</b><br>Ранг: %{x}<br>Частота: %{y}<extra></extra>",
        )
    )

    # Zipf fit line
    if result.zipf_alpha and result.zipf_r_squared:
        # Calculate expected counts based on Zipf
        c = counts[0]  # First element as reference
        expected = [c / (r**result.zipf_alpha) for r in ranks]

        fig.add_trace(
            go.Scatter(
                x=ranks,
                y=expected,
                mode="lines",
                name=f"Ципф (α={result.zipf_alpha:.2f}, R²={result.zipf_r_squared:.3f})",
                line={"color": COLORS["secondary"], "width": 2, "dash": "dash"},
            )
        )

    layout = _create_base_layout(
        "Анализ закона Ципфа",
        xaxis_title="Ранг",
        yaxis_title="Частота",
        xaxis_type="log",
        yaxis_type="log",
    )

    fig.update_layout(**layout)

    return fig


def create_comparison_heatmap(
    comparisons: list[list[float]],
    labels: list[str],
    metric_name: str = "Jensen-Shannon Divergence",
) -> go.Figure:
    """
    Create heatmap for pairwise text comparisons.

    Args:
        comparisons: 2D matrix of comparison values
        labels: Labels for texts
        metric_name: Name of the metric

    Returns:
        Plotly Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=comparisons,
            x=labels,
            y=labels,
            colorscale="Blues",
            reversescale=True,
            hovertemplate="%{y} vs %{x}<br>%{z:.4f}<extra></extra>",
            colorbar={"title": metric_name},
        )
    )

    layout = _create_base_layout(
        f"Матрица сравнения: {metric_name}",
        xaxis_title="",
        yaxis_title="",
    )

    # Update layout separately to avoid keyword conflicts
    fig.update_layout(**layout)
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

    # Add annotations with high contrast
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Determine text color based on cell value for readability
            value = comparisons[i][j]
            text_color = "white" if value > 0.4 else "#0D1B2A"

            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{value:.3f}",
                showarrow=False,
                font={
                    "size": 13,
                    "color": text_color,
                    "weight": 600,
                    "family": "JetBrains Mono",
                },
            )

    return fig


def create_radar_chart(
    results: list[TextAnalysisResult],
    names: list[str] | None = None,
) -> go.Figure:
    """
    Create radar chart comparing multiple texts across metrics.

    Args:
        results: List of analysis results
        names: Optional names for each result

    Returns:
        Plotly Figure
    """
    if not results:
        return go.Figure()

    if names is None:
        names = [r.source_name or f"Text {i + 1}" for i, r in enumerate(results)]

    # Metrics to compare (normalized to 0-1 scale)
    categories = ["Энтропия (норм)", "Разнообразие", "Ципф α", "Кол-во букв (норм)"]

    fig = go.Figure()

    for i, (result, name) in enumerate(zip(results, names, strict=False)):
        # Normalize metrics
        h_norm = result.normalized_entropy.h_normalized if result.normalized_entropy else 0
        diversity = result.gini_simpson_index or 0
        zipf = min(result.zipf_alpha / 2, 1) if result.zipf_alpha else 0
        letters_norm = result.n_unique_letters / result.alphabet_size

        values = [h_norm, diversity, zipf, letters_norm]

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # Close the shape
                theta=categories + [categories[0]],
                name=name[:20],
                line={"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
            )
        )

    layout = _create_base_layout(
        "Сравнительный анализ текстов",
        polar={
            "radialaxis": {"visible": True, "range": [0, 1]},
            "angularaxis": {"rotation": 90},
        },
    )

    fig.update_layout(**layout)

    return fig


def create_bootstrap_plot(
    result: TextAnalysisResult,
) -> go.Figure:
    """
    Create visualization of bootstrap confidence interval.

    Args:
        result: Analysis result with bootstrap data

    Returns:
        Plotly Figure
    """
    if not result.bootstrap:
        return go.Figure()

    bootstrap = result.bootstrap

    fig = go.Figure()

    # Point estimate with confidence interval
    fig.add_trace(
        go.Scatter(
            x=[bootstrap.ci_lower, bootstrap.estimate, bootstrap.ci_upper],
            y=[1, 1, 1],
            mode="markers+lines",
            marker={
                "size": [8, 16, 8],
                "color": [COLORS["primary"], COLORS["accent"], COLORS["primary"]],
                "symbol": ["line-ew", "circle", "line-ew"],
            },
            line={"color": COLORS["primary"], "width": 2},
            hoverinfo="text",
            text=[
                f"CI нижняя: {bootstrap.ci_lower:.4f}",
                f"Оценка: {bootstrap.estimate:.4f}",
                f"CI верхняя: {bootstrap.ci_upper:.4f}",
            ],
        )
    )

    # Add error bar
    fig.add_trace(
        go.Scatter(
            x=[bootstrap.estimate],
            y=[1],
            mode="markers",
            marker={"size": 16, "color": COLORS["accent"]},
            error_x={
                "type": "constant",
                "value": bootstrap.estimate - bootstrap.ci_lower,
                "valueminus": bootstrap.ci_upper - bootstrap.estimate,
                "color": COLORS["primary"],
                "thickness": 2,
            },
            showlegend=False,
        )
    )

    title = f"Bootstrap Confidence Interval ({bootstrap.confidence_level * 100:.0f}%)"
    subtitle = f"H = {bootstrap.estimate:.4f} ± {bootstrap.std_error:.4f}"

    layout = _create_base_layout(
        f"{title}<br><sub>{subtitle}</sub>",
        xaxis_title="Энтропия H (бит)",
        yaxis_visible=False,
        height=200,
    )

    fig.update_layout(**layout)

    return fig
