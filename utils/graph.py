from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Curve = Callable[[np.ndarray], np.ndarray]

WINDOW_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _check_params(alpha: float, x_min: float) -> None:
    if alpha <= 1 or x_min <= 0:
        raise ValueError("need alpha > 1 and x_min > 0")


def add_pdf(fig: go.Figure, col: int, pdf: Curve, x_min: float, x: np.ndarray) -> None:
    fig.add_trace(go.Scatter(x=x, y=pdf(x), mode="lines"), row=1, col=col)
    fig.add_vline(x=x_min, line_dash="dash", line_color="gray", row=1, col=col)
    fig.update_yaxes(title_text="p(x)", rangemode="tozero", row=1, col=col)


def add_cdf(fig: go.Figure, col: int, cdf: Curve, x_min: float, x: np.ndarray) -> None:
    fig.add_trace(go.Scatter(x=x, y=cdf(x), mode="lines"), row=1, col=col)
    fig.add_vline(x=x_min, line_dash="dash", line_color="gray", row=1, col=col)
    fig.update_yaxes(title_text="F(x)", range=[0, 1.05], row=1, col=col)


def add_sum_share(fig: go.Figure, col: int, share: Curve, n: int = 500) -> None:
    q = np.linspace(0, 1, n)
    fig.add_trace(go.Scatter(x=q, y=q, mode="lines", line_dash="dot", line_color="gray"), row=1, col=col)
    fig.add_trace(go.Scatter(x=q, y=share(q), mode="lines"), row=1, col=col)
    fig.add_vline(x=0.2, line_dash="dash", line_color="firebrick", row=1, col=col)
    fig.add_hline(y=0.8, line_dash="dash", line_color="firebrick", row=1, col=col)
    fig.update_xaxes(title_text="top fraction of events q", range=[0, 1], row=1, col=col)
    fig.update_yaxes(title_text="S(x_q)", range=[0, 1.05], row=1, col=col)


def plot_power_law(
    alpha: float,
    x_min: float,
    k: float,
    pdf: Curve,
    cdf: Curve,
    share: Curve,
    x_max: float | None = None,
    n: int = 500,
) -> go.Figure:
    _check_params(alpha, x_min)
    x_max = x_max if x_max is not None else 10 * x_min
    x = np.linspace(x_min, x_max, n)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "PDF  p(x) = k · x^(-α)",
            "CDF  F(x) = 1 - (x / x_min)^(-(α - 1))",
            "Share of the total sum  S(x_q) = q^((α - 2) / (α - 1))",
        ),
    )
    add_pdf(fig, 1, pdf, x_min, x)
    add_cdf(fig, 2, cdf, x_min, x)
    add_sum_share(fig, 3, share, n)

    fig.update_xaxes(title_text="x", range=[0, x_max], row=1, col=1)
    fig.update_xaxes(title_text="x", range=[0, x_max], matches="x", row=1, col=2)
    fig.update_layout(
        title=(
            f"Power law  α = {alpha}, x_min = {x_min}"
            f"<br><sup>k = {k:.4g}, "
            f"top q = 20% of events hold S(x_q) = {float(share(np.asarray(0.2))):.1%} of the total sum</sup>"
        ),
        showlegend=False,
        height=450,
        width=1400,
    )
    return fig


def log_binned_density(samples: np.ndarray, x_min: float, bins_per_decade: int = 5) -> tuple[np.ndarray, np.ndarray]:
    decades = np.log10(samples.max() / x_min)
    edges = x_min * np.logspace(0, decades, int(np.ceil(decades * bins_per_decade)) + 1)
    counts, _ = np.histogram(samples, bins=edges)
    density = counts / (len(samples) * np.diff(edges))
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mid of each bin
    return centers[counts > 0], density[counts > 0]


def add_log_log(
    fig: go.Figure, col: int, pdf: Curve, samples: np.ndarray, x_min: float, x: np.ndarray, windows: list[float]
) -> None:
    centers, density = log_binned_density(samples, x_min)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=density,
            mode="markers",
            name=f"{len(samples):,} samples, log bins",
            marker=dict(color="gray", size=8),
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(x=x, y=pdf(x), mode="lines", name="Y = log k - α · X", line=dict(color="black", width=2)),
        row=1,
        col=col,
    )
    # highlight the windows zoomed into on the right
    for x0, color in zip(windows, WINDOW_COLORS):
        xw = np.logspace(np.log10(x0), np.log10(10 * x0), 50)
        fig.add_trace(
            go.Scatter(x=xw, y=pdf(xw), mode="lines", line=dict(color=color, width=8), opacity=0.5, legendgroup=str(x0), showlegend=False),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="x  (log scale)", type="log", exponentformat="power", row=1, col=col)
    fig.update_yaxes(title_text="p(x)  (log scale)", type="log", exponentformat="power", row=1, col=col)


def add_scale_free(fig: go.Figure, col: int, pdf: Curve, windows: list[float], n: int = 500) -> None:
    ratio = np.linspace(1, 10, n)  # x / x0
    # every window collapses onto the same curve, wider lines underneath so all stay visible
    widths = np.linspace(10, 2, len(windows))
    for x0, color, width in zip(windows, WINDOW_COLORS, widths):
        fig.add_trace(
            go.Scatter(
                x=ratio,
                y=pdf(x0 * ratio) / pdf(np.asarray(x0)),
                mode="lines",
                name=f"window [{x0:g}, {10 * x0:g}]",
                line=dict(color=color, width=width),
                legendgroup=str(x0),
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="x / x0", range=[1, 10], row=1, col=col)
    fig.update_yaxes(title_text="p(x) / p(x0)", rangemode="tozero", row=1, col=col)


def plot_log_log(
    alpha: float,
    x_min: float,
    log_pdf: Curve,
    samples: np.ndarray,
    windows: list[float] | None = None,
) -> go.Figure:
    """`log_pdf` maps X = log10(x) to Y = log10(p(x))."""
    _check_params(alpha, x_min)

    def pdf(x: np.ndarray) -> np.ndarray:
        # back from log space, plotly's log axes expect raw values
        return 10 ** log_pdf(np.log10(x))

    windows = windows if windows is not None else [x_min * 10**i for i in range(4)]
    x = np.logspace(np.log10(x_min), np.log10(100 * windows[-1]), 500)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Log-log (both axes logarithmic)  log p(x) = log k - α · log x",
            "Scale free (linear axes)  p(x) / p(x0) = (x / x0)^(-α)",
        ),
    )
    add_log_log(fig, 1, pdf, samples, x_min, x, windows)
    add_scale_free(fig, 2, pdf, windows)

    fig.update_layout(
        title=(
            f"Power law on a log-log plot  α = {alpha}, x_min = {x_min}"
            f"<br><sup>straight line with slope -α = {-alpha:g}, every zoomed window has the same shape</sup>"
        ),
        height=450,
        width=1400,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
