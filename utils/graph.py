from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
import polars as pl
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
    fig.add_trace(
        go.Scatter(x=q, y=q, mode="lines", line_dash="dot", line_color="gray"), row=1, col=col
    )
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


def log_bin_edges(x_min: float, x_max: float, bins_per_decade: int = 5) -> np.ndarray:
    decades = np.log10(x_max / x_min)
    return x_min * np.logspace(0, decades, int(np.ceil(decades * bins_per_decade)) + 1)


def log_binned_density(
    samples: np.ndarray, x_min: float, bins_per_decade: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    edges = log_bin_edges(x_min, samples.max(), bins_per_decade)
    counts, _ = np.histogram(samples, bins=edges)
    density = counts / (len(samples) * np.diff(edges))
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mid of each bin
    return centers[counts > 0], density[counts > 0]


def add_log_log(
    fig: go.Figure,
    col: int,
    pdf: Curve,
    samples: np.ndarray,
    x_min: float,
    x: np.ndarray,
    windows: list[float],
) -> None:
    centers, density = log_binned_density(samples, x_min)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=density,
            mode="markers",
            name=f"{len(samples):,} samples, log bins",
            marker={"color": "gray", "size": 8},
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf(x),
            mode="lines",
            name="Y = log k - α · X",
            line={"color": "black", "width": 2},
        ),
        row=1,
        col=col,
    )
    # highlight the windows zoomed into on the right
    for x0, color in zip(windows, WINDOW_COLORS):
        xw = np.logspace(np.log10(x0), np.log10(10 * x0), 50)
        fig.add_trace(
            go.Scatter(
                x=xw,
                y=pdf(xw),
                mode="lines",
                line={"color": color, "width": 8},
                opacity=0.5,
                legendgroup=str(x0),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(
        title_text="x  (log scale)", type="log", exponentformat="power", row=1, col=col
    )
    fig.update_yaxes(
        title_text="p(x)  (log scale)", type="log", exponentformat="power", row=1, col=col
    )


def add_scale_free(
    fig: go.Figure, col: int, pdf: Curve, windows: list[float], n: int = 500
) -> None:
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
                line={"color": color, "width": width},
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
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


ALPHA_COLORS = ("#2a78d6", "#e34948", "#008300", "#1baf7a", "#4a3aa7", "#eda100")
ALPHA_SYMBOLS = ("circle", "square", "cross", "diamond", "triangle-up", "x")


def plot_alpha_comparison(
    alphas: list[float],
    curve: Callable[[float, np.ndarray], np.ndarray],
    x: np.ndarray,
    title: str,
    y_title: str,
    x_title: str = "x",
    log_x: bool = True,
    log_y: bool = False,
    dashed: Callable[[float], bool] = lambda alpha: False,
    label: Callable[[float], str] = lambda alpha: f"α = {alpha:g}",
    linear_panel: bool = False,
    linear_x_max: float | None = None,
    log_log_curve: Callable[[float, np.ndarray], np.ndarray] | None = None,
    log_log_y_title: str = "",
) -> go.Figure:
    if len(alphas) > len(ALPHA_COLORS):
        raise ValueError(f"at most {len(ALPHA_COLORS)} alphas")

    # (curve, x values, log x, log y, y title, subplot title)
    panels = []
    if linear_panel:
        x_lin = np.linspace(x.min(), linear_x_max if linear_x_max is not None else x.max(), len(x))
        panels.append((curve, x_lin, False, False, y_title, "linear axes"))
    main_title = "log-log axes" if log_x and log_y else "log x-axis" if log_x else "linear axes"
    panels.append((curve, x, log_x, log_y, y_title, main_title))
    if log_log_curve is not None:
        panels.append((log_log_curve, x, True, True, log_log_y_title, "log-log axes"))

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[p[-1] for p in panels] if len(panels) > 1 else None,
        horizontal_spacing=0.08,
    )
    for col, (panel_curve, xs, panel_log_x, panel_log_y, panel_y_title, _) in enumerate(
        panels, start=1
    ):
        for alpha, color, symbol in zip(sorted(alphas), ALPHA_COLORS, ALPHA_SYMBOLS):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=panel_curve(alpha, xs),
                    mode="lines+markers",
                    name=label(alpha),
                    legendgroup=str(alpha),
                    showlegend=col == 1,
                    line={"color": color, "width": 2, "dash": "dash" if dashed(alpha) else "solid"},
                    marker={"symbol": symbol, "size": 8, "color": color, "maxdisplayed": 12},
                    hovertemplate=f"{label(alpha)}<br>{x_title} = %{{x:.3g}}<br>{panel_y_title} = %{{y:.3g}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        if panel_log_x:
            fig.update_xaxes(
                title_text=f"{x_title}  (log scale)",
                type="log",
                exponentformat="power",
                row=1,
                col=col,
            )
        else:
            fig.update_xaxes(title_text=x_title, range=[0, xs.max()], row=1, col=col)
        if panel_log_y:
            fig.update_yaxes(
                title_text=f"{panel_y_title}  (log scale)",
                type="log",
                exponentformat="power",
                row=1,
                col=col,
            )
        elif linear_panel and col == 1:
            fig.update_yaxes(title_text=panel_y_title, rangemode="tozero", row=1, col=col)
        else:
            fig.update_yaxes(title_text=panel_y_title, range=[0, 1.05], row=1, col=col)
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=450,
        width=900 if len(panels) == 1 else 1400,
        template="plotly_white",
    )
    return fig


SAMPLE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948")


def auto_linear_bins(samples: np.ndarray, max_bins: int = 10_000) -> int:
    q25, q75 = np.percentile(samples, [25, 75])
    width = 2 * (q75 - q25) / len(samples) ** (1 / 3)
    return int(np.clip(np.ceil(np.ptp(samples) / width), 1, max_bins))


def fit_log_log(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), deg=1)
    return slope, intercept


def add_fit(fig: go.Figure, row: int, col: int, x: np.ndarray, y: np.ndarray, color: str) -> None:
    slope, intercept = fit_log_log(x, y)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=10**intercept * x_fit**slope,
            mode="lines",
            line={"color": color, "width": 3, "dash": "dash"},
        ),
        row=row,
        col=col,
    )
    fig.add_annotation(
        text=f"MSE fit slope = {slope:.2f}",
        x=0.98,
        y=0.98,
        xref="x domain",
        yref="y domain",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font={"color": color},
        row=row,
        col=col,
    )


def plot_binning_comparison(
    alpha: float,
    x_min: float,
    pdf: Curve,
    ccdf: Curve,
    samples: dict[int, np.ndarray],
    linear_bins: int | None = None,
) -> go.Figure:
    _check_params(alpha, x_min)
    sizes = sorted(samples)
    fig = make_subplots(
        rows=len(sizes),
        cols=3,
        subplot_titles=[
            "linear bins, density",
            "log bins, density",
            "CCDF  1 - F(x) at log bin edges",
        ]
        + [""] * 3 * (len(sizes) - 1),
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
    )
    color = SAMPLE_COLORS[0]
    true_style = {"color": "black", "width": 2}
    x_true = np.logspace(np.log10(x_min), np.log10(max(s.max() for s in samples.values())), 500)

    for row, n in enumerate(sizes, start=1):
        s = samples[n]
        counts, edges = np.histogram(s, bins=linear_bins or auto_linear_bins(s))
        width = edges[1] - edges[0]
        nonzero = counts > 0  # empty bins cannot be drawn on a log axis
        lin_x = ((edges[:-1] + edges[1:]) / 2)[nonzero]
        lin_y = counts[nonzero] / (n * width)
        log_x, log_y = log_binned_density(s, x_min)
        log_x, log_y = log_x[log_y > 0], log_y[log_y > 0]
        # empirical P(X >= b) at each log bin edge, the last edge can lie past the largest sample where it is 0
        ccdf_x = log_bin_edges(x_min, s.max())
        ccdf_y = 1 - np.searchsorted(np.sort(s), ccdf_x) / n
        ccdf_x, ccdf_y = ccdf_x[ccdf_y > 0], ccdf_y[ccdf_y > 0]

        panels = ((lin_x, lin_y, pdf), (log_x, log_y, pdf), (ccdf_x, ccdf_y, ccdf))
        for col, (x, y, true) in enumerate(panels, start=1):
            fig.add_trace(
                go.Scatter(x=x_true, y=true(x_true), mode="lines", line=true_style),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="markers", marker={"color": color, "size": 8}),
                row=row,
                col=col,
            )
            add_fit(fig, row, col, x, y, color)
            fig.update_xaxes(type="log", exponentformat="power", matches="x", row=row, col=col)
            fig.update_yaxes(
                type="log",
                exponentformat="power",
                matches="y3" if col == 3 else "y",
                row=row,
                col=col,
            )

        # a linear bin holding a single sample has density 1 / (N · Δx), the tail cannot go lower
        floor = 1 / (n * width)
        fig.add_hline(y=floor, line_dash="dot", line_color="gray", row=row, col=1)
        fig.add_annotation(
            text=f"1 sample per bin  (Δx = {width:.3g})",
            x=0.02,
            y=np.log10(floor),  # annotations on a log axis take log10 coordinates
            xref="x domain",
            yanchor="top",
            xanchor="left",
            showarrow=False,
            font={"size": 11, "color": "#555"},
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=f"<b>N = {n:,}</b><br>p(x)  (log scale)", row=row, col=1)
        fig.update_yaxes(title_text="1 - F(x)  (log scale)", row=row, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(title_text="x  (log scale)", row=len(sizes), col=col)
    fig.update_layout(
        title=(
            f"Linear bins flatten the tail, log bins and the CCDF recover the slope  (α = {alpha:g})"
            f"<br><sup>black: true curve, slope -α = {-alpha:g} for the density and -(α - 1) = {-(alpha - 1):g} for the CCDF, "
            "dashed: least-squares fit in log-log</sup>"
        ),
        showlegend=False,
        height=320 * len(sizes) + 100,
        width=1500,
        template="plotly_white",
    )
    return fig


def plot_alpha_estimates(
    true_alpha: float,
    estimates: dict[str, dict[int, float]],
    errors: dict[str, dict[int, float]] | None = None,
) -> go.Figure:
    errors = errors or {}
    if len(estimates) > len(ALPHA_COLORS):
        raise ValueError(f"at most {len(ALPHA_COLORS)} methods")
    fig = go.Figure()
    for (method, by_n), color, symbol in zip(estimates.items(), ALPHA_COLORS, ALPHA_SYMBOLS):
        sizes = sorted(by_n)
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[by_n[n] for n in sizes],
                error_y={"array": [errors[method][n] for n in sizes], "color": color}
                if method in errors
                else None,
                mode="lines+markers",
                name=method,
                line={"color": color, "width": 2},
                marker={"symbol": symbol, "size": 10, "color": color},
                hovertemplate=f"{method}<br>N = %{{x:,}}<br>α̂ = %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_hline(
        y=true_alpha,
        line_dash="dash",
        line_color="black",
        annotation_text=f"true α = {true_alpha:g}",
    )
    fig.update_xaxes(title_text="sample size N  (log scale)", type="log", exponentformat="power")
    fig.update_yaxes(title_text="estimated α̂")
    fig.update_layout(
        title="Estimated α vs sample size", height=450, width=900, template="plotly_white"
    )
    return fig


def plot_moments(
    n: np.ndarray,
    moments: dict[float, dict[str, np.ndarray]],
    true_values: dict[float, dict[str, float]],
    all_alphas: list[float],
) -> go.Figure:
    styles = dict(zip(sorted(all_alphas), zip(ALPHA_COLORS, ALPHA_SYMBOLS)))
    stats = list(next(iter(moments.values())))
    fig = make_subplots(rows=1, cols=len(stats), subplot_titles=[f"sample {s}" for s in stats])
    for alpha in sorted(moments):
        color, symbol = styles[alpha]
        for col, stat in enumerate(stats, start=1):
            fig.add_trace(
                go.Scatter(
                    x=n,
                    y=moments[alpha][stat],
                    mode="lines+markers",
                    name=f"α = {alpha:g}",
                    legendgroup=str(alpha),
                    showlegend=col == 1,
                    line={"color": color, "width": 2},
                    marker={"symbol": symbol, "size": 8, "color": color, "maxdisplayed": 12},
                    hovertemplate=f"α = {alpha:g}<br>n = %{{x:,}}<br>{stat} = %{{y:.4g}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
            true = true_values[alpha][stat]
            if np.isfinite(true):
                fig.add_trace(
                    go.Scatter(
                        x=[n[0], n[-1]],
                        y=[true, true],
                        mode="lines",
                        name=f"α = {alpha:g} true",
                        legendgroup=str(alpha),
                        showlegend=False,
                        line={"color": color, "width": 1.5, "dash": "dash"},
                        hovertemplate=f"α = {alpha:g} true {stat} = {true:.4g}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
    fig.update_xaxes(title_text="sample size n  (log scale)", type="log", exponentformat="power")
    for col, stat in enumerate(stats, start=1):
        fig.update_yaxes(
            title_text=f"{stat}  (log scale)", type="log", exponentformat="power", row=1, col=col
        )
    fig.update_layout(
        title="Sample statistics vs sample size<br><sup>solid: running value on one sample, dashed: true value (missing where it is infinite)</sup>",
        height=500,
        width=1600,
        template="plotly_white",
    )
    return fig


MODEL_COLORS = {"power law": "#3a6ea5", "normal": "#e76f51", "data": "#b8afa4"}
SURFACE = "#fcfcfb"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
GUIDE = "#b5b3ab"
REFERENCE_LINE = {"line_dash": "dash", "line_color": GUIDE, "line_width": 1}
RING = {"color": SURFACE, "width": 1}

REAL_WORLD_TEMPLATE = go.layout.Template(
    layout={
        "font": {
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": INK["secondary"],
        },
        "title_font": {"color": INK["primary"]},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "xaxis": {
            "gridcolor": "#e1e0d9",
            "linecolor": "#c3c2b7",
            "zerolinecolor": "#c3c2b7",
            "tickcolor": "#c3c2b7",
            "showline": True,
            "ticks": "outside",
        },
        "yaxis": {
            "gridcolor": "#e1e0d9",
            "linecolor": "#c3c2b7",
            "zerolinecolor": "#c3c2b7",
            "tickcolor": "#c3c2b7",
            "showline": True,
            "ticks": "outside",
        },
        "hoverlabel": {"bgcolor": "white", "font_color": INK["primary"]},
    }
)
LOG_AXIS = {"type": "log", "dtick": 1, "exponentformat": "power"}


def plot_population_histogram(population: np.ndarray, bins: int = 100) -> go.Figure:
    mean, median = population.mean(), np.median(population)
    edges = np.logspace(np.log10(population.min()), np.log10(population.max()), bins + 1)
    counts, _ = np.histogram(population, bins=edges)

    fig = go.Figure(
        go.Bar(
            x=np.sqrt(edges[:-1] * edges[1:]),
            y=counts,
            width=np.diff(edges),
            marker={"color": MODEL_COLORS["data"], "line": {"color": SURFACE, "width": 1}},
            showlegend=False,
        )
    )
    for value, label, side in ((median, "median", "left"), (mean, "mean", "right")):
        fig.add_vline(x=value, line_color="black", line_dash="dash", line_width=2)
        fig.add_annotation(
            x=np.log10(value),  # annotations on a log axis take log10 coordinates
            y=1,
            yref="y domain",
            text=f" {label} {value:,.0f} ",
            font_color=INK["primary"],
            xanchor="right" if side == "left" else "left",
            yanchor="top",
            showarrow=False,
        )

    fig.update_xaxes(title_text="inhabitants (log scale)", **LOG_AXIS)
    fig.update_yaxes(title_text="number of municipalities")
    fig.update_layout(
        title=f"Distribution of municipality populations ({len(population):,} municipalities, logarithmic bins)",
        bargap=0,
        showlegend=False,
        height=450,
        width=900,
        template=REAL_WORLD_TEMPLATE,
    )
    return fig


def plot_rank_survival(
    sizes: np.ndarray, names: list[str], alpha: float, x_min: float
) -> go.Figure:
    n = len(sizes)
    rank = np.arange(1, n + 1)
    survival = rank / n  # P(X >= x)
    share = np.cumsum(sizes) / sizes.sum()  # share held by the top `rank`
    # the fit only describes x >= x_min, scale it by the tail's share to sit on the data
    x_fit = np.logspace(np.log10(x_min), np.log10(sizes[0]), 200)
    fitted = (sizes >= x_min).mean() * (x_fit / x_min) ** (1 - alpha)
    data = {
        "mode": "markers",
        "marker": {"size": 6, "color": MODEL_COLORS["data"], "opacity": 0.8},
        "text": names,
    }

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Zipf plot: size vs rank",
            "Survival function P(X ≥ x)",
            "Sum share of the top fraction",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=rank,
            y=sizes,
            name="municipalities",
            hovertemplate="%{text}<br>%{y:,}<extra></extra>",
            **data,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=survival,
            showlegend=False,
            hovertemplate="%{text}<br>%{x:,}<extra></extra>",
            **data,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=fitted,
            mode="lines",
            line={"color": MODEL_COLORS["power law"], "width": 2},
            name=f"power law, α = {alpha:.2f}",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=x_min,
        line_dash="dot",
        line_color=GUIDE,
        line_width=1,
        annotation_text=f"x_min = {x_min:,.0f}",
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=survival,
            y=share,
            mode="lines",
            line={"color": MODEL_COLORS["data"], "width": 2},
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dot", "color": GUIDE, "width": 1},
            name="equality",
        ),
        row=1,
        col=3,
    )
    fig.add_vline(x=0.2, row=1, col=3, **REFERENCE_LINE)
    fig.add_hline(y=0.8, row=1, col=3, **REFERENCE_LINE)

    fig.update_xaxes(title_text="rank (log scale)", row=1, col=1, **LOG_AXIS)
    fig.update_yaxes(title_text="inhabitants (log scale)", row=1, col=1, **LOG_AXIS)
    fig.update_xaxes(title_text="inhabitants x (log scale)", row=1, col=2, **LOG_AXIS)
    fig.update_yaxes(title_text="P(X ≥ x) (log scale)", row=1, col=2, **LOG_AXIS)
    fig.update_xaxes(
        title_text="top fraction of municipalities", tickformat=".0%", range=[0, 1], row=1, col=3
    )
    fig.update_yaxes(
        title_text="share of population", tickformat=".0%", range=[0, 1.02], row=1, col=3
    )
    fig.update_layout(
        title=f"Power law fit, α = {alpha:.2f} above x_min = {x_min:,.0f}<br><sup>top 20% of municipalities hold {share[int(0.2 * n) - 1]:.1%} of the population</sup>",
        height=450,
        width=1400,
        legend={"orientation": "h", "y": -0.2},
        template=REAL_WORLD_TEMPLATE,
    )
    return fig


def plot_tail_comparison(
    tail: np.ndarray, names: list[str], x_min: float, models: dict[str, dict]
) -> go.Figure:
    n = len(tail)
    survival = 1 - (np.arange(1, n + 1) - 0.5) / n  # plotting positions
    x_fit = np.logspace(np.log10(x_min), np.log10(tail[-1]), 300)

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.37, 0.37, 0.26],
        subplot_titles=(
            "Survival within the tail P(X ≥ x | X ≥ x_min)",
            "Q-Q plot: model quantile vs observed",
            "Total population of the tail",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=tail,
            y=survival,
            mode="markers",
            marker={"size": 7, "color": MODEL_COLORS["data"], "opacity": 0.8},
            name="municipalities",
            text=names,
            hovertemplate="%{text}<br>%{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for name, model in models.items():
        line = {"color": MODEL_COLORS[name], "width": 2}
        fig.add_trace(
            go.Scatter(
                x=x_fit, y=model["sf"](x_fit), mode="lines", line=line, name=name, legendgroup=name
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tail,
                y=model["isf"](survival),
                mode="markers",
                marker={"size": 7, "color": line["color"], "opacity": 0.6},
                legendgroup=name,
                showlegend=False,
                text=names,
                hovertemplate="%{text}<br>observed %{x:,}<br>"
                + name
                + " expects %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[x_min, tail[-1]],
            y=[x_min, tail[-1]],
            mode="lines",
            line={"dash": "dash", "color": GUIDE, "width": 1},
            name="perfect fit",
        ),
        row=1,
        col=2,
    )
    totals = {"observed": tail.sum()} | {name: model["total"] for name, model in models.items()}
    fig.add_trace(
        go.Bar(
            x=list(totals),
            y=list(totals.values()),
            marker={
                "color": [MODEL_COLORS.get(name, MODEL_COLORS["data"]) for name in totals],
                "line": {"color": SURFACE, "width": 2},
            },
            width=0.5,
            text=[
                f"{total / 1e6:.1f} M<br>({total / tail.sum():.0%})" for total in totals.values()
            ],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="inhabitants x (log scale)", row=1, col=1, **LOG_AXIS)
    fig.update_yaxes(
        title_text="P(X ≥ x | X ≥ x_min) (log scale)",
        range=[np.log10(0.5 / n), 0.05],
        row=1,
        col=1,
        **LOG_AXIS,
    )
    fig.update_xaxes(title_text="observed inhabitants (log scale)", row=1, col=2, **LOG_AXIS)
    fig.update_yaxes(
        title_text="inhabitants expected by the model (log scale)", row=1, col=2, **LOG_AXIS
    )
    fig.update_yaxes(
        title_text="inhabitants in the tail", range=[0, max(totals.values()) * 1.25], row=1, col=3
    )
    fig.update_layout(
        title=f"Power law vs normal on the {n:,} municipalities with x ≥ {x_min:,.0f}<br><sup>normal: μ, σ from all municipalities, conditioned on x ≥ x_min</sup>",
        height=480,
        width=1400,
        legend={"orientation": "h", "y": -0.2},
        template=REAL_WORLD_TEMPLATE,
    )
    return fig


def band(experiment: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        experiment.group_by("sample_size")
        .agg(pl.col(column).mean().alias("mean"), pl.col(column).std().alias("std"))
        .with_columns(low=pl.col("mean") - pl.col("std"), high=pl.col("mean") + pl.col("std"))
        .sort("sample_size")
    )


def add_band(fig: go.Figure, frame: pl.DataFrame, color: str, name: str, col: int) -> None:
    x = frame["sample_size"].to_list()
    rgb = ", ".join(str(int(color[i : i + 2], 16)) for i in (1, 3, 5))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["high"].to_list(),
            mode="lines",
            line_width=0,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["low"].to_list(),
            mode="lines",
            line_width=0,
            fill="tonexty",
            fillcolor=f"rgba({rgb}, 0.15)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["mean"].to_list(),
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "line": RING},
            name=name,
            showlegend=False,
            customdata=frame["std"].to_list(),
            hovertemplate="n = %{x:,}<br>mean %{y:,.3g} ± %{customdata:,.3g}<extra></extra>",
        ),
        row=1,
        col=col,
    )


SAMPLE_SIZE_AXIS = dict(title_text="sample size n (log scale)", **LOG_AXIS)


def plot_sample_parameters(experiment: pl.DataFrame, full_data: dict) -> go.Figure:
    repeats = experiment.filter(pl.col("sample_size") == experiment["sample_size"].min()).height
    parameters = (("alpha", "α"), ("x_min", "x_min"))
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[f"Fitted {label}" for _, label in parameters]
    )
    for col, (column, label) in enumerate(parameters, start=1):
        add_band(fig, band(experiment, column), MODEL_COLORS["power law"], label, col=col)
        fig.add_hline(
            y=full_data[column],
            annotation_text=f"full data: {full_data[column]:,.3g}",
            annotation_position="top right",
            row=1,
            col=col,
            **REFERENCE_LINE,
        )
        fig.update_yaxes(title_text=label, row=1, col=col)
    fig.update_xaxes(**SAMPLE_SIZE_AXIS)
    fig.update_layout(
        title=f"Power law fits on random samples ({repeats} per size)<br><sup>line: mean, band: ± 1 std, dashed: full data</sup>",
        height=450,
        width=1400,
        template=REAL_WORLD_TEMPLATE,
    )
    return fig
