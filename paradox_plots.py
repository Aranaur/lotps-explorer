# =============================================================================
# Paradox Explorer — Plotly chart helpers
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.spatial import cKDTree

from theme import _base_fig, _theme

# ── colour tokens ──────────────────────────────────────────────────────────────
_C_POINTS   = "#38bdf8"   # cyan   — scatter points
_C_CLUSTER  = "#f97316"   # orange — clustered points
_C_GRID     = "#475569"   # slate  — grid lines
_C_GRID_LT  = "#94a3b8"   # lighter slate for light mode
_C_OBSERVED = "#38bdf8"   # cyan   — observed histogram bars
_C_EXPECTED = "#f87171"   # red    — expected (Poisson) line
_C_NN_HIST  = "#a78bfa"   # violet — NN distance histogram
_C_NN_THEO  = "#34d399"   # green  — theoretical NN curve
_C_GOLD     = "#fbbf24"   # gold   — annotations

# Simpson's Paradox — group palette (up to 6 groups)
_GROUP_COLORS = [
    "#38bdf8",  # cyan
    "#f97316",  # orange
    "#a78bfa",  # violet
    "#34d399",  # green
    "#f87171",  # red
    "#fbbf24",  # gold
]
_C_OVERALL = "#ef4444"    # bright red — overall trendline


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Main 2D scatter plot  (Clustering Illusion)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_cluster_scatter(x, y, n_points, title,
                         show_grid=False, grid_k=5, dark=True):
    """2D scatter of the points with optional quadrat grid overlay."""
    fig = _base_fig(dark,
                    height=None,
                    margin=dict(l=40, r=12, t=16, b=24))
    th = _theme(dark)

    marker_col = _C_POINTS

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=marker_col, size=5, opacity=0.65,
            line=dict(width=0),
        ),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))

    # Grid overlay
    if show_grid:
        grid_col = _C_GRID if dark else _C_GRID_LT
        for i in range(1, grid_k):
            val = i / grid_k
            fig.add_shape(
                type="line", x0=val, x1=val, y0=0, y1=1,
                line=dict(color=grid_col, width=0.8, dash="dot"),
            )
            fig.add_shape(
                type="line", x0=0, x1=1, y0=val, y1=val,
                line=dict(color=grid_col, width=0.8, dash="dot"),
            )

    fig.add_annotation(
        x=0.5, y=1.02, xref="paper", yref="paper",
        text=f"{title}  ·  N = {n_points}",
        showarrow=False,
        font=dict(size=12, color=th["label"]),
        xanchor="center",
    )

    fig.update_layout(
        xaxis=dict(range=[0, 1], constrain="domain",
                   title="x", dtick=0.2),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1,
                   title="y", dtick=0.2),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Quadrat analysis bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def draw_quadrat_chart(x, y, grid_k=5, dark=True):
    """Bar chart of observed counts per quadrat vs expected Poisson distribution."""
    fig = _base_fig(dark, height=None, margin=dict(l=40, r=12, t=8, b=30))
    th = _theme(dark)
    n = len(x)
    n_cells = grid_k ** 2

    # Count points in each cell
    ix = np.clip((x * grid_k).astype(int), 0, grid_k - 1)
    iy = np.clip((y * grid_k).astype(int), 0, grid_k - 1)
    cell_ids = iy * grid_k + ix
    counts = np.bincount(cell_ids, minlength=n_cells)

    # Observed frequency of counts
    max_count = int(counts.max()) + 1
    obs_freq = np.bincount(counts, minlength=max_count + 1)
    bins = np.arange(len(obs_freq))

    # Expected Poisson frequencies
    lam = n / n_cells
    expected = stats.poisson.pmf(bins, lam) * n_cells

    fig.add_trace(go.Bar(
        x=bins, y=obs_freq,
        marker_color=_C_OBSERVED, opacity=0.75,
        name="Observed",
    ))
    fig.add_trace(go.Scatter(
        x=bins, y=expected, mode="lines+markers",
        line=dict(color=_C_EXPECTED, width=2.5),
        marker=dict(color=_C_EXPECTED, size=6),
        name=f"Poisson(\u03bb={lam:.1f})",
    ))

    # CSR test: χ² of observed vs Poisson
    exp_counts = stats.poisson.pmf(np.arange(max_count + 1), lam) * n_cells
    # Pool small expected cells
    mask = exp_counts >= 1
    if mask.sum() >= 2:
        o = obs_freq[mask].astype(float)
        e = exp_counts[mask]
        # Pool remainder
        o_rest = obs_freq[~mask].sum()
        e_rest = exp_counts[~mask].sum()
        if e_rest > 0:
            o = np.append(o, o_rest)
            e = np.append(e, e_rest)
        chi2 = float(np.sum((o - e) ** 2 / e))
        dof = max(len(o) - 1, 1)
        p_val = float(1 - stats.chi2.cdf(chi2, dof))
        verdict = "consistent with CSR" if p_val > 0.05 else "departs from CSR"
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"\u03c7\u00b2={chi2:.1f}, p={p_val:.3f}<br><i>{verdict}</i>",
            showarrow=False,
            font=dict(size=10, color=th["label"]),
            xanchor="right", yanchor="top",
            bgcolor=th["annot_bg"],
            bordercolor=th["annot_border"],
            borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        xaxis_title="Points per cell",
        yaxis_title="Number of cells",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Nearest-neighbour distance distribution
# ═══════════════════════════════════════════════════════════════════════════════

def draw_nn_distance(x, y, n_points, dark=True):
    """Histogram of nearest-neighbour distances vs theoretical CSR distribution."""
    fig = _base_fig(dark, height=None, margin=dict(l=40, r=12, t=8, b=30))
    th = _theme(dark)

    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]  # skip self (distance=0)

    # Theoretical NN distance CDF for 2D Poisson process on unit square:
    #   F(r) = 1 − exp(−λ π r²),  where λ = intensity = N/area = N
    lam = n_points  # unit square area = 1
    r_max = float(nn_dists.max()) * 1.2
    r_th = np.linspace(0, r_max, 200)
    pdf_th = 2 * lam * np.pi * r_th * np.exp(-lam * np.pi * r_th ** 2)

    # Histogram of observed NN distances
    fig.add_trace(go.Histogram(
        x=nn_dists, nbinsx=30,
        marker_color=_C_NN_HIST, opacity=0.65,
        name="Observed",
        histnorm="probability density",
    ))

    # Theoretical curve
    fig.add_trace(go.Scatter(
        x=r_th, y=pdf_th, mode="lines",
        line=dict(color=_C_NN_THEO, width=2.5),
        name="Theoretical (CSR)",
    ))

    # Clark-Evans R statistic
    mean_obs = float(nn_dists.mean())
    mean_exp = 0.5 / np.sqrt(lam)  # E[r] for 2D Poisson
    se = 0.26136 / np.sqrt(n_points * lam)  # SE for Clark–Evans
    R = mean_obs / mean_exp if mean_exp > 0 else 1.0
    z = (mean_obs - mean_exp) / se if se > 0 else 0.0
    p_ce = float(2 * (1 - stats.norm.cdf(abs(z))))

    if R < 1:
        pattern = "clustered"
    elif R > 1:
        pattern = "dispersed"
    else:
        pattern = "random"

    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=(f"Clark\u2013Evans R = {R:.3f}<br>"
              f"z = {z:.2f}, p = {p_ce:.3f}<br>"
              f"<i>Pattern: {pattern}</i>"),
        showarrow=False,
        font=dict(size=10, color=th["label"]),
        xanchor="right", yanchor="top",
        bgcolor=th["annot_bg"],
        bordercolor=th["annot_border"],
        borderwidth=1, borderpad=4,
    )

    fig.update_layout(
        xaxis_title="Nearest-neighbour distance",
        yaxis_title="Density",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Simpson's Paradox — scatter plot
# ═══════════════════════════════════════════════════════════════════════════════

def generate_simpsons_data(n_per_group, k_groups, noise, rng=None):
    """Generate synthetic Simpson's Paradox data.

    Each group has a *positive* within-group slope, but the group centres
    are arranged so that the overall (pooled) slope is *negative*.

    Returns
    -------
    x, y : 1-D arrays
    group : 1-D int array  (0 … k-1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Group centres: ascending x, descending y → negative overall slope
    cx = np.linspace(2, 8, k_groups)
    cy = np.linspace(8, 2, k_groups)

    within_slope = 0.6  # positive slope inside each group

    xs, ys, gs = [], [], []
    for g in range(k_groups):
        x_g = rng.normal(cx[g], 1.0, n_per_group)
        # y = cy + slope * (x - cx) + noise
        y_g = cy[g] + within_slope * (x_g - cx[g]) + rng.normal(0, noise, n_per_group)
        xs.append(x_g)
        ys.append(y_g)
        gs.append(np.full(n_per_group, g, dtype=int))

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def draw_simpsons_scatter(x, y, group, show_groups=False,
                          show_trends=False, dark=True):
    """Interactive scatter for Simpson's Paradox."""
    fig = _base_fig(dark, height=None,
                    margin=dict(l=48, r=16, t=32, b=40))
    th = _theme(dark)
    k = int(group.max()) + 1

    if show_groups:
        # Colour-coded by group
        for g in range(k):
            mask = group == g
            col = _GROUP_COLORS[g % len(_GROUP_COLORS)]
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(color=col, size=6, opacity=0.6,
                            line=dict(width=0)),
                name=f"Group {g + 1}",
                hovertemplate=(f"Group {g + 1}<br>"
                               "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"),
            ))

            # Per-group trendline
            if show_trends:
                xg, yg = x[mask], y[mask]
                slope, intercept = np.polyfit(xg, yg, 1)
                x_line = np.array([xg.min(), xg.max()])
                fig.add_trace(go.Scatter(
                    x=x_line, y=intercept + slope * x_line, mode="lines",
                    line=dict(color=col, width=2.5),
                    name=f"Group {g + 1} trend (\u03b2={slope:+.2f})",
                    showlegend=True,
                ))
    else:
        # Single colour — confound hidden
        muted_col = "#94a3b8" if dark else "#64748b"
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=muted_col, size=6, opacity=0.55,
                        line=dict(width=0)),
            name="All data",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        ))

    # Overall trendline (always drawn when trends are on)
    if show_trends:
        slope_all, intercept_all = np.polyfit(x, y, 1)
        x_range = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=x_range, y=intercept_all + slope_all * x_range, mode="lines",
            line=dict(color=_C_OVERALL, width=3, dash="dash"),
            name=f"Overall trend (\u03b2={slope_all:+.2f})",
            showlegend=True,
        ))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
    )
    return fig
