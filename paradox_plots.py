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


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Main 2D scatter plot
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
        name=f"Poisson(λ={lam:.1f})",
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
            text=f"χ²={chi2:.1f}, p={p_val:.3f}<br><i>{verdict}</i>",
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
        text=(f"Clark–Evans R = {R:.3f}<br>"
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
