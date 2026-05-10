# =============================================================================
# Paradox Explorer — Plotly chart helpers
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Base Rate Fallacy — Sankey & Waffle
# ═══════════════════════════════════════════════════════════════════════════════

def draw_baserate_sankey(tp, fp, fn, tn, dark=True):
    """Sankey diagram showing probability flow from population to test results."""
    fig = _base_fig(dark, height=450, margin=dict(l=24, r=24, t=16, b=16))
    th = _theme(dark)

    sick = tp + fn
    healthy = fp + tn
    total = sick + healthy
    pos = tp + fp
    neg = tn + fn

    # Use integer labels when close to whole numbers, otherwise 1 decimal
    def _fmt(v):
        return f"{v:,.0f}" if v >= 1 else f"{v:,.1f}"

    labels = [
        f"Population ({_fmt(total)})",
        f"Sick ({_fmt(sick)})",
        f"Healthy ({_fmt(healthy)})",
        f"Test + ({_fmt(pos)})",
        f"Test \u2212 ({_fmt(neg)})",
    ]

    c_pop     = "#64748b"
    c_sick    = "#ef4444"
    c_healthy = "#3b82f6"
    c_pos     = "#f97316"
    c_neg     = "#10b981"

    # Custom link labels
    link_labels = [
        f"Sick: {_fmt(sick)}",
        f"Healthy: {_fmt(healthy)}",
        f"TP: {_fmt(tp)}",
        f"FN: {_fmt(fn)}",
        f"FP: {_fmt(fp)}",
        f"TN: {_fmt(tn)}",
    ]

    fig.add_trace(go.Sankey(
        node=dict(
            pad=25, thickness=30,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=labels,
            color=[c_pop, c_sick, c_healthy, c_pos, c_neg],
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 3, 4],
            value=[max(sick, 0.01), max(healthy, 0.01),
                   max(tp, 0.01), max(fn, 0.01),
                   max(fp, 0.01), max(tn, 0.01)],
            label=link_labels,
            color=[
                "rgba(239, 68, 68, 0.3)",    # pop → sick
                "rgba(59, 130, 246, 0.25)",  # pop → healthy
                "rgba(239, 68, 68, 0.65)",   # sick → pos (TP) - Red
                "rgba(168, 85, 247, 0.65)",  # sick → neg (FN) - Purple
                "rgba(59, 130, 246, 0.65)",  # healthy → pos (FP) - Blue
                "rgba(100, 116, 139, 0.25)", # healthy → neg (TN) - Muted Grey
            ],
        ),
    ))

    fig.update_layout(font=dict(size=13, color=th["label"]))
    return fig


def draw_baserate_waffle(tp, fp, fn, tn, dark=True):
    """Icon Array (waffle chart) for Base Rate Fallacy.

    Uses a 50×50 grid = 2,500 dots.  Scales input counts proportionally.
    Places the small categories (TP, FP, FN) first so they appear in the
    top-left corner and remain visible even at very low prevalence.
    """
    GRID = 30   # 30×30 = 900 dots — larger and more readable
    N = GRID * GRID

    fig = _base_fig(dark, height=None, margin=dict(l=8, r=8, t=8, b=8))
    th = _theme(dark)

    total = tp + fp + fn + tn
    if total == 0:
        return fig

    # ── Scale to N dots ──────────────────────────────────────────────────
    scale = N / total
    s_tp = max(int(round(tp * scale)), 1 if tp > 0 else 0)
    s_fp = max(int(round(fp * scale)), 1 if fp > 0 else 0)
    s_fn = max(int(round(fn * scale)), 1 if fn > 0 else 0)
    s_tn = N - (s_tp + s_fp + s_fn)
    s_tn = max(0, s_tn)

    # ── Assign each dot a category (TP first → top-left) ────────────────
    cat_ids = np.zeros(N, dtype=int)   # 0 = TN by default
    idx = 0
    for cat_id, cnt in [(1, s_tp), (2, s_fp), (3, s_fn)]:
        cat_ids[idx:idx + cnt] = cat_id
        idx += cnt

    # ── Grid coordinates (row-major, top-left origin) ────────────────────
    col, row = np.meshgrid(np.arange(GRID), np.arange(GRID - 1, -1, -1))
    xs = col.flatten()
    ys = row.flatten()

    # ── Colours and labels ───────────────────────────────────────────────
    spec = [
        (0, '#334155' if dark else '#cbd5e1',
         f'Healthy, Test \u2212 (TN: {tn:,.0f})'),
        (1, '#ef4444',  # Red
         f'Sick, Test + (TP: {tp:,.0f})'),
        (2, '#3b82f6',  # Blue (distinct from Red/Pink)
         f'Healthy, Test + (FP: {fp:,.0f})'),
        (3, '#a855f7',  # Purple
         f'Sick, Test \u2212 (FN: {fn:,.0f})'),
    ]

    for cat_id, color, label in spec:
        mask = cat_ids == cat_id
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask], mode="markers",
            marker=dict(symbol="square", size=12,
                        color=color, line=dict(width=0)),
            name=label, hoverinfo="name",
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1, GRID + 1]),
        yaxis=dict(visible=False, range=[-1, GRID + 1],
                   scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Coupon Collector's Problem ───────────────────────────────────────────────

def draw_ccp_curve(N, B, expected_packs, dark=True):
    """Hockey-stick: expected packs to collect the k-th unique item.

    P(pack has ≥1 new | k collected) = 1 − C(k,B)/C(N,B)
    Computed via log-ratio to avoid huge integer overflow for large N,B.
    """
    fig = _base_fig(dark, height=None, margin=dict(l=52, r=16, t=24, b=40))
    th = _theme(dark)

    ks = np.arange(0, N)  # k items already collected; seeking (k+1)-th

    # log C(k,B)/C(N,B) = sum_{i=0}^{B-1} log(k-i) - log(N-i)  [only for k >= B]
    log_ratio = np.full(N, -np.inf)  # log(0) for k < B
    for k in range(B, N):
        lr = 0.0
        for i in range(B):
            lr += np.log(k - i) - np.log(N - i)
        log_ratio[k] = lr

    p_new = 1.0 - np.exp(log_ratio)
    p_new = np.clip(p_new, 1e-12, 1.0)
    marginal = 1.0 / p_new  # expected packs to get the next new item

    # Cumulative: expected total packs to collect exactly k items
    cumulative = np.cumsum(marginal)

    # Shade area under marginal curve
    fig.add_trace(go.Scatter(
        x=np.concatenate([[1], ks[1:] + 1, [N]]),
        y=np.concatenate([[marginal[0]], marginal[1:], [marginal[-1]]]),
        mode="none",
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.08)",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ks + 1,
        y=marginal,
        mode="lines",
        line=dict(color="#0ea5e9", width=2.5),
        name="Packs per new item",
        hovertemplate="Item #%{x}<br>Expected packs: %{y:.2f}<extra></extra>",
    ))

    # Mark N/2 — halfway point
    half = N // 2
    fig.add_vline(
        x=half, line_dash="dash", line_color="#10b981", opacity=0.6,
        annotation_text=f"50% ({marginal[half - 1]:.1f} packs/item)",
        annotation_position="top left",
        annotation_font_color="#10b981",
        annotation_font_size=10,
    )

    # Mark last item cost = N/B
    last_cost = N / B
    fig.add_annotation(
        x=N, y=marginal[-1],
        text=f"Last item: {last_cost:.1f} packs",
        showarrow=True, arrowhead=2, arrowcolor="#f59e0b",
        font=dict(size=10, color="#f59e0b"),
        xanchor="right", yanchor="bottom",
        ax=-40, ay=-30,
    )

    fig.update_layout(
        xaxis_title="Unique items collected",
        yaxis_title="Expected packs to get next item",
        showlegend=False,
    )
    return fig


def draw_ccp_distribution(sim_results, expected_packs, dark=True):
    fig = _base_fig(dark, height=None)
    th = _theme(dark)
    
    fig.add_trace(go.Histogram(
        x=sim_results,
        nbinsx=50,
        marker_color="#10b981",
        name="Simulated Packs",
        hovertemplate="Packs: %{x}<br>Frequency: %{y}<extra></extra>"
    ))
    
    fig.add_vline(
        x=expected_packs, line_dash="dash", line_color="#f59e0b", line_width=2,
        annotation_text=f"Mean: {expected_packs:.1f}", annotation_position="top right",
        annotation_font_color="#f59e0b"
    )
    
    fig.update_layout(
        xaxis_title="Total Packs to Complete Set",
        yaxis_title="Number of Collectors",
        showlegend=False,
        bargap=0.05
    )
    return fig


def draw_ccp_cdf(sim_results, expected_packs, dark=True):
    fig = _base_fig(dark, height=None)
    th = _theme(dark)
    
    # Calculate ECDF
    x = np.sort(sim_results)
    y = np.arange(1, len(x) + 1) / len(x)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#a855f7", width=3),
        name="Probability",
        hovertemplate="Packs: %{x}<br>Probability: %{y:.1%}<extra></extra>"
    ))
    
    fig.add_vline(
        x=expected_packs, line_dash="dash", line_color="#f59e0b", line_width=2,
        annotation_text=f"E[Packs] = {expected_packs:.1f}",
        annotation_position="bottom right",
        annotation_font_color="#f59e0b",
    )
    
    fig.update_layout(
        xaxis_title="Total Packs to Complete Set",
        yaxis_title="Cumulative Probability",
        yaxis_tickformat=".0%",
        showlegend=False,
    )
    return fig


# ── Gambler's Fallacy ───────────────────────────────────────────────────────

_C_SUCCESS  = "#10b981"   # green  — successes
_C_FAIL     = "#ef4444"   # red    — failures
_C_STREAK   = "#f59e0b"   # amber  — highlighted streak
_C_FALLACY  = "#a855f7"   # violet — gambler's (wrong) belief


def _find_streaks(flips, k):
    """Return list of (start, end) for maximal runs of 1s with length >= k.
    Indices are 0-based, end is exclusive (Python slice convention)."""
    arr = np.asarray(flips, dtype=int)
    if arr.size == 0:
        return []
    padded = np.concatenate([[0], arr, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= k]


def draw_gf_sequence(flips, p, k, dark=True):
    """Single run as barcode strip + running proportion (LLN view)."""
    th = _theme(dark)
    flips = np.asarray(flips, dtype=int)
    n = int(len(flips))
    xs = np.arange(1, n + 1)
    running = np.cumsum(flips) / xs

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.04,
    )

    # Top: running proportion line
    fig.add_trace(go.Scatter(
        x=xs, y=running, mode="lines",
        line=dict(color="#0ea5e9", width=2),
        name="Running proportion",
        hovertemplate="Flip %{x}<br>Running prop: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # True p reference line on top subplot
    fig.add_hline(
        y=p, line_dash="dash", line_color=_C_FAIL, opacity=0.7,
        annotation_text=f"True p = {p:.3f}",
        annotation_position="top left",
        annotation_font_color=_C_FAIL,
        annotation_font_size=10,
        row=1, col=1,
    )

    # Bottom: 1D barcode heatmap
    fig.add_trace(go.Heatmap(
        z=[flips], x=xs,
        colorscale=[[0, _C_FAIL], [1, _C_SUCCESS]],
        showscale=False, zmin=0, zmax=1,
        xgap=0, ygap=0,
        hovertemplate="Flip %{x}: %{z}<extra></extra>",
    ), row=2, col=1)

    # Streak highlights — span both rows via yref="paper"
    streaks = _find_streaks(flips, k)
    for s, e in streaks:
        fig.add_shape(
            type="rect",
            x0=s + 0.5, x1=e + 0.5,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=_C_STREAK, opacity=0.18,
            layer="below", line_width=0,
        )

    # Annotation: arrow at last streak
    if streaks:
        last_s, last_e = streaks[-1]
        midx = (last_s + last_e) / 2 + 0.5
        run_height = float(running[min(last_e - 1, n - 1)])
        fig.add_annotation(
            x=midx, y=run_height,
            xref="x", yref="y",
            text=f"streak of {last_e - last_s}",
            showarrow=True, arrowhead=2, arrowcolor=_C_STREAK,
            font=dict(size=10, color=_C_STREAK),
            ax=0, ay=-30,
        )

    # Final-proportion text
    fig.add_annotation(
        x=0.99, y=1.05, xref="paper", yref="paper",
        text=f"Final proportion: {running[-1]:.3f}  ·  True p: {p:.3f}",
        showarrow=False,
        font=dict(size=10, color=th["label"]),
        xanchor="right", yanchor="bottom",
    )

    # Apply shared theme manually (make_subplots bypasses _base_fig)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=th["label"], size=11),
        margin=dict(l=48, r=16, t=28, b=40),
        showlegend=False, dragmode=False,
    )
    fig.update_xaxes(
        gridcolor=th["grid"], linecolor=th["grid"],
        tickfont=dict(size=10, color=th["axis"]),
        zeroline=False, row=1, col=1,
    )
    fig.update_xaxes(
        gridcolor=th["grid"], linecolor=th["grid"],
        tickfont=dict(size=10, color=th["axis"]),
        zeroline=False, title_text="Flip number",
        row=2, col=1,
    )
    fig.update_yaxes(
        gridcolor=th["grid"], linecolor=th["grid"],
        tickfont=dict(size=10, color=th["axis"]),
        zeroline=False, title_text="Running proportion",
        range=[0, 1], row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, zeroline=False,
        range=[-0.5, 0.5], row=2, col=1,
    )
    return fig


def draw_gf_after_k(after_k, p, k_target, dark=True):
    """Bar chart of P(next | last >= k) — opacity scales with sample size."""
    fig = _base_fig(dark, height=None,
                    margin=dict(l=52, r=16, t=24, b=44))
    th = _theme(dark)

    if not after_k:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No data — increase n or decrease p",
            showarrow=False, font=dict(size=12, color=th["label"]),
        )
        return fig

    ks = sorted(after_k.keys())
    ns = np.array([len(after_k[kk]) for kk in ks], dtype=int)
    n_max = int(ns.max()) if (ns.size and ns.max() > 0) else 1

    bars_x, bars_y, bars_err, bars_op = [], [], [], []
    bars_line_color, bars_line_width = [], []
    hover_text, na_x = [], []

    for kk in ks:
        arr = after_k[kk]
        n = int(len(arr))
        if n == 0:
            bars_x.append(kk); bars_y.append(p); bars_err.append(0)
            bars_op.append(0.10)
            bars_line_color.append("rgba(120,120,120,0.4)")
            bars_line_width.append(0.5)
            hover_text.append(f"k={kk}: no data")
            na_x.append(kk)
            continue
        bars_x.append(kk)
        bars_y.append(float(arr.mean()))
        bars_err.append(2.0 * float(np.sqrt(p * (1 - p) / n)))
        bars_op.append(0.30 + 0.70 * (n / n_max))
        if kk == k_target:
            bars_line_color.append(_C_STREAK)
            bars_line_width.append(2.5)
        else:
            bars_line_color.append("rgba(0,0,0,0)")
            bars_line_width.append(0)
        hover_text.append(
            f"k={kk} · n={n} cases · mean={arr.mean():.3f}"
        )

    fig.add_trace(go.Bar(
        x=bars_x, y=bars_y,
        error_y=dict(type="data", array=bars_err,
                     color=th["label"], width=4, thickness=1.2),
        marker=dict(
            color=_C_SUCCESS,
            opacity=bars_op,
            line=dict(color=bars_line_color, width=bars_line_width),
        ),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    # ±2 SE band around p (uses average n of non-empty bars)
    nonzero = ns[ns > 0]
    if nonzero.size:
        avg_n = float(nonzero.mean())
        se_band = 2.0 * float(np.sqrt(p * (1 - p) / avg_n))
        fig.add_hrect(
            y0=max(0.0, p - se_band), y1=min(1.0, p + se_band),
            fillcolor=_C_FAIL, opacity=0.10,
            layer="below", line_width=0,
        )

    # True-p reference line
    fig.add_hline(
        y=p, line_dash="dash", line_color=_C_FAIL, opacity=0.85,
        annotation_text=f"True p = {p:.3f}",
        annotation_position="top left",
        annotation_font_color=_C_FAIL,
        annotation_font_size=10,
    )

    # N/A annotations
    for kk in na_x:
        fig.add_annotation(
            x=kk, y=0.04, xref="x", yref="paper",
            text="N/A", showarrow=False,
            font=dict(size=9, color=th["axis"]),
        )

    fig.update_layout(
        xaxis=dict(title="Streak length k (last k or more were successes)",
                   dtick=1, range=[0.5, max(ks) + 0.5]),
        yaxis=dict(title="P(next success | last ≥ k)",
                   range=[0, 1]),
        showlegend=False, bargap=0.25,
    )
    return fig


def draw_gf_streaks(max_streaks, p, n_flips, dark=True):
    """Histogram of longest streak per simulation."""
    fig = _base_fig(dark, height=None,
                    margin=dict(l=52, r=16, t=24, b=44))
    th = _theme(dark)

    if len(max_streaks) == 0:
        return fig

    if 0 < p < 1 and n_flips > 1:
        expected = max(1, int(np.log(n_flips * (1 - p)) / np.log(1 / p)))
    else:
        expected = 1

    max_val = int(max_streaks.max())
    fig.add_trace(go.Histogram(
        x=max_streaks,
        xbins=dict(start=-0.5, end=max_val + 0.5, size=1),
        marker_color=_C_FALLACY, opacity=0.78,
        marker_line=dict(color=th["annot_border"], width=0.5),
        hovertemplate="Longest streak: %{x}<br>Sims: %{y}<extra></extra>",
        showlegend=False,
    ))

    # Theoretical expectation line
    fig.add_vline(
        x=expected, line_dash="dash", line_color=_C_STREAK, line_width=2,
        annotation_text=f"E[L_n] ≈ {expected}",
        annotation_position="top right",
        annotation_font_color=_C_STREAK,
    )

    # 5th–95th percentile range
    p5 = int(np.percentile(max_streaks, 5))
    p95 = int(np.percentile(max_streaks, 95))
    if p5 < p95:
        fig.add_vline(x=p5, line_dash="dot", line_color=th["axis"], opacity=0.5)
        fig.add_vline(x=p95, line_dash="dot", line_color=th["axis"], opacity=0.5)

    p10_threshold = int(np.percentile(max_streaks, 10))
    pct_above = int(round((max_streaks >= p10_threshold).mean() * 100))

    annot_text = (
        f"In {n_flips} flips · p = {p:.3f}<br>"
        f"Expected longest ≈ {expected}<br>"
        f"5th–95th percentile: [{p5}, {p95}]<br>"
        f"{pct_above}% of runs had streak ≥ {p10_threshold}"
    )
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=annot_text, showarrow=False,
        font=dict(size=10, color=th["label"]),
        xanchor="right", yanchor="top",
        bgcolor=th["annot_bg"],
        bordercolor=th["annot_border"],
        borderwidth=1, borderpad=6,
        align="left",
    )

    fig.update_layout(
        xaxis=dict(title="Longest streak of successes per run",
                   dtick=max(1, max_val // 12)),
        yaxis=dict(title="Number of simulations"),
        showlegend=False, bargap=0.05,
    )
    return fig
