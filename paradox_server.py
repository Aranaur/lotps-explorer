# =============================================================================
# Paradox Explorer — server logic
#
#   Tab 1  Base Rate Fallacy     (pdx_brf_*)
#   Tab 2  Clustering Illusion   (pdx_clust_*)
#   Tab 3  Simpson's Paradox     (pdx_simp_*)
#   Tab 4  Coupon Collector       (pdx_ccp_*)
# =============================================================================

import math
import random
from fractions import Fraction

import numpy as np
from shiny import reactive, render, ui

from utils import tip
from paradox_plots import (
    draw_cluster_scatter, draw_quadrat_chart, draw_nn_distance,
    generate_simpsons_data, draw_simpsons_scatter,
    draw_baserate_sankey, draw_baserate_waffle,
    draw_ccp_curve, draw_ccp_distribution, draw_ccp_cdf,
    draw_gf_sequence, draw_gf_after_k, draw_gf_streaks,
)
from theme import fig_to_ui


# ── Gambler's Fallacy — module-level helpers ──────────────────────────────
_GF_N_SIMS = 1_000


def _compute_max_streaks(data):
    """Vectorised max consecutive-1 streak per row."""
    N, _ = data.shape
    padded = np.concatenate(
        [np.zeros((N, 1), int), data.astype(int), np.zeros((N, 1), int)],
        axis=1,
    )
    diffs = np.diff(padded, axis=1)
    max_s = np.zeros(N, dtype=int)
    for i in range(N):
        starts = np.where(diffs[i] == 1)[0]
        ends = np.where(diffs[i] == -1)[0]
        if len(starts):
            max_s[i] = int((ends - starts).max())
    return max_s


def _expected_max_streak(p, n):
    """E[L_n] ≈ floor(log(n*(1-p)) / log(1/p)) for p in (0,1)."""
    if p <= 0 or p >= 1 or n < 2:
        return 1
    return max(1, int(np.log(n * (1 - p)) / np.log(1 / p)))


def paradox_server(input, output, session, is_dark):

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Base Rate Fallacy
    # ═════════════════════════════════════════════════════════════════════════

    @reactive.calc
    def _brf_stats():
        try: pop = int(input.pdx_brf_pop())
        except Exception: pop = 10000

        try: prev = float(input.pdx_brf_prev()) / 100.0
        except Exception: prev = 0.01

        try: sens = float(input.pdx_brf_sens()) / 100.0
        except Exception: sens = 0.99

        try: spec = float(input.pdx_brf_spec()) / 100.0
        except Exception: spec = 0.95

        sick = pop * prev
        healthy = pop - sick

        tp = sick * sens
        fn = sick - tp
        tn = healthy * spec
        fp = healthy - tn

        total_pos = tp + fp
        ppv = tp / total_pos if total_pos > 0 else 0

        return {
            "pop": pop, "prev": prev, "sens": sens, "spec": spec,
            "sick": sick, "healthy": healthy,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "total_pos": total_pos, "ppv": ppv
        }

    @render.ui
    def pdx_brf_stats_row():
        s = _brf_stats()
        return ui.div(
            ui.div(
                ui.div("P(DISEASE | TEST +)", class_="stat-label"),
                ui.div(f"{s['ppv'] * 100:.1f}%", class_="stat-value total"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("TOTAL POSITIVE TESTS", class_="stat-label"),
                ui.div(f"{s['total_pos']:,.0f}", class_="stat-value coverage"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("TRUE POSITIVES (SICK)", class_="stat-label"),
                ui.div(f"{s['tp']:,.0f}", class_="stat-value included"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("FALSE POSITIVES (HEALTHY)", class_="stat-label"),
                ui.div(f"{s['fp']:,.0f}", class_="stat-value missed"),
                class_="stat-card",
            ),
            class_="stats-row",
        )

    @render.ui
    def pdx_brf_formula():
        s = _brf_stats()
        math = (
            f"$$\\begin{{aligned}}"
            f"P(D|+) &= \\frac{{P(+|D)P(D)}}{{P(+|D)P(D) + P(+|\\neg D)P(\\neg D)}} \\\\[1ex]"
            f"&= \\frac{{{s['sens']:.3f} \\times {s['prev']:.4f}}}"
            f"{{{s['sens']:.3f} \\times {s['prev']:.4f} + {(1-s['spec']):.3f} \\times {(1-s['prev']):.4f}}} \\\\[1ex]"
            f"&\\approx {s['ppv']*100:.1f}\\%"
            f"\\end{{aligned}}$$"
        )
        return ui.HTML(math)

    _brf_active_preset = reactive.value("none")

    @reactive.effect
    @reactive.event(input.pdx_brf_pre_disease)
    def _brf_pre_disease():
        ui.update_numeric("pdx_brf_prev", value=1.0)
        ui.update_slider("pdx_brf_sens", value=99)
        ui.update_slider("pdx_brf_spec", value=95)
        _brf_active_preset.set("disease")

    @reactive.effect
    @reactive.event(input.pdx_brf_pre_terror)
    def _brf_pre_terror():
        ui.update_numeric("pdx_brf_prev", value=0.01)
        ui.update_slider("pdx_brf_sens", value=99)
        ui.update_slider("pdx_brf_spec", value=99)
        _brf_active_preset.set("terror")

    @reactive.effect
    @reactive.event(input.pdx_brf_pre_quality)
    def _brf_pre_quality():
        ui.update_numeric("pdx_brf_prev", value=5.0)
        ui.update_slider("pdx_brf_sens", value=95)
        ui.update_slider("pdx_brf_spec", value=90)
        _brf_active_preset.set("quality")

    @reactive.effect
    @reactive.event(input.pdx_brf_pre_lottery)
    def _brf_pre_lottery():
        ui.update_numeric("pdx_brf_prev", value=0.001)
        ui.update_slider("pdx_brf_sens", value=100)
        ui.update_slider("pdx_brf_spec", value=99)
        _brf_active_preset.set("lottery")

    @reactive.effect
    @reactive.event(input.pdx_brf_prev, input.pdx_brf_sens, input.pdx_brf_spec)
    def _brf_clear_preset():
        # Clear description if user manually changes sliders
        pass  # We could clear _brf_active_preset here but it's simpler to just let it stay

    @render.ui
    def pdx_brf_preset_desc():
        p = _brf_active_preset()
        if p == "none":
            return ui.div(
                "\u2190 Select an example to auto-fill the values.",
                class_="np-preset-hint",
            )
        elif p == "disease":
            return ui.div(
                ui.tags.strong("Rare Disease Screening: "),
                "Even with a 99% sensitive test, a 1% prevalence means most positive results are false alarms.",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "terror":
            return ui.div(
                ui.tags.strong("Facial Recognition: "),
                "Searching for 1 terrorist in 10,000 people. A 99% specific AI will flag 100 innocent people for every 1 guilty person.",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "quality":
            return ui.div(
                ui.tags.strong("Quality Control: "),
                "Checking factory parts. 5% are defective. The scanner is decent, but many good parts are thrown away (False Positives).",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "lottery":
            return ui.div(
                ui.tags.strong("Lottery Fraud: "),
                "Detecting a 1 in 100,000 event. The system must be practically perfect (99.999% TNR) to avoid being flooded with false alarms.",
                class_="np-preset-hint np-preset-hint--active"
            )
        return None

    @render.ui
    def pdx_brf_chart_area():
        s = _brf_stats()
        try: view = input.pdx_brf_view()
        except Exception: view = "Icon Array (Waffle)"

        if "Waffle" in view:
            fig = draw_baserate_waffle(s['tp'], s['fp'], s['fn'], s['tn'], dark=is_dark())
        else:
            fig = draw_baserate_sankey(s['tp'], s['fp'], s['fn'], s['tn'], dark=is_dark())
        return fig_to_ui(fig)


    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Clustering Illusion
    # ═════════════════════════════════════════════════════════════════════════

    _points_a_x = reactive.value(np.array([]))
    _points_a_y = reactive.value(np.array([]))
    _points_b_x = reactive.value(np.array([]))
    _points_b_y = reactive.value(np.array([]))
    _is_a_random = reactive.value(True)

    def _generate_clust_data():
        try: n = int(input.pdx_clust_n())
        except Exception: n = 200
        n = max(10, min(n, 2000))

        rx = np.random.uniform(0, 1, n)
        ry = np.random.uniform(0, 1, n)

        m = int(np.ceil(np.sqrt(n)))
        sx, sy = [], []
        for i in range(m):
            for j in range(m):
                jx = np.random.uniform(-0.4, 0.4) / m
                jy = np.random.uniform(-0.4, 0.4) / m
                sx.append((i + 0.5) / m + jx)
                sy.append((j + 0.5) / m + jy)

        indices = np.random.permutation(len(sx))[:n]
        sx = np.array(sx)[indices]
        sy = np.array(sy)[indices]

        if np.random.rand() > 0.5:
            _points_a_x.set(rx); _points_a_y.set(ry)
            _points_b_x.set(sx); _points_b_y.set(sy)
            _is_a_random.set(True)
        else:
            _points_a_x.set(sx); _points_a_y.set(sy)
            _points_b_x.set(rx); _points_b_y.set(ry)
            _is_a_random.set(False)

        ui.update_switch("pdx_clust_reveal", value=False)

    @reactive.effect
    def _init_clust():
        if len(_points_a_x()) == 0:
            _generate_clust_data()

    @reactive.effect
    @reactive.event(input.pdx_clust_btn_generate)
    def _on_generate_clust():
        _generate_clust_data()

    @reactive.effect
    @reactive.event(input.pdx_clust_n)
    def _on_param_change_clust():
        _generate_clust_data()

    @render.ui
    def pdx_clust_grid_controls():
        if not input.pdx_clust_show_grid():
            return ui.div()
        return ui.input_slider(
            "pdx_clust_grid_k",
            ui.TagList("Grid divisions (k)", tip("Number of divisions per axis. Total cells = k\u00b2.")),
            min=2, max=10, value=5, step=1, width="100%",
        )

    @render.text
    def pdx_clust_title_a():
        if input.pdx_clust_reveal():
            return "Plot A: Random" if _is_a_random() else "Plot A: Stratified (Even)"
        return "Plot A"

    @render.text
    def pdx_clust_title_b():
        if input.pdx_clust_reveal():
            return "Plot B: Stratified (Even)" if _is_a_random() else "Plot B: Random"
        return "Plot B"

    @render.ui
    def pdx_clust_scatter_a():
        x = _points_a_x()
        if len(x) == 0: return ui.div("Generating...")
        try: show_grid = input.pdx_clust_show_grid()
        except Exception: show_grid = False
        try: grid_k = int(input.pdx_clust_grid_k())
        except Exception: grid_k = 5
        title = "Random" if _is_a_random() else "Stratified"
        title = title if input.pdx_clust_reveal() else "Unknown"
        fig = draw_cluster_scatter(x, _points_a_y(), len(x), title, show_grid=show_grid, grid_k=grid_k, dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def pdx_clust_scatter_b():
        x = _points_b_x()
        if len(x) == 0: return ui.div()
        try: show_grid = input.pdx_clust_show_grid()
        except Exception: show_grid = False
        try: grid_k = int(input.pdx_clust_grid_k())
        except Exception: grid_k = 5
        title = "Stratified" if _is_a_random() else "Random"
        title = title if input.pdx_clust_reveal() else "Unknown"
        fig = draw_cluster_scatter(x, _points_b_y(), len(x), title, show_grid=show_grid, grid_k=grid_k, dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def pdx_clust_sidebar_analysis_text():
        if not input.pdx_clust_show_analysis():
            return None
        return ui.div(
            "Notice how the random plot has a wider variance in quadrat counts and matches the theoretical Poisson distribution, "
            "while the stratified plot has artificially consistent quadrat counts and avoids small NN distances.",
            style="margin-bottom: 16px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
        )

    @render.ui
    def pdx_clust_analysis_area():
        if not input.pdx_clust_show_analysis():
            return None

        x_a = _points_a_x(); y_a = _points_a_y()
        x_b = _points_b_x(); y_b = _points_b_y()
        if len(x_a) == 0 or len(x_b) == 0: return ui.div()

        try: grid_k = int(input.pdx_clust_grid_k())
        except Exception: grid_k = 5

        fig_quad_a = draw_quadrat_chart(x_a, y_a, grid_k=grid_k, dark=is_dark())
        fig_quad_b = draw_quadrat_chart(x_b, y_b, grid_k=grid_k, dark=is_dark())
        fig_nn_a = draw_nn_distance(x_a, y_a, len(x_a), dark=is_dark())
        fig_nn_b = draw_nn_distance(x_b, y_b, len(x_b), dark=is_dark())

        return ui.TagList(
            ui.div(
                ui.div(
                    ui.div("QUADRAT ANALYSIS (Plot A)\u00a0", tip("If points are random, counts per grid cell match a Poisson distribution (red line)."), class_="card-title"),
                    fig_to_ui(fig_quad_a), class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div("QUADRAT ANALYSIS (Plot B)\u00a0", tip("If points are random, counts per grid cell match a Poisson distribution (red line)."), class_="card-title"),
                    fig_to_ui(fig_quad_b), class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; height: 22vh; min-height: 150px;",
            ),
            ui.div(
                ui.div(
                    ui.div("NEAREST-NEIGHBOUR (Plot A)\u00a0", tip("Distance to the closest point. Random patterns naturally have many small distances (clumps)."), class_="card-title"),
                    fig_to_ui(fig_nn_a), class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div("NEAREST-NEIGHBOUR (Plot B)\u00a0", tip("Distance to the closest point. Random patterns naturally have many small distances (clumps)."), class_="card-title"),
                    fig_to_ui(fig_nn_b), class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; height: 22vh; min-height: 150px;",
            ),
        )


    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 3 — Simpson's Paradox
    # ═════════════════════════════════════════════════════════════════════════

    _simp_x = reactive.value(np.array([]))
    _simp_y = reactive.value(np.array([]))
    _simp_g = reactive.value(np.array([], dtype=int))

    def _generate_simpsons():
        try: n = int(input.pdx_simp_n())
        except Exception: n = 80

        try: k = int(input.pdx_simp_k())
        except Exception: k = 3

        try: noise = float(input.pdx_simp_noise())
        except Exception: noise = 1.8

        x, y, g = generate_simpsons_data(n, k, noise)
        _simp_x.set(x)
        _simp_y.set(y)
        _simp_g.set(g)

    @reactive.effect
    def _init_simp():
        if len(_simp_x()) == 0:
            _generate_simpsons()

    @reactive.effect
    @reactive.event(input.pdx_simp_btn_generate)
    def _on_generate_simp():
        _generate_simpsons()

    @reactive.effect
    @reactive.event(input.pdx_simp_n, input.pdx_simp_k, input.pdx_simp_noise)
    def _on_param_change_simp():
        _generate_simpsons()

    @render.ui
    def pdx_simp_explanation():
        show_g = input.pdx_simp_show_groups()
        show_t = input.pdx_simp_show_trends()

        if not show_g and not show_t:
            return ui.div(
                "Look at the data. Does there seem to be a relationship between x and y? Try enabling ",
                ui.tags.strong("Show trendlines"), " to check your intuition, then ",
                ui.tags.strong("Show groups"), " to reveal the hidden confounder.",
                style="margin-bottom: 12px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        elif show_t and not show_g:
            return ui.div(
                "The overall trendline shows a ", ui.tags.strong("negative"),
                " slope. But is the whole story told? Enable ", ui.tags.strong("Show groups"),
                " to reveal a hidden structure in the data.",
                style="margin-bottom: 12px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        elif show_g and not show_t:
            return ui.div(
                "The groups are now visible. Enable ", ui.tags.strong("Show trendlines"),
                " to compare the within-group slopes with the overall slope.",
                style="margin-bottom: 12px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        else:
            x = _simp_x(); g = _simp_g(); y = _simp_y()
            if len(x) == 0: return ui.div()
            k = int(g.max()) + 1
            slopes = []
            for gi in range(k):
                m = g == gi
                if m.sum() > 1:
                    s, _ = np.polyfit(x[m], y[m], 1)
                    slopes.append(s)
            overall_s, _ = np.polyfit(x, y, 1)
            avg_within = np.mean(slopes) if slopes else 0

            return ui.div(
                ui.tags.strong("Simpson\u2019s Paradox revealed! "),
                f"The overall slope is \u03b2 = {overall_s:+.2f} (negative), but the average within-group slope is \u03b2 = {avg_within:+.2f} (positive). ",
                "Aggregating data across groups hides the confounding variable and reverses the true relationship.",
                style="margin-bottom: 12px; font-size: 0.82rem; color: var(--c-text3); line-height: 1.4;",
            )

    @render.ui
    def pdx_simp_stats_row():
        x = _simp_x(); y = _simp_y(); g = _simp_g()
        if len(x) == 0: return ui.div()

        k = int(g.max()) + 1
        n_total = len(x)
        overall_slope, _ = np.polyfit(x, y, 1)

        slopes = []
        for gi in range(k):
            m = g == gi
            if m.sum() > 1:
                s, _ = np.polyfit(x[m], y[m], 1)
                slopes.append(s)
        avg_within = np.mean(slopes) if slopes else 0

        return ui.div(
            ui.div(ui.div("N TOTAL", class_="stat-label"), ui.div(f"{n_total}", class_="stat-value coverage"), class_="stat-card"),
            ui.div(ui.div("GROUPS", class_="stat-label"), ui.div(f"{k}", class_="stat-value included"), class_="stat-card"),
            ui.div(ui.div("OVERALL \u03b2", class_="stat-label"), ui.div(f"{overall_slope:+.3f}", class_="stat-value missed"), class_="stat-card"),
            ui.div(ui.div("AVG WITHIN \u03b2", class_="stat-label"), ui.div(f"{avg_within:+.3f}", class_="stat-value total"), class_="stat-card"),
            class_="stats-row",
        )

    @render.ui
    def pdx_simp_chart_title():
        show_g = input.pdx_simp_show_groups()
        show_t = input.pdx_simp_show_trends()
        parts = ["SIMPSON\u2019S PARADOX"]
        if show_g: parts.append("GROUPS SHOWN")
        if show_t: parts.append("TRENDLINES")
        return " \u00b7 ".join(parts)

    @render.ui
    def pdx_simp_scatter():
        x = _simp_x(); y = _simp_y(); g = _simp_g()
        if len(x) == 0: return ui.div("Generating data\u2026")
        try: show_g = input.pdx_simp_show_groups()
        except Exception: show_g = False
        try: show_t = input.pdx_simp_show_trends()
        except Exception: show_t = False

        fig = draw_simpsons_scatter(x, y, g, show_groups=show_g, show_trends=show_t, dark=is_dark())
        return fig_to_ui(fig)


    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 4 — Coupon Collector's Problem
    # ═════════════════════════════════════════════════════════════════════════

    _ccp_active_preset = reactive.value("none")

    @reactive.effect
    @reactive.event(input.pdx_ccp_pre_d6)
    def _ccp_pre_d6():
        ui.update_slider("pdx_ccp_n", value=6)
        ui.update_slider("pdx_ccp_b", value=1)
        _ccp_active_preset.set("d6")

    @reactive.effect
    @reactive.event(input.pdx_ccp_pre_d20)
    def _ccp_pre_d20():
        ui.update_slider("pdx_ccp_n", value=20)
        ui.update_slider("pdx_ccp_b", value=1)
        _ccp_active_preset.set("d20")

    @reactive.effect
    @reactive.event(input.pdx_ccp_pre_mtg_c)
    def _ccp_pre_mtg():
        ui.update_slider("pdx_ccp_n", value=101)
        ui.update_slider("pdx_ccp_b", value=10)
        _ccp_active_preset.set("mtg")

    @reactive.effect
    @reactive.event(input.pdx_ccp_pre_poke)
    def _ccp_pre_poke():
        ui.update_slider("pdx_ccp_n", value=151)
        ui.update_slider("pdx_ccp_b", value=11)
        _ccp_active_preset.set("poke")

    # ── Guard: clamp B ≤ N dynamically ──────────────────────────────────
    @reactive.effect
    @reactive.event(input.pdx_ccp_n)
    def _ccp_clamp_b():
        N = input.pdx_ccp_n()
        ui.update_slider("pdx_ccp_b", max=N)

    @render.ui
    def pdx_ccp_preset_desc():
        p = _ccp_active_preset()
        if p == "none":
            return ui.div(
                "← Select an example to auto-fill the values.",
                class_="np-preset-hint",
            )
        elif p == "d6":
            return ui.div(
                ui.tags.strong("D6 Dice: "),
                "Rolling a standard 6-sided die until you see all 6 faces.",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "d20":
            return ui.div(
                ui.tags.strong("D20 Set: "),
                "Rolling a 20-sided die until you've hit every single number from 1 to 20.",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "mtg":
            return ui.div(
                ui.tags.strong("MtG Common Set: "),
                "Collecting a 101-card common set by opening Draft Boosters (10 distinct commons each).",
                class_="np-preset-hint np-preset-hint--active"
            )
        elif p == "poke":
            return ui.div(
                ui.tags.strong("Pokémon Gen 1: "),
                "Collecting all 151 original Pokémon by opening 11-card booster packs.",
                class_="np-preset-hint np-preset-hint--active"
            )
        return None

    @render.ui
    def pdx_ccp_formula():
        b = input.pdx_ccp_b()
        if b == 1:
            math_str = (
                f"$$\\begin{{aligned}}"
                f"E(T) &= N \\sum_{{i=1}}^N \\frac{{1}}{{i}} = N \\cdot H_N \\\\"
                f"&\\approx N \\ln(N) + \\gamma N + 0.5"
                f"\\end{{aligned}}$$"
            )
        else:
            math_str = (
                f"$$\\begin{{aligned}}"
                f"E(Packs) &= \\sum_{{j=1}}^{{N}} (-1)^{{j-1}} \\binom{{N}}{{j}} \\frac{{1}}{{1 - \\binom{{N-j}}{{B}} / \\binom{{N}}{{B}}}} \\\\"
                f"E(Items) &= E(Packs) \\times B"
                f"\\end{{aligned}}$$"
            )
        return ui.HTML(math_str)

    @render.ui
    def pdx_ccp_formula_note():
        b = input.pdx_ccp_b()
        if b == 1:
            return ui.div(
                "The harmonic sum reflects how each item gets harder to find: "
                "the k-th unique item needs on average ",
                ui.tags.strong("N / (N−k+1)"),
                " draws. Early items are cheap; the last one costs N draws.",
                style="font-size:0.72rem; color:var(--c-text3); line-height:1.45;",
            )
        return ui.div(
            "Exact inclusion–exclusion over all subsets of missing items. "
            "Each pack draws B distinct items without replacement, so "
            "P(specific item missed) = ",
            ui.tags.strong("(N−B)/N"),
            " per pack.",
            style="font-size:0.72rem; color:var(--c-text3); line-height:1.45;",
        )

    @render.ui
    def pdx_ccp_chart_caption():
        view = input.pdx_ccp_view()
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)
        caption_style = (
            "font-size:0.72rem; color:var(--c-text3); "
            "font-style:italic; line-height:1.45; margin-bottom:4px;"
        )
        if view == "Simulation (Distribution)":
            return ui.div(
                "2 000 simulated collectors, each opening packs until the set is complete. "
                "The histogram is ",
                ui.tags.strong("right-skewed"),
                ": lucky collectors finish early, but a long right tail "
                "means many need well above E[Packs]. "
                "Vertical line marks the exact expected value.",
                style=caption_style,
            )
        elif view == "Probability Curve (CDF)":
            return ui.div(
                "S-curve of completion probability derived from the simulation. "
                "The vertical line marks E[Packs]: roughly ",
                ui.tags.strong("63 % of collectors finish by that point"),
                " — the long right tail pulls the mean above the median. "
                "Use this to answer “How many packs to be 90 % sure?”",
                style=caption_style,
            )
        else:
            return ui.div(
                "Y-axis: expected packs to collect the ",
                ui.tags.em("next"),
                " new item, given k already in the collection. "
                "Stays near 1 early on; shoots up near N — the last item alone needs "
                "on average ",
                ui.tags.strong(f"N/B = {N/B:.1f} packs"),
                ". This exponential growth at the right is why “almost done” takes as long as the first half.",
                style=caption_style,
            )

    @reactive.calc
    def _ccp_expected_packs():
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)
        if B >= N:
            return 1.0

        # Exact Expected Value using fractions to prevent overflow
        s = Fraction(0)
        for j in range(1, N + 1):
            num = math.comb(N, j)
            num_b = math.comb(N - j, B) if (N - j) >= B else 0
            den_b = math.comb(N, B)
            prob = Fraction(num_b, den_b)
            if prob == 1:
                continue
            term = Fraction(num, 1) * Fraction(1, 1 - prob)
            if (j - 1) % 2 == 0:
                s += term
            else:
                s -= term
        return float(s)
        
    @reactive.calc
    def _ccp_expected_half():
        """Expected packs/draws to collect the first N/2 unique items."""
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)
        target = N // 2
        if B >= target:
            return 1.0

        if B == 1:
            # Exact: sum of N/(N-k) for k = 0 .. target-1
            return sum(N / (N - k) for k in range(target))

        # B > 1: use simulation average
        return _ccp_sim_data()['avg_half']

    @reactive.calc
    def _ccp_expected_last():
        """Expected packs to get the very last missing item.
        P(miss one specific item) = (N-B)/N  →  E[packs] = N/B.
        """
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)
        return N / B

    @reactive.calc
    def _ccp_sim_data():
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)
        C = 2000

        target_half = N // 2

        packs_total = np.zeros(C, dtype=int)
        packs_half = np.zeros(C, dtype=int)

        if B == 1:
            # Vectorised geometric jumps.
            # k = number of distinct items already collected (0 .. N-1)
            # p(new) = (N-k)/N  →  geometric(p)
            cumulative = np.zeros(C, dtype=int)
            for k in range(N):
                jumps = np.random.geometric((N - k) / N, size=C)
                cumulative += jumps
                if k + 1 == target_half:          # just collected the target_half-th item
                    packs_half[:] = cumulative
            packs_total[:] = cumulative
        else:
            pool = range(N)
            for c in range(C):
                col = set()
                p = 0
                got_half = 0
                while len(col) < N:
                    col.update(random.sample(pool, B))
                    p += 1
                    if len(col) >= target_half and got_half == 0:
                        got_half = p
                packs_total[c] = p
                packs_half[c] = got_half

        return {
            'total_packs': packs_total,
            'avg_half': float(np.mean(packs_half)),
        }

    @render.text
    def pdx_ccp_exp_packs():
        val = _ccp_expected_packs()
        return f"{val:,.1f}"

    @render.text
    def pdx_ccp_exp_items():
        val = _ccp_expected_packs() * input.pdx_ccp_b()
        return f"{val:,.0f}"

    @render.text
    def pdx_ccp_cost_cmp():
        cost = input.pdx_ccp_cost() or 0

        half = _ccp_expected_half()
        last = _ccp_expected_last()

        if cost > 0:
            return f"${half * cost:,.2f} vs ${last * cost:,.2f}"
        return f"{half:,.1f} packs vs {last:,.1f} packs"

    @render.text
    def pdx_ccp_total_cost():
        val = _ccp_expected_packs()
        cost = input.pdx_ccp_cost() or 0
        if cost <= 0:
            return "---"
        return f"${val * cost:,.2f}"

    @render.ui
    def pdx_ccp_chart_area():
        view = input.pdx_ccp_view()
        dark = is_dark()
        N = input.pdx_ccp_n()
        B = min(input.pdx_ccp_b(), N)

        expected = _ccp_expected_packs()
        sim_data = _ccp_sim_data()

        if view == "Simulation (Distribution)":
            fig = draw_ccp_distribution(sim_data['total_packs'], expected, dark)
        elif view == "Probability Curve (CDF)":
            fig = draw_ccp_cdf(sim_data['total_packs'], expected, dark)
        else:
            fig = draw_ccp_curve(N, B, expected, dark)

        return fig_to_ui(fig)


    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 5 — Gambler's Fallacy
    # ═════════════════════════════════════════════════════════════════════════

    _gf_active_preset = reactive.value("none")

    # ── Preset handlers ─────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pdx_gf_pre_coin)
    def _gf_pre_coin():
        ui.update_slider("pdx_gf_p", value=0.50)
        ui.update_slider("pdx_gf_k", value=3)
        ui.update_slider("pdx_gf_n", value=300)
        _gf_active_preset.set("coin")

    @reactive.effect
    @reactive.event(input.pdx_gf_pre_roulette)
    def _gf_pre_roulette():
        ui.update_slider("pdx_gf_p", value=0.4865)  # 18/37, European
        ui.update_slider("pdx_gf_k", value=5)
        ui.update_slider("pdx_gf_n", value=500)
        _gf_active_preset.set("roulette")

    @reactive.effect
    @reactive.event(input.pdx_gf_pre_mc1913)
    def _gf_pre_mc1913():
        ui.update_slider("pdx_gf_p", value=0.4865)
        ui.update_slider("pdx_gf_k", value=10)
        ui.update_slider("pdx_gf_n", value=1000)
        _gf_active_preset.set("mc1913")

    @reactive.effect
    @reactive.event(input.pdx_gf_pre_ft)
    def _gf_pre_ft():
        ui.update_slider("pdx_gf_p", value=0.75)
        ui.update_slider("pdx_gf_k", value=5)
        ui.update_slider("pdx_gf_n", value=400)
        _gf_active_preset.set("ft")

    # ── Core simulation ─────────────────────────────────────────────────
    @reactive.calc
    def _gf_sim_data():
        p = input.pdx_gf_p()
        n = input.pdx_gf_n()
        # Trigger fresh seed on resimulate-button click. We deliberately do NOT
        # depend on pdx_gf_k — k only changes which slice/highlight is rendered.
        input.pdx_gf_btn_resim()

        rng = np.random.default_rng()
        data = rng.binomial(1, p, (_GF_N_SIMS, n))

        single = data[0]

        # Streak conditioning is "AT LEAST kk" (last kk flips are all 1).
        # By independence the empirical fraction tends to p regardless of kk.
        # For each kk, find streaks of length kk and record the NEXT flip.
        #   define cs0 with a leading zero so cs0[:, j] = sum(data[:, 0..j-1]);
        #   then sum(data[:, j..j+kk-1]) = cs0[:, j+kk] - cs0[:, j].
        # The "next flip" is data[:, j+kk]; valid j range is 0..n-kk-1 so that
        # j+kk is an in-bounds column.
        k_max = min(10, n - 1)
        after_k = {}
        cs = np.cumsum(data, axis=1)
        cs0 = np.hstack([np.zeros((_GF_N_SIMS, 1), dtype=int), cs])  # (N, n+1)
        for kk in range(1, k_max + 1):
            if kk < n:
                sums = cs0[:, kk:n] - cs0[:, :n - kk]   # shape (N_SIMS, n-kk)
                mask = (sums == kk)                      # last kk are all 1
                next_flips = data[:, kk:n][mask]         # data[:, j+kk]
                after_k[kk] = next_flips     # may be empty for large kk + small n

        max_streaks = _compute_max_streaks(data)

        return {
            'single': single,
            'after_k': after_k,
            'max_streaks': max_streaks,
            'p': p, 'n': n,
        }

    # ── Preset description ──────────────────────────────────────────────
    @render.ui
    def pdx_gf_preset_desc():
        p = _gf_active_preset()
        if p == "none":
            return ui.div(
                "← Select an example to auto-fill the values.",
                class_="np-preset-hint",
            )
        elif p == "coin":
            return ui.div(
                ui.tags.strong("Fair Coin: "),
                "Classic 50/50. Even here, streaks of 7+ are expected in 300 flips — most people are surprised.",
                class_="np-preset-hint np-preset-hint--active",
            )
        elif p == "roulette":
            return ui.div(
                ui.tags.strong("Roulette (Red): "),
                "European roulette: 18 red, 18 black, 1 green. P(red) = 18/37 ≈ 48.65 %. The casino edge comes from never paying you when green hits.",
                class_="np-preset-hint np-preset-hint--active",
            )
        elif p == "mc1913":
            return ui.div(
                ui.tags.strong("Monte Carlo 1913: "),
                "On 18 August 1913, black came up 26 times in a row at Casino de Monte‑Carlo. Gamblers, convinced red was overdue, lost millions. After 26 blacks, the probability of red on the next spin was still 18/37 ≈ 48.6 %.",
                class_="np-preset-hint np-preset-hint--active",
            )
        elif p == "ft":
            return ui.div(
                ui.tags.strong("Free Throws: "),
                "Inspired by NBA studies (Gilovich, Vallone & Tversky, 1985) that found no evidence of the Hot Hand. ",
                ui.tags.strong("This simulation hard-codes independence between shots"),
                " — yet streaks of 5–7 makes still emerge naturally. The pattern your intuition labels “in the zone” is exactly what pure randomness produces.",
                class_="np-preset-hint np-preset-hint--active",
            )
        return None

    # ── Formula card ────────────────────────────────────────────────────
    @render.ui
    def pdx_gf_formula():
        math_str = (
            f"$$\\begin{{aligned}}"
            f"P(H_{{n+1}}=1\\mid\\text{{streak of }}k) &= p \\\\"
            f"E[L_n] &\\approx \\lfloor\\log_{{1/p}}(n(1-p))\\rfloor"
            f"\\end{{aligned}}$$"
        )
        return ui.HTML(math_str)

    @render.ui
    def pdx_gf_formula_note():
        return ui.div(
            "Top: by independence, conditioning on history changes nothing. ",
            "Bottom: even pure randomness produces long streaks — they are "
            "expected, not anomalous.",
            style="font-size:0.72rem; color:var(--c-text3); line-height:1.45;",
        )

    # ── Stats ───────────────────────────────────────────────────────────
    @render.text
    def pdx_gf_stat_p():
        return f"{input.pdx_gf_p():.3f}"

    @render.text
    def pdx_gf_stat_after_k():
        d = _gf_sim_data()
        k = input.pdx_gf_k()
        after = d['after_k'].get(k, np.array([]))
        if len(after) == 0:
            return "—"
        return f"{after.mean():.3f}"

    @render.text
    def pdx_gf_stat_avg_max():
        return f"{_gf_sim_data()['max_streaks'].mean():.1f}"

    @render.text
    def pdx_gf_stat_exp_max():
        return str(_expected_max_streak(input.pdx_gf_p(), input.pdx_gf_n()))

    # ── Chart caption (dynamic per view) ────────────────────────────────
    @render.ui
    def pdx_gf_chart_caption():
        view = input.pdx_gf_view()
        caption_style = (
            "font-size:0.72rem; color:var(--c-text3); "
            "font-style:italic; line-height:1.45; margin-bottom:4px;"
        )
        if view == "Independence Proof":
            return ui.div(
                "Each bar shows the empirical P(success | last ≥ k successes) "
                "across 1 000 simulated runs. The red dashed line is true p; "
                "the pink band is ±2 SE around p. ",
                ui.tags.strong("Bars become semi-transparent for large k"),
                " because fewer cases supply each estimate — fading communicates "
                "uncertainty without falsely framing deviations as “bad”. "
                "Almost all bars sit inside the SE band: deviations are sampling "
                "noise, not evidence of dependence.",
                style=caption_style,
            )
        elif view == "Streak Distribution":
            return ui.div(
                "Histogram of the longest streak of successes in each of 1 000 "
                "simulated runs. The amber line marks the theoretical expectation "
                "E[L_n] ≈ log_{1/p}(n·(1−p)). ",
                ui.tags.strong("Long streaks are not anomalies"),
                " — they are precisely what independence predicts.",
                style=caption_style,
            )
        else:  # Flip Sequence
            return ui.div(
                "Top: running proportion of successes (blue) converges to p (red dashed) "
                "by the Law of Large Numbers — not because the coin corrects itself, "
                "but because early runs matter less over time. Bottom: actual flip "
                "sequence as a barcode (green = success, red = failure). ",
                ui.tags.strong("Amber bands"),
                " mark every streak of length ≥ k.",
                style=caption_style,
            )

    # ── Chart area dispatch ─────────────────────────────────────────────
    @render.ui
    def pdx_gf_chart_area():
        view = input.pdx_gf_view()
        dark = is_dark()
        d = _gf_sim_data()
        k = input.pdx_gf_k()

        if view == "Independence Proof":
            fig = draw_gf_after_k(d['after_k'], d['p'], k, dark=dark)
        elif view == "Streak Distribution":
            fig = draw_gf_streaks(d['max_streaks'], d['p'], d['n'], dark=dark)
        else:
            fig = draw_gf_sequence(d['single'], d['p'], k, dark=dark)

        return fig_to_ui(fig)
