# =============================================================================
# Paradox Explorer — server logic
#
#   Tab 1  Base Rate Fallacy     (pdx_brf_*)
#   Tab 2  Clustering Illusion   (pdx_clust_*)
#   Tab 3  Simpson's Paradox     (pdx_simp_*)
# =============================================================================

import numpy as np
from shiny import reactive, render, ui

from utils import tip
from paradox_plots import (
    draw_cluster_scatter, draw_quadrat_chart, draw_nn_distance,
    generate_simpsons_data, draw_simpsons_scatter,
    draw_baserate_sankey, draw_baserate_waffle
)
from theme import fig_to_ui


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
