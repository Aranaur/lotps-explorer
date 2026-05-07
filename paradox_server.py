# =============================================================================
# Paradox Explorer — server logic
#
#   Tab 1  Clustering Illusion   (pdx1_*)
#   Tab 2  Simpson's Paradox     (pdx2_*)
# =============================================================================

import numpy as np
from shiny import reactive, render, ui

from utils import tip
from paradox_plots import (
    draw_cluster_scatter, draw_quadrat_chart, draw_nn_distance,
    generate_simpsons_data, draw_simpsons_scatter,
)
from theme import fig_to_ui


def paradox_server(input, output, session, is_dark):

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Clustering Illusion
    # ═════════════════════════════════════════════════════════════════════════

    # ── Reactive state ────────────────────────────────────────────────────────
    _points_a_x = reactive.value(np.array([]))
    _points_a_y = reactive.value(np.array([]))
    _points_b_x = reactive.value(np.array([]))
    _points_b_y = reactive.value(np.array([]))
    _is_a_random = reactive.value(True)

    # ── Data generation ───────────────────────────────────────────────────────
    def _generate_data():
        try:
            n = int(input.pdx1_n())
        except Exception:
            n = 200
        n = max(10, min(n, 2000))

        # Uniform Random (Poisson process)
        rx = np.random.uniform(0, 1, n)
        ry = np.random.uniform(0, 1, n)

        # Stratified (Jittered grid)
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

        ui.update_switch("pdx1_reveal", value=False)

    # ── Initial generation ────────────────────────────────────────────────────
    @reactive.effect
    def _init():
        if len(_points_a_x()) == 0:
            _generate_data()

    @reactive.effect
    @reactive.event(input.pdx1_btn_generate)
    def _on_generate():
        _generate_data()

    @reactive.effect
    @reactive.event(input.pdx1_n)
    def _on_param_change():
        _generate_data()

    # ── Grid controls ─────────────────────────────────────────────────────────
    @render.ui
    def pdx1_grid_controls():
        if not input.pdx1_show_grid():
            return ui.div()
        return ui.input_slider(
            "pdx1_grid_k",
            ui.TagList("Grid divisions (k)",
                       tip("Number of divisions per axis. Total cells = k\u00b2.")),
            min=2, max=10, value=5, step=1, width="100%",
        )

    # ── Titles ────────────────────────────────────────────────────────────────
    @render.text
    def pdx1_title_a():
        if input.pdx1_reveal():
            return "Plot A: Random" if _is_a_random() else "Plot A: Stratified (Even)"
        return "Plot A"

    @render.text
    def pdx1_title_b():
        if input.pdx1_reveal():
            return "Plot B: Stratified (Even)" if _is_a_random() else "Plot B: Random"
        return "Plot B"

    # ── Scatter A & B ─────────────────────────────────────────────────────────
    @render.ui
    def pdx1_scatter_a():
        x = _points_a_x()
        if len(x) == 0:
            return ui.div("Generate data to see the scatter plot.")

        try: show_grid = input.pdx1_show_grid()
        except Exception: show_grid = False

        try: grid_k = int(input.pdx1_grid_k())
        except Exception: grid_k = 5

        title = "Random" if _is_a_random() else "Stratified"
        title = title if input.pdx1_reveal() else "Unknown"

        fig = draw_cluster_scatter(
            x, _points_a_y(), len(x), title,
            show_grid=show_grid, grid_k=grid_k, dark=is_dark(),
        )
        return fig_to_ui(fig)

    @render.ui
    def pdx1_scatter_b():
        x = _points_b_x()
        if len(x) == 0:
            return ui.div()

        try: show_grid = input.pdx1_show_grid()
        except Exception: show_grid = False

        try: grid_k = int(input.pdx1_grid_k())
        except Exception: grid_k = 5

        title = "Stratified" if _is_a_random() else "Random"
        title = title if input.pdx1_reveal() else "Unknown"

        fig = draw_cluster_scatter(
            x, _points_b_y(), len(x), title,
            show_grid=show_grid, grid_k=grid_k, dark=is_dark(),
        )
        return fig_to_ui(fig)

    # ── Analysis sidebar text ─────────────────────────────────────────────────
    @render.ui
    def pdx1_sidebar_analysis_text():
        if not input.pdx1_show_analysis():
            return None
        return ui.div(
            "Notice how the random plot has a wider variance in quadrat "
            "counts and matches the theoretical Poisson distribution, "
            "while the stratified plot has artificially consistent quadrat "
            "counts and avoids small NN distances.",
            style="margin-bottom: 16px; font-size: 0.82rem; "
                  "font-style: italic; color: var(--c-text3); line-height: 1.4;",
        )

    # ── Analysis area ─────────────────────────────────────────────────────────
    @render.ui
    def pdx1_analysis_area():
        if not input.pdx1_show_analysis():
            return None

        x_a = _points_a_x(); y_a = _points_a_y()
        x_b = _points_b_x(); y_b = _points_b_y()

        if len(x_a) == 0 or len(x_b) == 0:
            return ui.div()

        try: grid_k = int(input.pdx1_grid_k())
        except Exception: grid_k = 5

        fig_quad_a = draw_quadrat_chart(x_a, y_a, grid_k=grid_k, dark=is_dark())
        fig_quad_b = draw_quadrat_chart(x_b, y_b, grid_k=grid_k, dark=is_dark())
        fig_nn_a = draw_nn_distance(x_a, y_a, len(x_a), dark=is_dark())
        fig_nn_b = draw_nn_distance(x_b, y_b, len(x_b), dark=is_dark())

        return ui.TagList(
            ui.div(
                ui.div(
                    ui.div(
                        "QUADRAT ANALYSIS (Plot A)\u00a0",
                        tip("If points are random, counts per grid cell match "
                            "a Poisson distribution (red line). Stratified "
                            "patterns have artificially consistent counts."),
                        class_="card-title",
                    ),
                    fig_to_ui(fig_quad_a),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div(
                        "QUADRAT ANALYSIS (Plot B)\u00a0",
                        tip("If points are random, counts per grid cell match "
                            "a Poisson distribution (red line). Stratified "
                            "patterns have artificially consistent counts."),
                        class_="card-title",
                    ),
                    fig_to_ui(fig_quad_b),
                    class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; "
                      "gap: 8px; margin-bottom: 8px; height: 22vh; min-height: 150px;",
            ),
            ui.div(
                ui.div(
                    ui.div(
                        "NEAREST-NEIGHBOUR (Plot A)\u00a0",
                        tip("Distance to the closest point. Random patterns "
                            "naturally have many small distances (clumps). "
                            "Stratified patterns lack small distances because "
                            "points are spaced apart."),
                        class_="card-title",
                    ),
                    fig_to_ui(fig_nn_a),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div(
                        "NEAREST-NEIGHBOUR (Plot B)\u00a0",
                        tip("Distance to the closest point. Random patterns "
                            "naturally have many small distances (clumps). "
                            "Stratified patterns lack small distances because "
                            "points are spaced apart."),
                        class_="card-title",
                    ),
                    fig_to_ui(fig_nn_b),
                    class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; "
                      "gap: 8px; height: 22vh; min-height: 150px;",
            ),
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Simpson's Paradox
    # ═════════════════════════════════════════════════════════════════════════

    _simp_x = reactive.value(np.array([]))
    _simp_y = reactive.value(np.array([]))
    _simp_g = reactive.value(np.array([], dtype=int))

    def _generate_simpsons():
        try: n = int(input.pdx2_n())
        except Exception: n = 80

        try: k = int(input.pdx2_k())
        except Exception: k = 3

        try: noise = float(input.pdx2_noise())
        except Exception: noise = 1.8

        x, y, g = generate_simpsons_data(n, k, noise)
        _simp_x.set(x)
        _simp_y.set(y)
        _simp_g.set(g)

    # ── Initial generation ────────────────────────────────────────────────────
    @reactive.effect
    def _init_simp():
        if len(_simp_x()) == 0:
            _generate_simpsons()

    @reactive.effect
    @reactive.event(input.pdx2_btn_generate)
    def _on_generate_simp():
        _generate_simpsons()

    @reactive.effect
    @reactive.event(input.pdx2_n, input.pdx2_k, input.pdx2_noise)
    def _on_param_change_simp():
        _generate_simpsons()

    # ── Dynamic explanation ───────────────────────────────────────────────────
    @render.ui
    def pdx2_explanation():
        show_g = input.pdx2_show_groups()
        show_t = input.pdx2_show_trends()

        if not show_g and not show_t:
            return ui.div(
                "Look at the data. Does there seem to be a relationship "
                "between x and y? Try enabling ",
                ui.tags.strong("Show trendlines"),
                " to check your intuition, then ",
                ui.tags.strong("Show groups"),
                " to reveal the hidden confounder.",
                style="margin-bottom: 12px; font-size: 0.82rem; "
                      "font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        elif show_t and not show_g:
            return ui.div(
                "The overall trendline shows a ",
                ui.tags.strong("negative"),
                " slope. But is the whole story told? Enable ",
                ui.tags.strong("Show groups"),
                " to reveal a hidden structure in the data.",
                style="margin-bottom: 12px; font-size: 0.82rem; "
                      "font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        elif show_g and not show_t:
            return ui.div(
                "The groups are now visible. Enable ",
                ui.tags.strong("Show trendlines"),
                " to compare the within-group slopes with the overall slope.",
                style="margin-bottom: 12px; font-size: 0.82rem; "
                      "font-style: italic; color: var(--c-text3); line-height: 1.4;",
            )
        else:
            # Both on — the paradox is revealed
            x = _simp_x()
            g = _simp_g()
            if len(x) == 0:
                return ui.div()
            k = int(g.max()) + 1
            slopes = []
            for gi in range(k):
                m = g == gi
                if m.sum() > 1:
                    s, _ = np.polyfit(x[m], _simp_y()[m], 1)
                    slopes.append(s)
            overall_s, _ = np.polyfit(x, _simp_y(), 1)
            avg_within = np.mean(slopes) if slopes else 0

            return ui.div(
                ui.tags.strong("Simpson\u2019s Paradox revealed! "),
                f"The overall slope is \u03b2 = {overall_s:+.2f} (negative), "
                f"but the average within-group slope is "
                f"\u03b2 = {avg_within:+.2f} (positive). ",
                "Aggregating data across groups hides the confounding "
                "variable and reverses the true relationship.",
                style="margin-bottom: 12px; font-size: 0.82rem; "
                      "color: var(--c-text3); line-height: 1.4;",
            )

    # ── Stats row ─────────────────────────────────────────────────────────────
    @render.ui
    def pdx2_stats_row():
        x = _simp_x()
        y = _simp_y()
        g = _simp_g()
        if len(x) == 0:
            return ui.div()

        k = int(g.max()) + 1
        n_total = len(x)
        overall_slope, _ = np.polyfit(x, y, 1)

        # Average within-group slope
        slopes = []
        for gi in range(k):
            m = g == gi
            if m.sum() > 1:
                s, _ = np.polyfit(x[m], y[m], 1)
                slopes.append(s)
        avg_within = np.mean(slopes) if slopes else 0

        return ui.div(
            ui.div(
                ui.div("N TOTAL", class_="stat-label"),
                ui.div(f"{n_total}", class_="stat-value coverage"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("GROUPS", class_="stat-label"),
                ui.div(f"{k}", class_="stat-value included"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("OVERALL \u03b2", class_="stat-label"),
                ui.div(f"{overall_slope:+.3f}", class_="stat-value missed"),
                class_="stat-card",
            ),
            ui.div(
                ui.div("AVG WITHIN \u03b2", class_="stat-label"),
                ui.div(f"{avg_within:+.3f}", class_="stat-value total"),
                class_="stat-card",
            ),
            class_="stats-row",
        )

    # ── Chart title ───────────────────────────────────────────────────────────
    @render.ui
    def pdx2_chart_title():
        show_g = input.pdx2_show_groups()
        show_t = input.pdx2_show_trends()
        parts = ["SIMPSON\u2019S PARADOX"]
        if show_g:
            parts.append("GROUPS SHOWN")
        if show_t:
            parts.append("TRENDLINES")
        return " \u00b7 ".join(parts)

    # ── Main scatter ──────────────────────────────────────────────────────────
    @render.ui
    def pdx2_scatter():
        x = _simp_x()
        y = _simp_y()
        g = _simp_g()
        if len(x) == 0:
            return ui.div("Generating data\u2026")

        try: show_g = input.pdx2_show_groups()
        except Exception: show_g = False

        try: show_t = input.pdx2_show_trends()
        except Exception: show_t = False

        fig = draw_simpsons_scatter(
            x, y, g,
            show_groups=show_g,
            show_trends=show_t,
            dark=is_dark(),
        )
        return fig_to_ui(fig)
