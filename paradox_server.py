# =============================================================================
# Paradox Explorer — server logic
# =============================================================================

import numpy as np
from shiny import reactive, render, ui

from utils import tip
from paradox_plots import draw_cluster_scatter, draw_quadrat_chart, draw_nn_distance
from theme import fig_to_ui


def paradox_server(input, output, session, is_dark):

    # ── Reactive state ────────────────────────────────────────────────────────
    _points_a_x = reactive.value(np.array([]))
    _points_a_y = reactive.value(np.array([]))
    _points_b_x = reactive.value(np.array([]))
    _points_b_y = reactive.value(np.array([]))
    _is_a_random = reactive.value(True)

    # ── Data generation ───────────────────────────────────────────────────────
    def _generate_data():
        """Generate (x, y) points for both Random and Stratified distributions."""
        try:
            n = int(input.pdx_n())
        except Exception:
            n = 200
        n = max(10, min(n, 2000))

        # 1. Generate Uniform Random (Poisson process)
        rx = np.random.uniform(0, 1, n)
        ry = np.random.uniform(0, 1, n)

        # 2. Generate Stratified (Jittered grid)
        m = int(np.ceil(np.sqrt(n)))
        sx, sy = [], []
        for i in range(m):
            for j in range(m):
                # Center of cell (i, j) with some jitter (max 0.4 from center to avoid touching edges)
                jx = np.random.uniform(-0.4, 0.4) / m
                jy = np.random.uniform(-0.4, 0.4) / m
                sx.append((i + 0.5) / m + jx)
                sy.append((j + 0.5) / m + jy)
        
        # Shuffle and pick exactly n points
        indices = np.random.permutation(len(sx))[:n]
        sx = np.array(sx)[indices]
        sy = np.array(sy)[indices]

        # Randomly assign A and B
        if np.random.rand() > 0.5:
            _points_a_x.set(rx)
            _points_a_y.set(ry)
            _points_b_x.set(sx)
            _points_b_y.set(sy)
            _is_a_random.set(True)
        else:
            _points_a_x.set(sx)
            _points_a_y.set(sy)
            _points_b_x.set(rx)
            _points_b_y.set(ry)
            _is_a_random.set(False)

        # Reset reveal switch
        ui.update_switch("pdx_reveal", value=False)

    # ── Initial generation on first visit ─────────────────────────────────────
    @reactive.effect
    def _init():
        if len(_points_a_x()) == 0:
            _generate_data()

    # ── Generate button ───────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pdx_btn_generate)
    def _on_generate():
        _generate_data()

    # ── Reset on param change ─────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pdx_n)
    def _on_param_change():
        _generate_data()

    # ── Grid controls ─────────────────────────────────────────────────────────
    @render.ui
    def pdx_grid_controls():
        if not input.pdx_show_grid():
            return ui.div()
        return ui.input_slider(
            "pdx_grid_k",
            ui.TagList("Grid divisions (k)",
                       tip("Number of divisions per axis. Total cells = k\u00b2.")),
            min=2, max=10, value=5, step=1, width="100%",
        )

    # ── Titles ────────────────────────────────────────────────────────────────
    @render.text
    def pdx_title_a():
        if input.pdx_reveal():
            return "Plot A: Random" if _is_a_random() else "Plot A: Stratified (Even)"
        return "Plot A"

    @render.text
    def pdx_title_b():
        if input.pdx_reveal():
            return "Plot B: Stratified (Even)" if _is_a_random() else "Plot B: Random"
        return "Plot B"

    # ── Plot: 2D scatter A & B ────────────────────────────────────────────────
    @render.ui
    def pdx_scatter_a():
        x = _points_a_x()
        if len(x) == 0:
            return ui.div("Generate data to see the scatter plot.")
        
        try: show_grid = input.pdx_show_grid()
        except Exception: show_grid = False
        
        try: grid_k = int(input.pdx_grid_k())
        except Exception: grid_k = 5

        title = "Random" if _is_a_random() else "Stratified"
        title = title if input.pdx_reveal() else "Unknown"

        fig = draw_cluster_scatter(
            x, _points_a_y(), len(x), title,
            show_grid=show_grid, grid_k=grid_k, dark=is_dark(),
        )
        return fig_to_ui(fig)

    @render.ui
    def pdx_scatter_b():
        x = _points_b_x()
        if len(x) == 0:
            return ui.div()
        
        try: show_grid = input.pdx_show_grid()
        except Exception: show_grid = False
        
        try: grid_k = int(input.pdx_grid_k())
        except Exception: grid_k = 5

        title = "Stratified" if _is_a_random() else "Random"
        title = title if input.pdx_reveal() else "Unknown"

        fig = draw_cluster_scatter(
            x, _points_b_y(), len(x), title,
            show_grid=show_grid, grid_k=grid_k, dark=is_dark(),
        )
        return fig_to_ui(fig)

    # ── Analysis Explanation ──────────────────────────────────────────────────
    @render.ui
    def pdx_sidebar_analysis_text():
        if not input.pdx_show_analysis():
            return ui.div()
        return ui.div(
            "Notice how the random plot has a wider variance in quadrat counts and matches the theoretical Poisson distribution, "
            "while the stratified plot has artificially consistent quadrat counts and avoids small NN distances.",
            style="margin-bottom: 16px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
        )

    # ── Analysis Area ─────────────────────────────────────────────────────────
    @render.ui
    def pdx_analysis_area():
        if not input.pdx_show_analysis():
            return ui.div()
            
        x_a = _points_a_x()
        y_a = _points_a_y()
        x_b = _points_b_x()
        y_b = _points_b_y()

        if len(x_a) == 0 or len(x_b) == 0:
            return ui.div()

        try: grid_k = int(input.pdx_grid_k())
        except Exception: grid_k = 5

        # Quadrat Plots
        fig_quad_a = draw_quadrat_chart(x_a, y_a, grid_k=grid_k, dark=is_dark())
        fig_quad_b = draw_quadrat_chart(x_b, y_b, grid_k=grid_k, dark=is_dark())

        # NN Plots
        fig_nn_a = draw_nn_distance(x_a, y_a, len(x_a), dark=is_dark())
        fig_nn_b = draw_nn_distance(x_b, y_b, len(x_b), dark=is_dark())

        return ui.div(
            # Quadrat Row
            ui.div(
                ui.div(
                    ui.div(
                        "QUADRAT ANALYSIS (Plot A)\u00a0",
                        tip("If points are random, counts per grid cell match a Poisson distribution (red line). "
                            "Stratified patterns have artificially consistent counts."),
                        class_="card-title"
                    ),
                    fig_to_ui(fig_quad_a),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div(
                        "QUADRAT ANALYSIS (Plot B)\u00a0",
                        tip("If points are random, counts per grid cell match a Poisson distribution (red line). "
                            "Stratified patterns have artificially consistent counts."),
                        class_="card-title"
                    ),
                    fig_to_ui(fig_quad_b),
                    class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;"
            ),
            # NN Row
            ui.div(
                ui.div(
                    ui.div(
                        "NEAREST-NEIGHBOUR (Plot A)\u00a0",
                        tip("Distance to the closest point. Random patterns naturally have many small distances (clumps). "
                            "Stratified patterns lack small distances because points are spaced apart."),
                        class_="card-title"
                    ),
                    fig_to_ui(fig_nn_a),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div(
                        "NEAREST-NEIGHBOUR (Plot B)\u00a0",
                        tip("Distance to the closest point. Random patterns naturally have many small distances (clumps). "
                            "Stratified patterns lack small distances because points are spaced apart."),
                        class_="card-title"
                    ),
                    fig_to_ui(fig_nn_b),
                    class_="glass-card chart-card",
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"
            ),
        )
