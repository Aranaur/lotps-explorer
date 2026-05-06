# =============================================================================
# Paradox Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def paradox_panel() -> ui.Tag:
    return ui.nav_panel(
        "Paradox Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Clustering Illusion: "),
                    "Humans see patterns and clusters in random data — "
                    "our brains are wired to find structure even where none exists.",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "True randomness produces ",
                    ui.tags.em("clumps and gaps"),
                    " by chance. A perfectly uniform spread would actually be ",
                    ui.tags.em("non-random"),
                    ".",
                    class_="info-banner-text",
                ),

                # Challenge text
                ui.div(
                    "Can you tell which one is purely random and which one is stratified (regular)? "
                    "Our brains often pick the stratified one as 'random' because it looks more evenly spread.",
                    style="margin-bottom: 16px; font-size: 0.82rem; font-style: italic; color: var(--c-text3); line-height: 1.4;",
                ),

                # Sample size
                ui.input_numeric(
                    "pdx_n",
                    ui.TagList(
                        "Number of points (N)",
                        tip("Total points to scatter. "
                            "More points make the illusion more convincing."),
                    ),
                    value=200, min=10, max=2000, step=10, width="100%",
                ),

                # Grid toggle
                ui.input_checkbox(
                    "pdx_show_grid",
                    ui.TagList(
                        "Show quadrat grid",
                        tip("Overlay a grid to help visual inspection."),
                    ),
                    value=False,
                ),

                # Grid division slider (shown only when grid is on)
                ui.output_ui("pdx_grid_controls"),

                # Analysis toggle
                ui.input_checkbox(
                    "pdx_show_analysis",
                    ui.TagList(
                        "Show Analysis",
                        tip("Show Quadrat Analysis and Nearest-Neighbour Distances charts."),
                    ),
                    value=False,
                ),

                # Analysis explanatory text (shows when Analysis is checked)
                ui.output_ui("pdx_sidebar_analysis_text"),

                # Reveal toggle
                ui.input_switch(
                    "pdx_reveal",
                    "Reveal Answer",
                    value=False,
                ),

                # Generate button
                ui.div(
                    ui.input_action_button(
                        "pdx_btn_generate", "\u21bb  New Pair",
                        class_="btn-ctrl btn-sample btn-full",
                    ),
                    class_="sidebar-btn-row",
                ),

                # Footer
                ui.div(
                    ui.tags.a("LinkedIn",
                              href="https://www.linkedin.com/in/ihormiroshnychenko/",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Telegram", href="https://t.me/araprof",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Website", href="https://aranaur.rbind.io/",
                              target="_blank"),
                    class_="footer-links",
                ),

                class_="sidebar",
            ),

            # ── RIGHT MAIN PANEL ─────────────────────────────────────────────
            ui.div(

                # Plots Row 1: Scatter A & B
                ui.div(
                    ui.div(
                        ui.div(ui.output_text("pdx_title_a", inline=True), class_="card-title"),
                        ui.output_ui("pdx_scatter_a"),
                        class_="glass-card chart-card",
                    ),
                    ui.div(
                        ui.div(ui.output_text("pdx_title_b", inline=True), class_="card-title"),
                        ui.output_ui("pdx_scatter_b"),
                        class_="glass-card chart-card",
                    ),
                    style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;"
                ),

                # Analysis rows
                ui.output_ui("pdx_analysis_area"),

                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
