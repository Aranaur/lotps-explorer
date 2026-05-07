# =============================================================================
# Paradox Explorer — UI panel
#
# Sub-tab layout modelled after Bayesian Explorer:
#   Tab 1  Clustering Illusion   (pdx1_*)
#   Tab 2  Simpson's Paradox     (pdx2_*)
#
# A single outer sidebar holds both control groups; a tiny JS listener
# toggles visibility based on the active navset_underline tab.
# =============================================================================

from shiny import ui
from utils import tip


# ── Sidebar: Tab 1 — Clustering Illusion ─────────────────────────────────────

def _sidebar_tab1() -> ui.Tag:
    return ui.div(
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
            "Can you tell which one is purely random and which one is "
            "stratified (regular)? Our brains often pick the stratified "
            "one as \u2018random\u2019 because it looks more evenly spread.",
            style="margin-bottom: 16px; font-size: 0.82rem; font-style: italic; "
                  "color: var(--c-text3); line-height: 1.4;",
        ),

        # Sample size
        ui.input_numeric(
            "pdx1_n",
            ui.TagList(
                "Number of points (N)",
                tip("Total points to scatter. "
                    "More points make the illusion more convincing."),
            ),
            value=200, min=10, max=2000, step=10, width="100%",
        ),

        # Grid toggle
        ui.input_checkbox(
            "pdx1_show_grid",
            ui.TagList(
                "Show quadrat grid",
                tip("Overlay a grid to help visual inspection."),
            ),
            value=False,
        ),

        # Grid division slider (rendered dynamically)
        ui.output_ui("pdx1_grid_controls"),

        # Analysis toggle
        ui.input_checkbox(
            "pdx1_show_analysis",
            ui.TagList(
                "Show Analysis",
                tip("Show Quadrat Analysis and Nearest-Neighbour Distances charts."),
            ),
            value=False,
        ),

        # Analysis explanatory text
        ui.output_ui("pdx1_sidebar_analysis_text"),

        # Reveal toggle
        ui.input_switch(
            "pdx1_reveal",
            "Reveal Answer",
            value=False,
        ),

        # Generate button
        ui.div(
            ui.input_action_button(
                "pdx1_btn_generate", "\u21bb  New Pair",
                class_="btn-ctrl btn-sample btn-full",
            ),
            class_="sidebar-btn-row",
        ),

        class_="pdx-sidebar-group", **{"data-pdx-tab": "1"},
    )


# ── Sidebar: Tab 2 — Simpson's Paradox ───────────────────────────────────────

def _sidebar_tab2() -> ui.Tag:
    return ui.div(
        # Misconception banner
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Simpson\u2019s Paradox: "),
            "A trend that appears in several groups of data can ",
            ui.tags.em("reverse"),
            " when the groups are combined.",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "Ignoring a confounding variable (the groups) produces a "
            "misleading overall trend. Always check sub-group behaviour.",
            class_="info-banner-text",
        ),

        # Dynamic explanation
        ui.output_ui("pdx2_explanation"),

        # Number of points
        ui.input_slider(
            "pdx2_n",
            ui.TagList(
                "Points per group",
                tip("Number of observations generated within each group."),
            ),
            min=30, max=300, value=80, step=10, width="100%",
        ),

        # Number of groups
        ui.input_slider(
            "pdx2_k",
            ui.TagList(
                "Number of groups (k)",
                tip("The confounding variable with k levels. "
                    "Each group has a positive within-group slope, but the "
                    "overall slope reverses."),
            ),
            min=2, max=6, value=3, step=1, width="100%",
        ),

        # Noise level
        ui.input_slider(
            "pdx2_noise",
            ui.TagList(
                "Noise level",
                tip("Controls the scatter around each group\u2019s regression line. "
                    "Higher noise makes the paradox less obvious."),
            ),
            min=0.1, max=2.0, value=1.8, step=0.1, width="100%",
        ),

        # Show groups
        ui.input_checkbox(
            "pdx2_show_groups",
            ui.TagList(
                "Show groups (colour)",
                tip("Reveal the hidden confounding variable by "
                    "colour-coding the groups."),
            ),
            value=False,
        ),

        # Show trendlines
        ui.input_checkbox(
            "pdx2_show_trends",
            ui.TagList(
                "Show trendlines",
                tip("Overlay OLS regression lines \u2014 overall "
                    "and (when groups are shown) per-group."),
            ),
            value=False,
        ),

        # Generate button
        ui.div(
            ui.input_action_button(
                "pdx2_btn_generate", "\u21bb  New Data",
                class_="btn-ctrl btn-sample btn-full",
            ),
            class_="sidebar-btn-row",
        ),

        class_="pdx-sidebar-group", **{"data-pdx-tab": "2"},
    )


# ── Main-panel sub-tabs ─────────────────────────────────────────────────────

def _main_tab1() -> ui.Tag:
    """Clustering Illusion main panel content."""
    return ui.nav_panel(
        "Clustering Illusion",

        # Scatter plots row
        ui.div(
            ui.div(
                ui.div(ui.output_text("pdx1_title_a", inline=True),
                       class_="card-title"),
                ui.output_ui("pdx1_scatter_a"),
                class_="glass-card chart-card",
            ),
            ui.div(
                ui.div(ui.output_text("pdx1_title_b", inline=True),
                       class_="card-title"),
                ui.output_ui("pdx1_scatter_b"),
                class_="glass-card chart-card",
            ),
            style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; "
                  "margin-bottom: 8px; height: 40vh; min-height: 250px;",
        ),

        # Analysis rows (dynamic)
        ui.output_ui("pdx1_analysis_area"),
    )


def _main_tab2() -> ui.Tag:
    """Simpson's Paradox main panel content."""
    return ui.nav_panel(
        "Simpson\u2019s Paradox",

        # Stats row
        ui.output_ui("pdx2_stats_row"),

        # Main scatter
        ui.div(
            ui.div(
                ui.div(
                    ui.output_ui("pdx2_chart_title"),
                    class_="card-title",
                ),
                ui.output_ui("pdx2_scatter"),
                class_="glass-card chart-card",
            ),
            class_="charts-area",
            style="flex-direction: column;",
        ),
    )


# ── JS: toggle sidebar group visibility based on active sub-tab ─────────────

_PDX_TAB_SCRIPT = ui.tags.script("""
    (function() {
        function byTab(label) {
            if (label.indexOf('Clustering') !== -1) return '1';
            if (label.indexOf('Simpson')    !== -1) return '2';
            return '1';
        }
        function sync(active) {
            document.querySelectorAll('.pdx-sidebar-group').forEach(function(el) {
                el.style.display = (el.getAttribute('data-pdx-tab') === active)
                    ? '' : 'none';
            });
        }
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                var active = document.querySelector(
                    '#pdx_subtabs .nav-link.active, a.nav-link.active[data-bs-toggle]'
                );
                sync(byTab(active ? active.textContent : 'Clustering'));
            }, 50);
        });
        document.addEventListener('shown.bs.tab', function(e) {
            var t = e.target;
            if (!t) return;
            var container = t.closest('#pdx_subtabs');
            if (!container) return;
            sync(byTab(t.textContent || ''));
        });
    })();
""")


# ── Nav panel ───────────────────────────────────────────────────────────────

def paradox_panel() -> ui.Tag:
    return ui.nav_panel(
        "Paradox Explorer",

        _PDX_TAB_SCRIPT,

        ui.div(
            # ── Single outer sidebar ────────────────────────────────────────
            ui.div(
                _sidebar_tab1(),
                _sidebar_tab2(),

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

                class_="sidebar pdx-sidebar",
            ),

            # ── Main panel with nested navset ───────────────────────────────
            ui.div(
                ui.navset_underline(
                    _main_tab1(),
                    _main_tab2(),
                    id="pdx_subtabs",
                ),
                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
