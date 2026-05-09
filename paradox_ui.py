# =============================================================================
# Paradox Explorer — UI panel
#
# Sub-tab layout:
#   Tab 1  Base Rate Fallacy     (pdx_brf_*)
#   Tab 2  Clustering Illusion   (pdx_clust_*)
#   Tab 3  Simpson's Paradox     (pdx_simp_*)
#   Tab 4  Coupon Collector       (pdx_ccp_*)
# =============================================================================

from shiny import ui
from utils import tip


# ── Sidebar: Tab 1 — Base Rate Fallacy ───────────────────────────────────────

def _sidebar_tab_brf() -> ui.Tag:
    return ui.div(
        # Misconception banner
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Misconception: "),
            "If a test is 99\u202f% accurate and you test positive, "
            "you have a 99\u202f% chance of having the disease.",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "When a condition is rare (low base rate), the majority of "
            "positive results can be false positives, even with a highly "
            "accurate test.",
            class_="info-banner-text",
        ),

        # Presets
        ui.tags.label(
            "Examples",
            style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
        ),
        ui.div(
            ui.input_action_button("pdx_brf_pre_disease", "Rare Disease", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_brf_pre_terror", "Facial Recog.", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_brf_pre_quality", "Quality Ctrl", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_brf_pre_lottery", "Lottery Fraud", class_="btn-ctrl btn-preset"),
            class_="np-preset-grid",
            style="grid-template-columns: 1fr 1fr;",
        ),
        ui.output_ui("pdx_brf_preset_desc"),

        ui.div(
            ui.div("BAYES' THEOREM", class_="card-title"),
            ui.output_ui("pdx_brf_formula"),
            class_="glass-card formulas-card",
            style="margin-bottom: 8px; text-align: center; overflow-x: auto;",
        ),

        # Population
        ui.input_select(
            "pdx_brf_pop",
            ui.TagList(
                "Population Size",
                tip("Total number of people being tested. "
                    "Natural frequencies make the math intuitive."),
            ),
            choices={"10000": "10,000", "100000": "100,000",
                     "1000000": "1,000,000"},
            selected="10000", width="100%",
        ),

        # Prevalence
        ui.input_numeric(
            "pdx_brf_prev",
            ui.TagList(
                "Prevalence (Base Rate) %",
                tip("The percentage of the population that actually has the "
                    "disease. Try very small values like 0.01\u202f%."),
            ),
            value=1, min=0.001, max=100.0, step=0.01, width="100%",
        ),

        # Sensitivity
        ui.input_slider(
            "pdx_brf_sens",
            ui.TagList("Sensitivity % (TPR)",
                       tip("True Positive Rate: P(+\u2009|\u2009Disease). "
                           "Probability that a sick person tests positive.")),
            min=50, max=100, value=99, step=1, width="100%",
        ),

        # Specificity
        ui.input_slider(
            "pdx_brf_spec",
            ui.TagList("Specificity % (TNR)",
                       tip("True Negative Rate: P(\u2212\u2009|\u2009Healthy). "
                           "Probability that a healthy person tests negative.")),
            min=50, max=100, value=95, step=1, width="100%",
        ),

        class_="pdx-sidebar-group", **{"data-pdx-tab": "brf"},
    )


# ── Sidebar: Tab 2 — Clustering Illusion ─────────────────────────────────────

def _sidebar_tab_clust() -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Clustering Illusion: "),
            "Humans see patterns and clusters in random data \u2014 our brains "
            "are wired to find structure even where none exists.",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "True randomness produces ",
            ui.tags.em("clumps and gaps"), " by chance. "
            "A perfectly uniform spread would actually be ",
            ui.tags.em("non-random"), ".",
            class_="info-banner-text",
        ),

        ui.div(
            "Can you tell which one is purely random and which one is "
            "stratified (regular)? Our brains often pick the stratified "
            "one as \u2018random\u2019 because it looks more evenly spread.",
            style="margin-bottom: 16px; font-size: 0.82rem; "
                  "font-style: italic; color: var(--c-text3); "
                  "line-height: 1.4;",
        ),

        ui.input_numeric(
            "pdx_clust_n",
            ui.TagList("Number of points (N)",
                       tip("Total points to scatter on each plot.")),
            value=200, min=10, max=2000, step=10, width="100%",
        ),

        ui.input_checkbox("pdx_clust_show_grid",
                          "Show quadrat grid", value=False),
        ui.output_ui("pdx_clust_grid_controls"),

        ui.input_checkbox("pdx_clust_show_analysis",
                          "Show Analysis", value=False),
        ui.output_ui("pdx_clust_sidebar_analysis_text"),

        ui.input_switch("pdx_clust_reveal",
                        "Reveal Answer", value=False),

        ui.div(
            ui.input_action_button(
                "pdx_clust_btn_generate", "\u21bb  New Pair",
                class_="btn-ctrl btn-sample btn-full",
            ),
            class_="sidebar-btn-row",
        ),

        class_="pdx-sidebar-group", **{"data-pdx-tab": "clust"},
    )


# ── Sidebar: Tab 3 — Simpson's Paradox ───────────────────────────────────────

def _sidebar_tab_simp() -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Simpson\u2019s Paradox: "),
            "A trend that appears in several groups of data can ",
            ui.tags.em("reverse"), " when the groups are combined.",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "Ignoring a confounding variable (the groups) produces "
            "a misleading overall trend. Always check sub-group behaviour.",
            class_="info-banner-text",
        ),

        ui.output_ui("pdx_simp_explanation"),

        ui.input_slider(
            "pdx_simp_n",
            ui.TagList("Points per group",
                       tip("Number of observations in each sub-group.")),
            min=30, max=300, value=80, step=10, width="100%",
        ),
        ui.input_slider(
            "pdx_simp_k",
            ui.TagList("Number of groups (k)",
                       tip("Number of hidden sub-groups (confounders).")),
            min=2, max=6, value=3, step=1, width="100%",
        ),
        ui.input_slider(
            "pdx_simp_noise",
            ui.TagList("Noise level",
                       tip("Controls the scatter around each group\u2019s "
                           "regression line. Higher noise makes the paradox "
                           "less obvious.")),
            min=0.1, max=2.0, value=1.8, step=0.1, width="100%",
        ),

        ui.input_checkbox("pdx_simp_show_groups",
                          "Show groups (colour)", value=False),
        ui.input_checkbox("pdx_simp_show_trends",
                          "Show trendlines", value=False),

        ui.div(
            ui.input_action_button(
                "pdx_simp_btn_generate", "\u21bb  New Data",
                class_="btn-ctrl btn-sample btn-full",
            ),
            class_="sidebar-btn-row",
        ),

        class_="pdx-sidebar-group", **{"data-pdx-tab": "simp"},
    )


# ── Main-panel sub-tabs ─────────────────────────────────────────────────────

def _main_tab_brf() -> ui.Tag:
    return ui.nav_panel(
        "Base Rate Fallacy",

        # Stats row
        ui.output_ui("pdx_brf_stats_row"),

        # View selector and Chart area
        ui.div(
            ui.div(
                ui.input_radio_buttons(
                    "pdx_brf_view", "",
                    choices=["Icon Array (Waffle)", "Probability Flow (Sankey)"],
                    inline=True
                )
            ),
            ui.div(
                ui.output_ui("pdx_brf_chart_area"),
                style="flex: 1; min-height: 450px; position: relative; z-index: 1;"
            ),
            class_="glass-card",
            style="flex: 1; min-height: 0; overflow: hidden; display: flex; flex-direction: column; padding-top: 12px;",
        ),
    )


def _main_tab_clust() -> ui.Tag:
    return ui.nav_panel(
        "Clustering Illusion",
        ui.div(
            ui.div(
                ui.div(ui.output_text("pdx_clust_title_a", inline=True),
                       class_="card-title"),
                ui.output_ui("pdx_clust_scatter_a"),
                class_="glass-card chart-card",
            ),
            ui.div(
                ui.div(ui.output_text("pdx_clust_title_b", inline=True),
                       class_="card-title"),
                ui.output_ui("pdx_clust_scatter_b"),
                class_="glass-card chart-card",
            ),
            style="display: grid; grid-template-columns: 1fr 1fr; "
                  "gap: 8px; margin-bottom: 8px; "
                  "height: 40vh; min-height: 250px;",
        ),
        ui.output_ui("pdx_clust_analysis_area"),
    )


def _main_tab_simp() -> ui.Tag:
    return ui.nav_panel(
        "Simpson\u2019s Paradox",
        ui.output_ui("pdx_simp_stats_row"),
        ui.div(
            ui.div(
                ui.div(ui.output_ui("pdx_simp_chart_title"),
                       class_="card-title"),
                ui.output_ui("pdx_simp_scatter"),
                class_="glass-card chart-card",
            ),
            class_="charts-area",
            style="flex-direction: column;",
        ),
    )



# ── Sidebar: Tab 4 — Coupon Collector's Problem ─────────────────────────────

def _sidebar_tab_ccp() -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Misconception: "),
            "I have 90% of the collection, so I'll finish it soon.",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "Getting the last 10% takes exactly as long as getting the first 90%.",
            class_="info-banner-text",
        ),

        ui.tags.label(
            "Examples",
            style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
        ),
        ui.div(
            ui.input_action_button("pdx_ccp_pre_d6", "D6 Dice", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_ccp_pre_d20", "D20 Set", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_ccp_pre_mtg_c", "MtG Common", class_="btn-ctrl btn-preset"),
            ui.input_action_button("pdx_ccp_pre_poke", "Pokémon", class_="btn-ctrl btn-preset"),
            class_="np-preset-grid",
            style="grid-template-columns: 1fr 1fr;",
        ),
        ui.output_ui("pdx_ccp_preset_desc"),

        ui.input_slider(
            "pdx_ccp_n",
            ui.TagList("Unique Items to Collect (N)", tip("Total number of unique items in the set.")),
            min=2, max=300, value=20, step=1, width="100%",
        ),
        ui.input_slider(
            "pdx_ccp_b",
            ui.TagList("Booster Size (B)", tip("Number of unique items per pack/draw.")),
            min=1, max=30, value=1, step=1, width="100%",
        ),
        ui.input_numeric(
            "pdx_ccp_cost",
            ui.TagList("Cost per pack (optional)", tip("Enter price to calculate total expected cost.")),
            value=0, min=0, step=0.1,
        ),
        
        ui.div(
            ui.div("EXPECTED DRAWS", class_="card-title"),
            ui.output_ui("pdx_ccp_formula"),
            ui.div(
                ui.output_ui("pdx_ccp_formula_note"),
                style="text-align:left; margin-top:6px;",
            ),
            class_="glass-card formulas-card",
            style="margin-bottom: 8px; text-align: center; overflow-x: auto;",
        ),

        class_="pdx-sidebar-group",
        **{"data-pdx-tab": "ccp"},
    )


# ── Main: Tab 4 — Coupon Collector's Problem ────────────────────────────────

def _main_tab_ccp() -> ui.Tag:
    return ui.nav_panel(
        "Coupon Collector",
        
        ui.div(
            ui.div(
                ui.div(
                    "EXPECTED PACKS ",
                    tip("Exact expected number of packs (draws when B = 1) to collect "
                        "all N unique items. Computed via inclusion–exclusion; "
                        "equals N·Hₙ for B = 1."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pdx_ccp_exp_packs"), class_="stat-value", style="color: #10b981;"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "EXPECTED TOTAL ITEMS ",
                    tip("Total items opened on average: E[Packs] × B. "
                        "Includes all duplicates — most of which you will have already seen."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pdx_ccp_exp_items"), class_="stat-value", style="color: #0ea5e9;"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "1st HALF vs LAST ITEM ",
                    tip("Left: expected packs to collect the first N/2 unique items. "
                        "Right: expected packs when only 1 item remains (always N/B). "
                        "Comparing them reveals how steeply the cost grows near the end."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pdx_ccp_cost_cmp"), class_="stat-value", style="color: #f59e0b; font-size: 1.5rem;"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "TOTAL COST ",
                    tip("Expected total spend: E[Packs] × cost per pack. "
                        "Set a non-zero cost per pack in the sidebar to activate."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pdx_ccp_total_cost"), class_="stat-value", style="color: #ef4444;"),
                class_="stat-card",
            ),
            class_="stats-row",
        ),

        ui.div(
            ui.div(
                ui.input_radio_buttons(
                    "pdx_ccp_view", "",
                    choices=["Hockey-stick Curve", "Simulation (Distribution)", "Probability Curve (CDF)"],
                    inline=True
                ),
                id="pdx_ccp_view_container"
            ),
            ui.div(
                ui.output_ui("pdx_ccp_chart_caption"),
                style="padding: 0 12px;",
            ),
            ui.div(
                ui.output_ui("pdx_ccp_chart_area"),
                style="flex: 1; min-height: 420px; position: relative; z-index: 1;"
            ),
            class_="glass-card",
            style="flex: 1; min-height: 0; overflow: hidden; display: flex; flex-direction: column; padding-top: 12px;",
        ),
    )


# ── JS: toggle sidebar group visibility based on active sub-tab ─────────────

_PDX_TAB_SCRIPT = ui.tags.script("""
    (function() {
        function byTab(label) {
            if (label.indexOf('Base Rate') !== -1) return 'brf';
            if (label.indexOf('Clustering') !== -1) return 'clust';
            if (label.indexOf('Simpson')    !== -1) return 'simp';
            if (label.indexOf('Coupon')     !== -1) return 'ccp';
            return 'brf';
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
                    '#pdx_subtabs .nav-link.active'
                );
                sync(byTab(active ? active.textContent : 'Base Rate'));
            }, 50);
        });
        // Only respond to outer sub-tab changes (not inner viz tabs)
        document.addEventListener('shown.bs.tab', function(e) {
            var t = e.target;
            if (!t) return;
            var container = t.closest('#pdx_subtabs');
            if (!container) return;
            sync(byTab(t.textContent || ''));
        });
        // Re-typeset MathJax whenever Shiny updates the formula output
        if (window.jQuery) {
            $(document).on('shiny:value', function(event) {
                if (event.name === 'pdx_brf_formula' || event.name === 'pdx_ccp_formula') {
                    setTimeout(function() {
                        if (window.MathJax && MathJax.typesetPromise) {
                            MathJax.typesetPromise();
                        }
                    }, 50);
                }
            });
        }
    })();
""")


# ── MathJax config (loaded ONCE, before the library) ────────────────────────

_MATHJAX_CONFIG = ui.tags.script("""
    window.MathJax = {
        tex:  { inlineMath: [['$','$'], ['\\\\(','\\\\)']] },
        options: { skipHtmlTags: ['script','noscript','style','textarea','code'] }
    };
""")

_MATHJAX_SRC = ui.tags.script(
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
    async_="",
)


# ── Nav panel ───────────────────────────────────────────────────────────────

def paradox_panel() -> ui.Tag:
    return ui.nav_panel(
        "Paradox Explorer",

        _MATHJAX_CONFIG,
        _MATHJAX_SRC,
        _PDX_TAB_SCRIPT,

        ui.div(
            ui.div(
                _sidebar_tab_brf(),
                _sidebar_tab_clust(),
                _sidebar_tab_simp(),
                _sidebar_tab_ccp(),

                ui.div(
                    ui.tags.a("LinkedIn",
                              href="https://www.linkedin.com/in/"
                                   "ihormiroshnychenko/",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Telegram",
                              href="https://t.me/araprof",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Website",
                              href="https://aranaur.rbind.io/",
                              target="_blank"),
                    class_="footer-links",
                ),
                class_="sidebar pdx-sidebar",
            ),
            ui.div(
                ui.navset_underline(
                    _main_tab_brf(),
                    _main_tab_clust(),
                    _main_tab_simp(),
                    _main_tab_ccp(),
                    id="pdx_subtabs",
                ),
                class_="main-panel",
            ),
            class_="app-body",
        ),
    )
