# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.build_metadata import load_build_metadata
from src.predict import (
    predict_golden_ball,
    predict_golden_boot,
    predict_golden_glove,
    predict_team_surprise,
    predict_team_upset,
    load_model,
)
from components.macos_sidebar import macos_sidebar_controls
from src.wc2026_simulator import run_wc2026_simulation

# macOS design tokens
def _build_macos_css(*, dark: bool) -> str:
    if dark:
        tokens = """
    --mac-bg: #000000;
    --mac-surface: rgba(44, 44, 46, 0.82);
    --mac-surface-solid: #1c1c1e;
    --mac-surface-elevated: #2c2c2e;
    --mac-border: rgba(255, 255, 255, 0.1);
    --mac-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
    --mac-blue: #0a84ff;
    --mac-blue-hover: #409cff;
    --mac-text: #f5f5f7;
    --mac-text-secondary: #98989d;
    --mac-text-tertiary: #636366;
    --mac-green: #30d158;
    --mac-orange: #ff9f0a;
    --mac-red: #ff453a;
    --mac-app-bg: #000000;
    --mac-sidebar-bg: #1c1c1e;
    --mac-tab-bg: rgba(118, 118, 128, 0.24);
    --mac-pill-ok-bg: rgba(48, 209, 88, 0.18);
    --mac-pill-ok-text: #30d158;
    --mac-pill-warn-bg: rgba(255, 159, 10, 0.18);
    --mac-pill-warn-text: #ff9f0a;
    --mac-card-inset: 0 0.5px 0 rgba(255, 255, 255, 0.06) inset;
    --mac-alert-bg: rgba(44, 44, 46, 0.95);
    --primary-color: #0a84ff;
    --background-color: #000000;
    --secondary-background-color: #1c1c1e;
    --text-color: #f5f5f7;
"""
    else:
        tokens = """
    --mac-bg: #f5f5f7;
    --mac-surface: rgba(255, 255, 255, 0.82);
    --mac-surface-solid: #ffffff;
    --mac-surface-elevated: #ffffff;
    --mac-border: rgba(0, 0, 0, 0.08);
    --mac-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 8px 24px rgba(0, 0, 0, 0.04);
    --mac-blue: #007aff;
    --mac-blue-hover: #0066d6;
    --mac-text: #1d1d1f;
    --mac-text-secondary: #6e6e73;
    --mac-text-tertiary: #86868b;
    --mac-green: #34c759;
    --mac-orange: #ff9500;
    --mac-red: #ff3b30;
    --mac-app-bg: linear-gradient(165deg, #e8e8ed 0%, #f5f5f7 35%, #fafafa 100%);
    --mac-sidebar-bg: #f2f2f7;
    --mac-tab-bg: rgba(118, 118, 128, 0.12);
    --mac-pill-ok-bg: rgba(52, 199, 89, 0.12);
    --mac-pill-ok-text: #248a3d;
    --mac-pill-warn-bg: rgba(255, 149, 0, 0.12);
    --mac-pill-warn-text: #c93400;
    --mac-card-inset: 0 0.5px 0 rgba(255, 255, 255, 0.8) inset;
    --mac-alert-bg: rgba(255, 255, 255, 0.9);
    --primary-color: #007aff;
    --background-color: #f5f5f7;
    --secondary-background-color: #ffffff;
    --text-color: #1d1d1f;
"""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
{tokens}
    --mac-radius: 12px;
    --mac-radius-sm: 8px;
    --mac-radius-pill: 980px;
}}

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Inter, sans-serif !important;
    color: var(--mac-text);
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {{
    background: var(--mac-app-bg) !important;
}}

.block-container {{
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1240px !important;
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background: var(--mac-sidebar-bg) !important;
    border-right: 1px solid var(--mac-border);
}}

section[data-testid="stSidebar"] > div {{
    padding: 1.5rem 1rem 1.25rem;
}}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.35rem;
}}

/* Sidebar collapse (X) */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarHeader"] button,
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] button[kind="headerNoPadding"] {{
    color: {"#f5f5f7" if dark else "#1d1d1f"} !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarHeader"] svg,
section[data-testid="stSidebar"] button[kind="header"] svg,
section[data-testid="stSidebar"] button[kind="headerNoPadding"] svg {{
    color: {"#f5f5f7" if dark else "#1d1d1f"} !important;
    fill: currentColor !important;
    stroke: none !important;
    width: 1.25rem !important;
    height: 1.25rem !important;
}}

[data-testid="collapsedControl"]:focus,
[data-testid="collapsedControl"]:focus-visible,
[data-testid="stSidebarCollapseButton"]:focus,
[data-testid="stSidebarCollapseButton"]:focus-visible,
[data-testid="stSidebarHeader"] button:focus,
[data-testid="stSidebarHeader"] button:focus-visible {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}}

.macos-hero {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 0.5rem 0 1.75rem;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--mac-border);
}}

.macos-headline {{
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--mac-text);
    margin: 0;
    line-height: 1.1;
}}

.macos-subhead {{
    font-size: 15px;
    color: var(--mac-text-secondary);
    margin: 0.4rem 0 0;
    font-weight: 400;
    line-height: 1.45;
}}

.macos-pill {{
    display: inline-flex;
    align-items: center;
    padding: 5px 12px;
    border-radius: var(--mac-radius-pill);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: -0.01em;
    border: 1px solid transparent;
}}

.macos-pill.ok {{
    background: var(--mac-pill-ok-bg);
    color: var(--mac-pill-ok-text);
}}

.macos-pill.warn {{
    background: var(--mac-pill-warn-bg);
    color: var(--mac-pill-warn-text);
}}

.macos-section-title {{
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--mac-text);
    margin: 0 0 0.35rem;
}}

.macos-section-desc {{
    font-size: 14px;
    color: var(--mac-text-secondary);
    line-height: 1.55;
    margin: 0 0 1.25rem;
}}

.sidebar-brand {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
    padding: 2.5rem 2rem 1.35rem 0;
    border-bottom: 1px solid var(--mac-border);
}}

.sidebar-eyebrow {{
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--mac-text-tertiary);
    margin: 0;
    padding: 0;
    line-height: 1.4;
}}

.sidebar-title {{
    font-size: 2.35rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
    padding: 0;
    color: var(--mac-text);
    text-transform: none;
}}

.sidebar-section {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--mac-text-tertiary);
    margin: 1.25rem 0 1rem;
}}

section[data-testid="stSidebar"] [data-testid="stCustomComponentV1"] {{
    margin: 0.25rem 0 1rem !important;
}}

.sidebar-bottom {{
    margin-top: 1.75rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--mac-border);
}}

.sidebar-footnote {{
    font-size: 11px;
    color: var(--mac-text-tertiary);
    line-height: 1.75;
    margin: 0.75rem 0 0;
}}

.sidebar-social {{
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--mac-border);
}}

.sidebar-links {{
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
}}

.sidebar-link {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 13px;
    font-weight: 500;
    color: var(--mac-text) !important;
    text-decoration: none !important;
    line-height: 1.4;
    transition: color 0.15s ease;
}}

.sidebar-link:hover {{
    color: var(--mac-blue) !important;
}}

.sidebar-link svg {{
    width: 16px;
    height: 16px;
    flex-shrink: 0;
    fill: currentColor;
    opacity: 0.9;
}}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--mac-text) !important;
}}

section[data-testid="stSidebar"] [data-testid="stCustomComponentV1"],
section[data-testid="stSidebar"] [data-testid="stCustomComponentV1"] > div,
section[data-testid="stSidebar"] iframe {{
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: var(--mac-tab-bg);
    border-radius: var(--mac-radius-sm);
    padding: 3px;
    border: none;
}}

.stTabs [data-baseweb="tab"] {{
    height: 34px;
    border-radius: 6px !important;
    background: transparent !important;
    color: var(--mac-text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0 14px !important;
    border: none !important;
}}

.stTabs [aria-selected="true"] {{
    background: var(--mac-surface-solid) !important;
    color: var(--mac-text) !important;
    box-shadow: var(--mac-shadow) !important;
    font-weight: 600 !important;
}}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.25rem; }}

.stButton > button {{
    background: var(--mac-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--mac-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

.stButton > button:hover {{
    background: var(--mac-blue-hover) !important;
    color: white !important;
}}

[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot {{
    background: transparent !important;
}}

.macos-table-wrap {{
    overflow-x: auto;
    margin: 0 -0.15rem;
}}

.macos-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
}}

.macos-table th {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--mac-text-tertiary);
    background: transparent;
    border-bottom: 1px solid var(--mac-border);
    padding: 0.55rem 0.65rem;
    text-align: left;
}}

.macos-table td {{
    color: var(--mac-text);
    background: transparent;
    border-bottom: 1px solid var(--mac-border);
    padding: 0.55rem 0.65rem;
    text-align: left;
}}

.macos-table tbody tr:last-child td {{
    border-bottom: none;
}}

.stAlert {{
    border-radius: var(--mac-radius-sm) !important;
    border: 1px solid var(--mac-border) !important;
    background: var(--mac-alert-bg) !important;
    color: var(--mac-text) !important;
}}

.macos-footer {{
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    font-size: 12px;
    color: var(--mac-text-tertiary);
}}

</style>
"""


def _build_plot_layout(*, dark: bool) -> dict:
    text = "#f5f5f7" if dark else "#1d1d1f"
    secondary = "#98989d" if dark else "#6e6e73"
    grid = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.05)"
    line = "rgba(255,255,255,0.1)" if dark else "rgba(0,0,0,0.08)"
    hover_bg = "#2c2c2e" if dark else "#ffffff"
    return dict(
        font=dict(
            family="-apple-system, BlinkMacSystemFont, SF Pro Display, Inter, sans-serif",
            size=13,
            color=text,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=16, t=16, b=8),
        xaxis=dict(
            title=dict(text=""),
            showticklabels=True,
            gridcolor=grid,
            linecolor=line,
            tickfont=dict(size=12, color=secondary),
        ),
        yaxis=dict(
            title=dict(text=""),
            gridcolor="rgba(0,0,0,0)",
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=13, color=text),
        ),
        coloraxis_showscale=False,
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=line,
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", size=12, color=text),
        ),
    )


MACOS_COLOR_SCALES = {
    "Greens": {"light": ["#e8f5ec", "#34c759"], "dark": ["#1a3d24", "#30d158"]},
    "Oranges": {"light": ["#fff4e5", "#ff9500"], "dark": ["#3d2a10", "#ff9f0a"]},
    "Blues": {"light": ["#e8f2ff", "#007aff"], "dark": ["#102a4d", "#0a84ff"]},
    "Viridis": {"light": ["#e8f0ff", "#5856d6"], "dark": ["#252450", "#5e5ce6"]},
    "Reds": {"light": ["#ffeceb", "#ff3b30"], "dark": ["#3d1816", "#ff453a"]},
}

_FAVICON = Path(__file__).parent / "src" / "img" / "favicon.webp"

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor - Sasiru Virajith",
    page_icon=str(_FAVICON),
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "top_n" not in st.session_state:
    st.session_state.top_n = 15

MODELS_LOADED = load_model("match_outcome") is not None


def _macos_plot_ranked_bar(
    df: pd.DataFrame,
    *,
    y: str,
    x: str,
    title: str,
    color_scale: str = "Blues",
    dark: bool = False,
):
    plot_df = df.sort_values(x, ascending=True).copy()
    scale = MACOS_COLOR_SCALES.get(color_scale, MACOS_COLOR_SCALES["Blues"])
    colors = scale["dark" if dark else "light"]
    vals = plot_df[x].astype(float)
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax != vmin else 1.0
    bar_colors = [
        _interp_hex(colors[0], colors[1], (v - vmin) / span)
        for v in vals
    ]

    x_label = x.replace("_", " ").title()
    fig = go.Figure(
        go.Bar(
            x=plot_df[x],
            y=plot_df[y],
            orientation="h",
            name="",
            showlegend=False,
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate=f"<b>%{{y}}</b><br>{x_label}: %{{x:.3f}}<extra></extra>",
        )
    )
    base_layout = _build_plot_layout(dark=dark)
    layout = {
        k: v for k, v in base_layout.items() if k not in ("xaxis", "yaxis")
    }
    layout["height"] = max(380, 36 * len(plot_df))
    layout["title"] = dict(text="")
    layout["bargap"] = 0.28
    layout["xaxis"] = {**base_layout["xaxis"], "title": dict(text="")}
    layout["yaxis"] = {
        **base_layout["yaxis"],
        "title": dict(text=""),
        "categoryorder": "array",
        "categoryarray": list(plot_df[y]),
    }
    fig.update_layout(**layout)
    return fig


def _interp_hex(c0: str, c1: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r = int(r0 + (r1 - r0) * t)
    g = int(g0 + (g1 - g0) * t)
    b = int(b0 + (b1 - b0) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_ranked_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<div class="macos-table-wrap"><table class="macos-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


def _sidebar_social_html() -> str:
    github_icon = (
        '<svg viewBox="0 0 16 16" aria-hidden="true">'
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
        '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
        '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 '
        '0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 '
        '0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
        '</svg>'
    )
    linkedin_icon = (
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>'
        '</svg>'
    )
    return (
        '<div class="sidebar-social"><div class="sidebar-links">'
        '<a class="sidebar-link" href="https://github.com/SasiruVirajith/" target="_blank" rel="noopener noreferrer">'
        f"{github_icon}<span>Github.com/SasiruVirajith</span></a>"
        '<a class="sidebar-link" href="https://www.linkedin.com/in/sasiru-virajith/" target="_blank" rel="noopener noreferrer">'
        f"{linkedin_icon}<span>Linkedin.com/sasiru-virajith</span></a>"
        "</div></div>"
    )


def _section_header(title: str, description: str):
    st.markdown(
        f'<p class="macos-section-title">{title}</p>'
        f'<p class="macos-section-desc">{description}</p>',
        unsafe_allow_html=True,
    )


def _ranked_tab_content(
    preds: pd.DataFrame,
    *,
    y_col: str,
    x_col: str,
    color_scale: str,
    format_fn=None,
    empty_msg: str,
    dark: bool = False,
):
    if preds.empty:
        st.warning(empty_msg)
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(
            _macos_plot_ranked_bar(
                preds,
                y=y_col,
                x=x_col,
                title="",
                color_scale=color_scale,
                dark=dark,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col2:
        display = preds.copy()
        if format_fn:
            display = format_fn(display)
        display.insert(0, "#", range(1, len(display) + 1))
        _render_ranked_table(display)


@st.cache_data
def load_striker_features():
    path = Path("data/processed/striker_features.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_goalkeeper_features():
    path = Path("data/processed/goalkeeper_features.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_wc2026_champions():
    cached = Path("outputs/wc2026_champion_probabilities.csv")
    if cached.exists():
        return pd.read_csv(cached)
    return pd.DataFrame()


# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-eyebrow">FIFA World Cup 2026<br>Predictor by</p>
            <h2 class="sidebar-title">Sasiru Virajith</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-section">Preferences</p>', unsafe_allow_html=True)
    dark_mode, top_n = macos_sidebar_controls(
        dark_mode=st.session_state.dark_mode,
        top_n=st.session_state.top_n,
        key="macos_sidebar_controls",
    )
    st.session_state.dark_mode = dark_mode
    st.session_state.top_n = top_n

    status_class = "ok" if MODELS_LOADED else "warn"
    status_label = "Model ready" if MODELS_LOADED else "Cached sims"
    build_meta = load_build_metadata()

    footnote_lines = []
    if build_meta:
        built_at = build_meta.get("built_at", "")[:19].replace("T", " ")
        max_date = build_meta.get("martj42_results_max_date")
        sims = build_meta.get("n_simulations")
        footnote_lines.append(f"Built {built_at} UTC")
        if max_date:
            footnote_lines.append(f"Matches through {max_date}")
        if sims:
            footnote_lines.append(f"{sims:,} simulations")

    footnote_html = (
        '<p class="sidebar-footnote">' + "<br>".join(footnote_lines) + "</p>"
        if footnote_lines else ""
    )
    st.markdown(
        '<div class="sidebar-bottom">'
        f'<div class="macos-pill {status_class}">{status_label}</div>'
        f"{footnote_html}"
        f"{_sidebar_social_html()}"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown(_build_macos_css(dark=st.session_state.dark_mode), unsafe_allow_html=True)

DARK = st.session_state.dark_mode

# Header
status_class = "ok" if MODELS_LOADED else "warn"
status_label = "Model loaded" if MODELS_LOADED else "Cached outputs"

st.markdown(
    f"""
    <div class="macos-hero">
        <div>
            <h1 class="macos-headline">FIFA World Cup 2026 Predictor - Sasiru Virajith</h1>
            <p class="macos-subhead">Champion odds, player awards, and team surprises</p>
        </div>
        <div class="macos-pill {status_class}">{status_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

striker_features = load_striker_features()
gk_features = load_goalkeeper_features()
year_strikers = (
    striker_features[striker_features["year"] == 2026]
    if not striker_features.empty and "year" in striker_features.columns
    else striker_features
)
year_gk = (
    gk_features[gk_features["year"] == 2026]
    if not gk_features.empty and "year" in gk_features.columns
    else gk_features
)

tab_winner, tab_boot, tab_glove, tab_ball, tab_upset, tab_surprise = st.tabs([
    "Winner",
    "Golden Boot",
    "Golden Glove",
    "Golden Ball",
    "Upset",
    "Surprise",
])

# WC 2026 Winner
with tab_winner:
    _section_header(
        "Champion probabilities",
        "Monte Carlo simulation across 12 groups of 4, top 2 + 8 best third-place teams "
        "into Round of 32, then knockouts. Powered by international match model, FIFA rankings, and form.",
    )

    wc2026 = load_wc2026_champions()
    if wc2026.empty:
        st.warning("No simulation results yet. Run: `python scripts/build_wc2026.py`")
        if st.button("Run quick simulation (500 runs)"):
            with st.spinner("Simulating tournament…"):
                run_wc2026_simulation(n_simulations=500)
            st.rerun()
    else:
        show = wc2026.head(top_n).copy()

        def _fmt_winner(df):
            out = df.copy()
            out["win_probability"] = out["win_probability"].apply(lambda x: f"{x * 100:.2f}%")
            return out[["team", "win_probability", "simulations"]]

        _ranked_tab_content(
            show,
            y_col="team",
            x_col="win_probability",
            color_scale="Greens",
            format_fn=_fmt_winner,
            empty_msg="",
            dark=DARK,
        )

# Golden Boot
with tab_boot:
    _section_header(
        "Golden Boot: top scorer",
        "International form since 2023 plus league-difficulty-adjusted club stats "
        "(Saudi/MLS discounted), scaled by team progression in the sim.",
    )
    boot_preds = predict_golden_boot(year_strikers, top_n=top_n)
    rank_col = "boot_score" if "boot_score" in boot_preds.columns else "predicted_goals"

    def _fmt_boot(df):
        out = df.copy()
        if "boot_score" in out.columns:
            out["boot_score"] = out["boot_score"].round(3)
        if "predicted_goals" in out.columns:
            out["predicted_goals"] = out["predicted_goals"].round(2)
        if "p_qualify" in out.columns:
            out["p_qualify"] = out["p_qualify"].apply(lambda x: f"{x * 100:.0f}%")
        if "progression_factor" in out.columns:
            out["progression_factor"] = out["progression_factor"].round(3)
        if "league_difficulty" in out.columns:
            out["league_difficulty"] = out["league_difficulty"].round(2)
        return out

    _ranked_tab_content(
        boot_preds,
        y_col="player",
        x_col=rank_col,
        color_scale="Oranges",
        format_fn=_fmt_boot,
        empty_msg="Run: `python scripts/build_player_2026.py --no-fetch-club --use-cache`",
        dark=DARK,
    )

# Golden Glove
with tab_glove:
    _section_header(
        "Golden Glove: best goalkeeper",
        "Primary national-team keeper plus shot-stopping form, scaled by team defensive progression.",
    )
    glove_preds = predict_golden_glove(year_gk, top_n=top_n)
    rank_col = "glove_score" if "glove_score" in glove_preds.columns else "golden_glove_probability"

    def _fmt_glove(df):
        out = df.copy()
        if "glove_score" in out.columns:
            out["glove_score"] = out["glove_score"].round(3)
        out["golden_glove_probability"] = out["golden_glove_probability"].apply(
            lambda x: f"{x * 100:.1f}%"
        )
        return out

    _ranked_tab_content(
        glove_preds,
        y_col="player",
        x_col=rank_col,
        color_scale="Blues",
        format_fn=_fmt_glove,
        empty_msg="Run: `python scripts/build_player_2026.py`",
        dark=DARK,
    )

# Golden Ball
with tab_ball:
    _section_header(
        "Golden Ball: player of the tournament",
        "Attack and creation composite, scaled by how far the player's nation is projected to go. "
        "Favors deep-run stars.",
    )
    ball_preds = predict_golden_ball(top_n=top_n)

    def _fmt_ball(df):
        out = df.copy()
        out["pot_score"] = out["pot_score"].round(3)
        if "p_qualify" in out.columns:
            out["p_qualify"] = out["p_qualify"].apply(lambda x: f"{x * 100:.0f}%")
        if "team_win_prob" in out.columns:
            out["team_win_prob"] = out["team_win_prob"].apply(lambda x: f"{x * 100:.2f}%")
        return out

    _ranked_tab_content(
        ball_preds,
        y_col="player",
        x_col="pot_score",
        color_scale="Viridis",
        format_fn=_fmt_ball,
        empty_msg="Run: `python scripts/build_player_2026.py`",
        dark=DARK,
    )

# Biggest Upset (team)
with tab_upset:
    _section_header(
        "Biggest upset (team)",
        "Major nations (FIFA top 20) rated below their FIFA rank in tournament sims, "
        "e.g. FIFA #4 but win-probability rank #10.",
    )
    upset_preds = predict_team_upset(top_n=top_n)

    def _fmt_upset(df):
        out = df.copy()
        if "upset_score" in out.columns:
            out["upset_score"] = out["upset_score"].round(3)
        if "rank_gap" in out.columns:
            out["rank_gap"] = out["rank_gap"].apply(
                lambda x: f"+{x:.0f}" if pd.notna(x) and x > 0 else x
            )
        if "team_win_prob" in out.columns:
            out["team_win_prob"] = out["team_win_prob"].apply(lambda x: f"{x * 100:.1f}%")
        if "p_qualify" in out.columns:
            out["p_qualify"] = out["p_qualify"].apply(lambda x: f"{x * 100:.0f}%")
        return out

    _ranked_tab_content(
        upset_preds,
        y_col="team",
        x_col="upset_score",
        color_scale="Reds",
        format_fn=_fmt_upset,
        empty_msg="Run: `python scripts/build_player_2026.py` or `python scripts/build_wc2026.py`",
        dark=DARK,
    )

# Biggest Surprise (team)
with tab_surprise:
    _section_header(
        "Biggest surprise (team)",
        "Underdogs (FIFA rank 28+) projected to punch above their weight: "
        "low-rated sides expected to overperform at the World Cup.",
    )
    surprise_preds = predict_team_surprise(top_n=top_n)

    def _fmt_surprise(df):
        out = df.copy()
        if "surprise_score" in out.columns:
            out["surprise_score"] = out["surprise_score"].round(3)
        if "p_qualify" in out.columns:
            out["p_qualify"] = out["p_qualify"].apply(lambda x: f"{x * 100:.0f}%")
        return out

    _ranked_tab_content(
        surprise_preds,
        y_col="team",
        x_col="surprise_score",
        color_scale="Greens",
        format_fn=_fmt_surprise,
        empty_msg="Run: `python scripts/build_player_2026.py` or `python scripts/build_wc2026.py`",
        dark=DARK,
    )

st.markdown(
    """
    <div class="macos-footer">
        Python <span>·</span> scikit-learn <span>·</span> Streamlit
        <span>·</span> martj42 <span>·</span> API-Football
    </div>
    """,
    unsafe_allow_html=True,
)
