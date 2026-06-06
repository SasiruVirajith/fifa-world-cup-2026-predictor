"""
app.py
──────
Streamlit dashboard for the World Cup Predictor.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.predict import (
    get_playmaker_ranking,
    get_shap_explanation,
    predict_golden_boot,
    predict_golden_glove,
    predict_winner,
    load_model,
)
from src.player_tournament import build_player_tournament_scores
from src.simulator import run_group_simulation
from src.squad_depth import compute_squad_depth, counterfactual_win_impact
from src.upset_detector import predict_upsets
from src.wc2026_simulator import run_wc2026_simulation

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_LOADED = load_model("tournament_winner") is not None


def _parse_year(selection: str) -> int:
    if "2018" in selection:
        return 2018
    if "2022" in selection:
        return 2022
    if "2026" in selection:
        return 2026
    return 2026


@st.cache_data
def load_team_features():
    path = Path("data/processed/team_features.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


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
def load_group_simulation(year: int):
    cached = Path(f"outputs/group_simulation_{year}.csv")
    if cached.exists():
        return pd.read_csv(cached)
    return run_group_simulation(year=year, n_simulations=5000)


@st.cache_data
def load_upset_predictions(year: int):
    cached = Path(f"outputs/upset_predictions_{year}.csv")
    if cached.exists():
        return pd.read_csv(cached)
    return predict_upsets(year=year)


@st.cache_data
def load_pot_rankings(year: int):
    cached = Path(f"outputs/player_tournament_{year}.csv")
    if cached.exists():
        return pd.read_csv(cached)
    if year == 2026:
        return pd.DataFrame()
    return build_player_tournament_scores(year)


@st.cache_data
def load_wc2026_champions():
    cached = Path("outputs/wc2026_champion_probabilities.csv")
    if cached.exists():
        return pd.read_csv(cached)
    return pd.DataFrame()


@st.cache_data
def load_squad_depth(year: int):
    cached = Path(f"outputs/squad_depth_{year}.csv")
    if cached.exists():
        return pd.read_csv(cached)
    return compute_squad_depth(year)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/3/38/FIFA_World_Cup_Trophy.svg/200px-FIFA_World_Cup_Trophy.svg.png",
        width=80,
    )
    st.title("WC Predictor")
    st.caption("ML-powered World Cup predictions")

    st.divider()
    st.subheader("Settings")

    tournament_year = st.selectbox(
        "Tournament",
        ["2026 (forward)", "2022 (backtest)", "2018 (backtest)"],
    )
    year = _parse_year(tournament_year)
    top_n = st.slider("Teams / players to show", 3, 15, 5)

    st.divider()
    if MODELS_LOADED:
        st.success("Models loaded")
    else:
        st.warning("Using placeholder data — run `python src/models.py`")

    st.caption("2026: martj42 + FIFA + club stats")
    st.caption("Backtest: StatsBomb 2018/2022")

# ── Header ─────────────────────────────────────────────────────────────────
st.title("FIFA World Cup Predictor")
st.caption(f"Showing predictions for {tournament_year}")

all_team_features = load_team_features()
striker_features = load_striker_features()
gk_features = load_goalkeeper_features()

if not all_team_features.empty and "year" in all_team_features.columns:
    team_features = all_team_features[all_team_features["year"] == year].copy()
    if team_features.empty:
        team_features = all_team_features[all_team_features["year"] == 2022].copy()
else:
    team_features = all_team_features

if not striker_features.empty and "year" in striker_features.columns:
    year_strikers = striker_features[striker_features["year"] == year]
else:
    year_strikers = striker_features

if not gk_features.empty and "year" in gk_features.columns:
    year_gk = gk_features[gk_features["year"] == year]
else:
    year_gk = gk_features

# ── Tabs ───────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "WC 2026 Champion",
    "Winner",
    "Golden Boot",
    "Golden Glove",
    "Playmaker",
    "Explain",
    "Group Simulator",
    "Upset Detector",
    "Player of Tournament",
])

# ── Tab 0: WC 2026 Full Tournament Simulation ────────────────────────────────
with tab0:
    st.header("World Cup 2026 — Champion Probabilities")
    st.caption(
        "Monte Carlo simulation of all 48 teams in the official 2026 draw: "
        "12 groups of 4, top 2 + 8 best third-place teams → Round of 32, then knockouts. "
        "Powered by Gradient Boosting on 25,000+ international matches (2000–present)."
    )

    wc2026 = load_wc2026_champions()
    if wc2026.empty:
        st.warning(
            "No simulation results yet. Run: `python scripts/build_wc2026.py` "
            "(takes ~5 min for 5000 simulations)"
        )
        if st.button("Run quick simulation (500 runs)"):
            with st.spinner("Simulating..."):
                wc2026 = run_wc2026_simulation(n_simulations=500)
            st.rerun()
    else:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = px.bar(
                wc2026.head(15),
                x="win_probability",
                y="team",
                orientation="h",
                color="win_probability",
                color_continuous_scale="Greens",
                title="Top 15 champion probabilities (WC 2026 format)",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            display = wc2026.head(20).copy()
            display["win_probability"] = display["win_probability"].apply(
                lambda x: f"{x * 100:.2f}%"
            )
            st.dataframe(
                display[["team", "win_probability", "simulations"]],
                hide_index=True,
            )
            st.info(
                "Model uses FIFA rankings, last-5 form, head-to-head history, "
                "penalty shootout record, achievement recency, and modern football strength."
            )

# ── Tab 1: Tournament Winner ───────────────────────────────────────────────
with tab1:
    st.header("Tournament Winner Probabilities")
    if year == 2026:
        st.info(
            "For WC 2026, use the **WC 2026 Champion** tab — full 48-team Monte Carlo simulation. "
            "This tab remains for 2018/2022 backtests."
        )
    st.caption("ELO ratings, recent form, squad value, and historical WC performance.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        winner_preds = predict_winner(team_features, top_n=top_n) if year != 2026 else pd.DataFrame()
        if not winner_preds.empty:
            fig = px.bar(
                winner_preds,
                x="win_probability",
                y="team",
                orientation="h",
                color="win_probability",
                color_continuous_scale="Greens",
                title=f"Top {top_n} tournament winner predictions",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not winner_preds.empty:
            display_preds = winner_preds.copy()
            display_preds["win_probability"] = display_preds["win_probability"].apply(lambda x: f"{x * 100:.1f}%")
            st.dataframe(display_preds, use_container_width=True, hide_index=True)
            st.info("ELO rating is the single strongest predictor for tournament outcomes.")

        if year != 2026 and not team_features.empty:
            with st.expander("Squad depth & injury impact"):
                depth = load_squad_depth(year)
                if not depth.empty:
                    st.dataframe(depth[["team", "depth_score", "squad_size"]].head(10), hide_index=True)
                selected = st.selectbox("Team for counterfactual", team_features["team"].tolist(), key="cf_team")
                impact = counterfactual_win_impact(selected, year=year)
                if impact.get("baseline_prob") is not None:
                    st.write(f"Baseline win probability: **{impact['baseline_prob'] * 100:.1f}%**")
                    st.write(f"If key player out (-15% strength): **{impact['counterfactual_prob'] * 100:.1f}%**")
                    st.write(f"Drop: **{impact['drop_pct']}%**")

# ── Tab 2: Golden Boot ─────────────────────────────────────────────────────
with tab2:
    st.header("Golden Boot — Top Scorer Predictions")
    st.caption(
        "2026: weighted international goals (martj42) + club xG/goals when available. "
        "Run `python scripts/build_player_2026.py` to refresh."
    )
    boot_preds = predict_golden_boot(year_strikers, top_n=top_n)

    col1, col2 = st.columns([1.2, 1])
    if boot_preds.empty:
        st.warning(
            "No striker features for this year. Run: `python scripts/build_player_2026.py`"
        )
    else:
        with col1:
            fig = px.bar(
                boot_preds, x="predicted_goals", y="player", orientation="h",
                color="predicted_goals", color_continuous_scale="Oranges",
                title=f"Top {top_n} predicted scorers",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(boot_preds, use_container_width=True, hide_index=True)

# ── Tab 3: Golden Glove ────────────────────────────────────────────────────
with tab3:
    st.header("Golden Glove — Best Goalkeeper Predictions")
    st.caption("2026: primary national-team GK + club shot-stopping + team defensive form.")
    glove_preds = predict_golden_glove(year_gk, top_n=top_n)

    col1, col2 = st.columns([1.2, 1])
    if glove_preds.empty:
        st.warning(
            "No goalkeeper features for this year. Run: `python scripts/build_player_2026.py`"
        )
    else:
        with col1:
            fig = px.bar(
                glove_preds, x="golden_glove_probability", y="player", orientation="h",
                color="golden_glove_probability", color_continuous_scale="Blues",
                title=f"Top {top_n} goalkeeper predictions",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            display_glove = glove_preds.copy()
            display_glove["golden_glove_probability"] = display_glove["golden_glove_probability"].apply(
                lambda x: f"{x * 100:.1f}%"
            )
            st.dataframe(display_glove, use_container_width=True, hide_index=True)

# ── Tab 4: Best Playmaker ───────────────────────────────────────────────────
with tab4:
    st.header("Best Playmaker Rankings")
    playmaker_ranks = get_playmaker_ranking(top_n=top_n, year=year)

    col1, col2 = st.columns([1.2, 1])
    if playmaker_ranks.empty:
        st.warning(
            "No playmaker rankings for this year. Run: `python scripts/build_player_2026.py`"
        )
    else:
        with col1:
            fig = px.bar(
                playmaker_ranks, x="playmaker_score", y="player", orientation="h",
                color="playmaker_score", color_continuous_scale="Purples",
                title=f"Top {top_n} playmakers",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(playmaker_ranks, use_container_width=True, hide_index=True)

# ── Tab 5: Explainability ──────────────────────────────────────────────────
with tab5:
    st.header("Why did the model predict this?")
    shap_img = Path("outputs/shap_tournament_winner.png")
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP summary — tournament winner model")
    else:
        st.info("Run `python src/models.py` to generate SHAP plots.")

    if not team_features.empty:
        selected_team = st.selectbox("Select a team", team_features["team"].tolist())
        shap_vals = get_shap_explanation("tournament_winner", team_features, selected_team)
        if shap_vals:
            shap_df = pd.DataFrame({
                "feature": list(shap_vals.keys()),
                "shap_value": list(shap_vals.values()),
            }).sort_values("shap_value", key=abs, ascending=False)
            fig = px.bar(
                shap_df, x="shap_value", y="feature", orientation="h",
                color="shap_value", color_continuous_scale="RdYlGn",
                title=f"SHAP breakdown for {selected_team}",
            )
            fig.add_vline(x=0, line_width=1, line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 6: Group Simulator ─────────────────────────────────────────────────
with tab6:
    st.header("Monte Carlo Group Stage Simulator")
    st.caption("10,000 simulations using ELO-based match outcomes.")

    sim_results = load_group_simulation(year)
    if not sim_results.empty:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = px.bar(
                sim_results.head(20),
                x="p_qualify", y="team", orientation="h",
                color="p_qualify", color_continuous_scale="Teal",
                title="Probability of qualifying from group",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            display_sim = sim_results.copy()
            display_sim["p_qualify"] = display_sim["p_qualify"].apply(lambda x: f"{x * 100:.1f}%")
            display_sim["p_top_group"] = display_sim["p_top_group"].apply(lambda x: f"{x * 100:.1f}%")
            st.dataframe(
                display_sim[["team", "group", "p_qualify", "expected_points"]].head(top_n * 2),
                hide_index=True,
            )
    else:
        st.info("Group simulation data not available. Run the data pipeline first.")

# ── Tab 7: Upset Detector ──────────────────────────────────────────────────
with tab7:
    st.header("Upset Detector")
    st.caption("Matches most likely to produce giant-killings based on ELO gaps.")

    upsets = load_upset_predictions(year)
    if not upsets.empty:
        display_upsets = upsets.copy()
        display_upsets["upset_probability"] = display_upsets["upset_probability"].apply(
            lambda x: f"{x * 100:.1f}%"
        )
        st.dataframe(display_upsets, use_container_width=True, hide_index=True)
        fig = px.bar(
            upsets, x="upset_probability",
            y=upsets["home_team"] + " vs " + upsets["away_team"],
            orientation="h", color="upset_probability", color_continuous_scale="Reds",
            title="Upset probability by match",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upset predictions not available.")

# ── Tab 8: Player of the Tournament ────────────────────────────────────────
with tab8:
    st.header("Player of the Tournament")
    st.caption("Composite: goals+assists (40%), defense (25%), creation (20%), duels (15%)")

    pot = load_pot_rankings(year)
    if not pot.empty:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = px.bar(
                pot.head(top_n), x="pot_score", y="player", orientation="h",
                color="pot_score", color_continuous_scale="Viridis",
                title=f"Top {top_n} Players of the Tournament",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(
                pot[["player", "team", "goals", "assists", "pot_score"]].head(top_n),
                hide_index=True,
            )
    else:
        st.info("Player of the Tournament data not available.")

st.divider()
st.caption("Built with Python · XGBoost · SHAP · Streamlit · StatsBomb open data")
