# World Cup Predictor — Project Context & Conversation Summary

> This document is a full briefing of the project concept, features, datasets, architecture, and build plan discussed between Sasiru and Claude. Intended for a coding agent to understand the full scope and continue development.

---

## Project Overview

An ML-powered FIFA World Cup prediction system built in Python. The goal is to predict tournament outcomes using real football statistics, with a Streamlit dashboard and SHAP explainability layer. The project is portfolio-focused — aimed at standing out in job applications by demonstrating data engineering, ML modelling, and explainability skills.

**Developer:** Sasiru (Software Engineering undergraduate, University of Westminster via IIT, Colombo, Sri Lanka)  
**Experience level:** Intermediate Python/ML — comfortable with pandas and scikit-learn, not yet used XGBoost or SHAP in a real project  
**Environment:** Windows, VS Code, local machine  

---

## Core Predictions

| Award | Model type | Key features |
|---|---|---|
| Tournament winner | XGBoost classifier | ELO rating, squad value, recent form, group difficulty |
| Golden Boot (top scorer) | XGBoost regressor | xG per 90, shots on target, club season goals, tournament stage |
| Golden Glove (best keeper) | XGBoost classifier | PSxG, save %, clean sheets, GA vs expected |
| Best Playmaker | Weighted composite ranking | xA per 90, key passes, progressive passes, pass completion % |

---

## Full Feature Brainstorm

These were discussed and grouped into three tiers:

### Tier 1 — Core MVP
- Tournament winner predictor
- Golden Boot predictor
- Golden Glove predictor
- Best Playmaker predictor

### Tier 2 — Standout portfolio additions
- **Group stage Monte Carlo simulator** — run 10,000 scenario simulations of the 48 group matches, produce probabilistic standings
- **Player of the Tournament predictor** — composite score across goals, assists, duels, defensive actions, carry distance
- **Upset detector** — classify which matches are most likely to produce giant-killings based on ELO gaps and squad volatility
- **SHAP explainability layer** — show WHY the model made each prediction (which features drove the call)
- **Live data pipeline** — ingest real squad announcements and club form via FBref/Transfermarkt scraping or StatsBomb API
- **Squad depth score** — rate depth at each position, model what happens to win probability if key player is suspended/injured

### Tier 3 — Stretch goals
- **Interactive Streamlit dashboard** — bracket simulator, probability bars, SHAP charts, deployed on Streamlit Cloud
- **Sentiment as a feature** — scrape pre-match Twitter/Reddit hype scores, test if public mood has predictive signal
- **Penalty shootout model** — separate model for knockout tiebreakers using historical pen stats and goalkeeper dive tendency
- **What-if / counterfactual mode** — let users swap players in/out or change group fixtures and re-run simulation in real time

---

## Datasets

### Start with these (core)

| Dataset | Source | Format | Notes |
|---|---|---|---|
| StatsBomb open data | `github.com/statsbomb/open-data` | Python library (`statsbombpy`) | Free event-level data for WC 2018 and 2022 — every pass, shot, dribble with x/y coordinates. Includes xG and xA. Best starting point. |
| FBref player stats | `fbref.com` via `soccerdata` library | Scraped DataFrame | Most complete player stats on the web — xG, npxG, progressive carries, key passes, PSxG for keepers. `soccerdata` wraps it cleanly. |
| Kaggle WC historical results | `kaggle.com/datasets/abecklas/fifa-world-cup` | CSV | Every WC match result from 1930–2022 — teams, scores, stages. Clean, no scraping. |
| World Football ELO ratings | `eloratings.net` | Scraped / CSV | ELO ratings for every international team. Single most predictive feature for tournament outcomes. Also available via `github.com/martj42/international_results` |

### Add for richer features

| Dataset | Source | Notes |
|---|---|---|
| football-data.org API | `football-data.org` | Free tier — live and historical results, standings, squad data via REST API |
| Transfermarkt squad values | `transfermarkt.com` via `soccerdata` | Squad market value = strong proxy for squad quality. Also has age profiles and injury history. |
| FIFA / EA FC ratings | `kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset` | Pre-cleaned CSVs. Overall rating, pace, shooting, passing, defending per player going back to FIFA 15. |

### Bonus / advanced

| Dataset | Source | Notes |
|---|---|---|
| StatsBomb 360 data | `statsbombpy` | Freeze-frame data — positions of all 22 players at every event. Enables spatial pressure features. Free for some competitions. |
| Understat | `understat.com` | Shot-level xG for top European leagues. Unofficial Python wrapper `understat` on PyPI. Good for club form features. |

---

## Tech Stack

```
Language:       Python 3.11
Data:           pandas, numpy
Football data:  statsbombpy, soccerdata
ML:             scikit-learn, xgboost
Explainability: shap
Visualisation:  matplotlib, seaborn, plotly
Dashboard:      streamlit (deploy free to Streamlit Cloud)
Notebooks:      jupyter
Utilities:      requests, beautifulsoup4, tqdm, python-dotenv
```

---

## Project Structure (scaffolded)

The full scaffold has been built and provided as a downloadable zip (`wc-predictor.zip`). Structure:

```
wc-predictor/
├── app.py                       ← Streamlit dashboard (5 tabs, works with placeholder data)
├── requirements.txt             ← all dependencies pinned
├── README.md
├── .gitignore
├── data/
│   ├── raw/                     ← downloaded datasets (gitignored)
│   └── processed/               ← cleaned feature CSVs
├── notebooks/
│   ├── 01_eda.ipynb             ← exploratory data analysis
│   ├── 02_features.ipynb        ← feature engineering experiments
│   └── 03_modelling.ipynb       ← model training and evaluation
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         ← fetches raw data from all sources
│   ├── features.py              ← ELO calculation + feature matrix building
│   ├── models.py                ← trains and saves all XGBoost models
│   └── predict.py               ← loads saved models, returns predictions
├── models/                      ← saved .pkl model files
└── outputs/                     ← SHAP plots, prediction CSVs
```

---

## What Each Source File Does

### `src/data_pipeline.py`
- Fetches WC 2022 and 2018 match lists from StatsBomb (`competition_id=43`, `season_id=106` for 2022, `season_id=3` for 2018)
- Fetches all match events from StatsBomb (passes, shots, tackles per match)
- Downloads international match results CSV from `github.com/martj42/international_results` (used for ELO calculation)
- Fetches FBref player stats (shooting, passing, goalkeeper) via `soccerdata`
- Fetches Transfermarkt squad market values via `soccerdata`

### `src/features.py`
- `calculate_elo_ratings(results_df)` — computes ELO ratings for all teams from scratch using the standard ELO update formula (K=32)
- `build_team_features()` — merges ELO + recent form (last 20 matches) + squad value into one DataFrame per team
- `build_striker_features()` — loads FBref shooting CSV, flattens multi-level columns, outputs to `data/processed/striker_features.csv`
- `build_goalkeeper_features()` — same for keeper stats (PSxG, save %, clean sheets, GA vs expected)
- `build_playmaker_features()` — same for passing stats (xA, key passes, progressive passes)

### `src/models.py`
- `train_winner_model()` — XGBoost classifier on team features, needs `won_tournament` label column in team_features.csv
- `train_golden_boot_model()` — XGBoost regressor, needs `tournament_goals` label column in striker_features.csv
- `train_golden_glove_model()` — XGBoost classifier, needs `won_golden_glove` label column
- `rank_playmakers()` — no ML training, just weighted composite score (xA 40%, key passes 30%, progressive passes 20%, pass completion 10%)
- `plot_shap_summary()` — generates and saves SHAP feature importance plots to `outputs/`
- `save_model()` / `load_model()` — pickle serialisation helpers

### `src/predict.py`
- `predict_winner(team_features, top_n)` — returns DataFrame of teams + win probabilities
- `predict_golden_boot(player_features, top_n)` — returns top predicted scorers + goal counts
- `predict_golden_glove(keeper_features, top_n)` — returns keepers + award probabilities
- `get_playmaker_ranking(top_n)` — reads from `outputs/playmaker_rankings.csv`
- `get_shap_explanation(model_name, team_features, team)` — returns per-feature SHAP values for a single team
- All functions fall back to **placeholder/dummy data** if models aren't trained yet, so the dashboard always renders

### `app.py` (Streamlit dashboard)
- Tab 1: Tournament winner — horizontal bar chart of win probabilities
- Tab 2: Golden Boot — top predicted scorers bar chart
- Tab 3: Golden Glove — top keeper predictions bar chart
- Tab 4: Best Playmaker — composite score ranking bar chart
- Tab 5: Explainability — SHAP summary plot + per-team SHAP breakdown chart
- Sidebar: tournament year selector, top-N slider
- Uses `@st.cache_data` for performance

---

## Build Order

1. Set up environment (Python 3.11 venv, `pip install -r requirements.txt`)
2. Run `streamlit run app.py` — dashboard works immediately with dummy data
3. Run `python src/data_pipeline.py` — fetches all raw data into `data/raw/`
4. Open `notebooks/01_eda.ipynb` — explore what you have
5. Open `notebooks/02_features.ipynb` — build and test feature engineering
6. Add target variable labels (`won_tournament`, `tournament_goals`, `won_golden_glove`) to processed CSVs
7. Open `notebooks/03_modelling.ipynb` — train, evaluate, and tune models
8. Run `python src/models.py` — trains all models, saves to `models/`, generates SHAP plots
9. Dashboard auto-updates with real predictions
10. Deploy to Streamlit Cloud for a live public URL

---

## Key TODOs for the Coding Agent

These are the things that require manual work or are currently stubbed out:

1. **Add target variable labels** — the three model training functions will early-exit with a clear error message until these columns exist:
   - `team_features.csv` needs a `won_tournament` column (1 = that team won WC that year, 0 otherwise)
   - `striker_features.csv` needs a `tournament_goals` column (goals scored in the tournament)
   - `goalkeeper_features.csv` needs a `won_golden_glove` column (1 = award winner)
   - These are derived from StatsBomb event data + historical WC records — see `notebooks/02_features.ipynb` for the extraction code (currently commented out)

2. **Flatten FBref multi-level column headers** — FBref via `soccerdata` returns DataFrames with multi-level columns like `('Standard', 'Goals')`. The `features.py` file already handles this with `"_".join(col)`, but the resulting column names need to be mapped to the expected names (`xg_per90`, `shots_per90`, etc.) — this mapping depends on the actual column names returned by your version of `soccerdata` and should be inspected in `notebooks/01_eda.ipynb` first.

3. **Historical backtest data** — for the winner model to be meaningful, you need team features across multiple World Cup years (1990–2022), not just 2022. The `calculate_elo_ratings()` function supports this — filter `international_results.csv` by year and join with WC winners from the `wc_winners` dict in `notebooks/03_modelling.ipynb`.

4. **Monte Carlo group stage simulator** — not yet built, planned as Tier 2 feature. Should use win probabilities from the winner model to simulate 48 group stage matches 10,000 times and output probabilistic group standings.

5. **Streamlit Cloud deployment** — once models are trained, push to GitHub and connect to `streamlit.io/cloud` for a free live public URL. The `requirements.txt` is already formatted correctly for this.

---

## Important Notes for the Agent

- All `src/` functions have early-exit guards with clear error messages if data files are missing — run `data_pipeline.py` before `features.py` before `models.py`
- The dashboard (`app.py`) uses placeholder/dummy data for all four prediction tabs until real models are trained — this is intentional so the UI can be developed in parallel
- Windows path separators — use `pathlib.Path` throughout (already done in scaffold), avoid hardcoded `/` separators
- FBref rate-limits scraping — `soccerdata` handles this but add `time.sleep()` delays if you hit 429 errors
- StatsBomb event data is large (~several hundred MB for a full tournament) — the pipeline saves per-match CSVs to `data/raw/events_wc2022/` before combining

