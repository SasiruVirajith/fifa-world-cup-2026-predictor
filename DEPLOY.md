# Deploy to Streamlit Cloud

## Prerequisites

- GitHub account
- Trained models in `models/` (run `python src/models.py`)
- Processed features in `data/processed/`
- Tier 2 outputs in `outputs/` (run `python scripts/generate_outputs.py`)

## Steps

1. Create a GitHub repository and push this project:

```cmd
git add .
git commit -m "World Cup Predictor — full build"
git remote add origin https://github.com/YOUR_USERNAME/football-predictor.git
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app** and select your repository.

4. Set **Main file path** to `app.py`.

5. Python version is read from `runtime.txt` (3.12).

6. Click **Deploy**. First load may take 1-2 minutes.

## Notes

- `data/raw/` is gitignored (too large). Models and processed CSVs are committed for reliability.
- If you need to refresh data, run the pipeline locally and re-commit `data/processed/` and `models/`.
- FBref scraping often fails on cloud IPs — StatsBomb-derived features are used automatically.
