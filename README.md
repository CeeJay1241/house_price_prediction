# House Price Prediction — Resume Project

Minimal, reproducible house-price regression project. Uses the California housing dataset for a runnable demo and includes helpers to download the Ames dataset from Kaggle (if you provide `kaggle.json`).

Quickstart (Windows PowerShell):

1. Create and activate a venv:
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
2. Install deps:
   pip install -r requirements.txt
3. Run a quick smoke test (build sample CSV):
   python -m src.data --write-sample
4. Start the EDA notebook:
   jupyter lab notebooks/01-exploration.ipynb

CLI (train locally):

- Train on the included California sample (fast):
  python -m src.cli train --use-california --out models/cal_pipeline.pkl

- Train on your Ames CSV (if present in `data/raw/`):
  python -m src.cli train --data data/raw/train.csv --out models/best_pipeline.pkl

API (serve predictions):

- Start FastAPI server (after training and saving `models/best_pipeline.pkl`):
  uvicorn src.api:app --reload --port 8000

- Example request (single row):
  curl -sS -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
    -d @data/raw/sample_input.json

Docker (quick):

- Build: docker build -t houseprice:local .
- Run: docker run -p 8000:8000 houseprice:local

Files of interest:
- `src/data.py` — data loaders (California) + Kaggle helper for Ames
- `src/features.py` — preprocessing pipelines
- `src/models.py` — training & evaluation utilities
- `notebooks/` — EDA and modeling walkthroughs

If you want the **Ames** dataset downloaded, add your Kaggle API token to `%USERPROFILE%\\.kaggle\\kaggle.json` and run:

kaggle competitions download -c house-prices-advanced-regression-techniques -p data\\raw
