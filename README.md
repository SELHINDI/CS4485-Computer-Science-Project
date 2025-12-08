# GDP Predictor (Final Implementation)

This repository contains the source code for the GDP Predictor application, a full-stack tool that uses machine learning (XGBoost) to forecast GDP based on Yield Curve proxies and macroeconomic indicators.

**Branch**: `final-implementation`

## Project Structure

- `CS4485-Computer-Science-Project-frontend/`: React + Vite frontend.
- `CS4485-Computer-Science-Project-backend/`: Python Flask backend + ML Models.

## Features

- **Accurate Forecasting**: Uses XGBoost Regressor with recursive forecasting for multi-year predictions.
- **Yield Curve Proxies**: Incorporates specific financial indicators as proxies for the yield curve.
- **Instant Result Caching**:
    - **Data Caching**: Raw indicators from the World Bank are cached to disk (`data/cache`) to minimize API calls and ensure offline capability.
    - **Result Caching**: Analysis results are cached in-memory by the backend, making UI switches (like checking Metrics or changing tabs) instant after the first load.
- **Interactive UI**:
    - **Dashboard**: High-level view of actual vs. predicted GDP.
    - **Explorer**: Deep dive into individual indicators (Inflation, Unemployment, etc.).
    - **Multi-Step**: View forecasts for multiple years (Horizons).
    - **Metrics**: detailed model performance stats (MAE, RMSE, etc.).

## Setup & Running

### Prerequisites
- Python 3.9+
- Node.js 16+
- Git

### 1. Backend Setup
Navigate to the backend directory:
```bash
cd CS4485-Computer-Science-Project-backend
```

Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
# Ensure xgboost and libomp are installed (Mac users might need 'brew install libomp')
```

Run the server:
```bash
python app.py
```
*The backend will start at `http://localhost:5001`.*

### 2. Frontend Setup
Open a new terminal and navigate to the frontend directory:
```bash
cd CS4485-Computer-Science-Project-frontend
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
*The frontend will start at `http://localhost:5173` (or similar).*

## How Caching Works

### Offline Data Cache
The backend is pre-loaded with offline CSVs in `data/offline`. If a required indicator isn't found there, it fetches it from the World Bank API and saves it to `data/cache/*.json` for future layout. **This branch includes pre-cached data** in `data/cache` so you can run the app smoothly without hitting external APIs immediately.

### Analysis Result Cache
When you request a prediction for a specific country and time range (e.g., "USA, 2015-2025"), the backend runs the heavy ML training once. It then stores the resulting JSON in memory.
- **First Request**: ~3-5 seconds (Training XGBoost).
- **Subsequent Requests**: <0.1 seconds (Instant).

## Troubleshooting

- **"XGBoost Library not loaded"**: On macOS, you likely need OpenMP. Run `brew install libomp`.
- **"Port already in use"**: The backend uses port 5001. If it fails to start, kill the process on that port or change the port in `app.py`.
