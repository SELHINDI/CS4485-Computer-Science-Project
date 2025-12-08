from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModels

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# Core objects
# -------------------------------------------------------------------

# DataCollector now prefers OFFLINE CSVs in data/offline/indicators
data_collector = DataCollector(offline=True)

feature_engineer = FeatureEngineer()
ml_models = MLModels()

# Very small cache, mainly for the /api/health endpoint
cache = {
    "data": {},
    "models": {},
    "results": {},  # Stores final JSON responses for analysis
    "last_updated": {},
}


# -------------------------------------------------------------------
# Helper: convert numpy/pandas types to plain Python
# -------------------------------------------------------------------

def convert_numpy_types(obj):
    """Convert numpy / pandas types so they are JSON serializable."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict(orient="records"))
    if pd.isna(obj):
        return None
    return obj


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/api/countries", methods=["GET"])
def get_countries():
    """Return list of available countries."""
    try:
        countries = data_collector.get_available_countries()
        return jsonify(
            {
                "status": "success",
                "data": convert_numpy_types(countries),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/data/<country>", methods=["GET"])
def get_country_data(country):
    """
    Returns historical macro data only (no ML) for Explorer page.

    Shape:
    {
      "status": "success",
      "data": {
        "gdp": [{year, value}, ...],
        "gdp_per_capita": [...],
        "gdp_growth": [...],
        "inflation": [...],
        "unemployment": [...],
        "exports": [...],
        "imports": [...],
        "population": [...],
        "major_events": [...]
      }
    }
    """
    try:
        start_year = request.args.get("start_year", 2010, type=int)
        end_year = request.args.get("end_year", 2023, type=int)

        data = data_collector.get_country_data(country, start_year, end_year)

        return jsonify(
            {
                "status": "success",
                "data": convert_numpy_types(data),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/analyze/<country>", methods=["POST"])
def analyze_country(country):
    """
    Main analysis endpoint used by Dashboard, Multi-Step, Metrics pages.

    Returns:
    {
      status: "success",
      data: {
        historical_data: {...},
        predictions: [...],
        model_performance: {...},
        feature_importance: {...},
        analysis_metadata: {...}
      }
    }
    """
    try:
        payload = request.get_json() or {}
        start_year = int(payload.get("start_year", 2010))
        end_year = int(payload.get("end_year", 2023))
        prediction_years = int(payload.get("prediction_years", 5))
        model_type = payload.get("model_type", "random_forest")

        print(f"Analyzing {country} from {start_year} to {end_year}")

        # --- CACHE CHECK ---
        # Create a deterministic key for this request
        cache_key = (country, start_year, end_year, prediction_years, model_type)
        
        if cache_key in cache["results"]:
            print(f"Returning cached result for {country}")
            return jsonify(
                {
                    "status": "success",
                    "data": cache["results"][cache_key],
                }
            )
        # -------------------

        # 1) Historical data (offline CSVs + API fallback)
        historical_data = data_collector.get_country_data(
            country, start_year, end_year
        )
        print(
            f"Retrieved historical data with {len(historical_data)} keys "
            f"({list(historical_data.keys())})"
        )

        # 2) Feature engineering
        features_df = feature_engineer.create_features(historical_data, country)
        print(f"Features DataFrame shape: {features_df.shape}")

        # In case feature engineering produced nothing, return gracefully
        if features_df is None or features_df.empty:
            return jsonify(
                {
                    "status": "error",
                    "message": "Not enough data to train the model.",
                }
            )

        # 3) Model training + predictions
        model_results = ml_models.train_and_predict(
            features_df, prediction_years, model_type
        )
        print("Model training complete")

        response = {
            "historical_data": historical_data,
            "predictions": model_results["predictions"],
            "model_performance": model_results["performance"],
            "feature_importance": model_results.get("feature_importance", {}),
            "analysis_metadata": {
                "country": country,
                "start_year": start_year,
                "end_year": end_year,
                "prediction_years": prediction_years,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Convert to JSON-friendly format
        final_data = convert_numpy_types(response)
        
        # Store in cache
        cache["results"][cache_key] = final_data

        return jsonify(
            {
                "status": "success",
                "data": final_data,
            }
        )

    except Exception as e:
        print(f"Error in analyze_country: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/compare", methods=["POST"])
def compare_countries():
    """
    Compare multiple countries (used by Metrics / comparison views).
    """
    try:
        payload = request.get_json() or {}
        countries = payload.get("countries", [])
        start_year = int(payload.get("start_year", 2010))
        end_year = int(payload.get("end_year", 2023))

        if len(countries) < 2:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "At least 2 countries required for comparison",
                    }
                ),
                400,
            )

        comparison = {}

        for country in countries:
            try:
                data = data_collector.get_country_data(
                    country, start_year, end_year
                )
                comparison[country] = data
            except Exception as e:
                comparison[country] = {"error": str(e)}

        return jsonify(
            {
                "status": "success",
                "data": convert_numpy_types(comparison),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cache_size": {
                "data": len(cache["data"]),
                "models": len(cache["models"]),
            },
            "version": "1.0.0",
        }
    )


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Starting GDP Analysis Backend...")
    print("Available at: http://localhost:5001")

    app.run(debug=True, host="0.0.0.0", port=5001)
