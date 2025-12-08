import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost and give a clear error if missing
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError(
        "xgboost is required for this ml_models.py. Install it with `pip install xgboost`.\n"
        f"Original error: {e}"
    )


class MLModels:
    """
    Handles machine learning models for GDP prediction using XGBoost.
    """

    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': XGBRegressor(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        }

        self.trained_models = {}
        self.feature_models = {}   # XGB models for each feature
        self.feature_names = []
        self.target = "gdp"

    def train_and_predict(self, features_df, prediction_years=5, model_type='random_forest'):
        """Train models and make predictions"""
        from feature_engineer import FeatureEngineer

        feature_engineer = FeatureEngineer()
        X, y, feature_names = feature_engineer.prepare_features_for_training(
            features_df,
            target_column=self.target
        )

        self.feature_names = feature_names

        if X is None or len(X) < 5:
            return {
                'predictions': [],
                'performance': {'error': 'Insufficient data for training'},
                'feature_importance': {}
            }

        # Validate model type
        if model_type not in self.models:
            model_type = 'random_forest'

        gdp_model = self.models[model_type]

        # Split data (time-series style: no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Train GDP model
        gdp_model.fit(X_train, y_train)
        self.trained_models[model_type] = gdp_model

        # Evaluate
        y_pred = gdp_model.predict(X_test)
        performance = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        # Get feature importances (XGBoost has feature_importances_)
        feature_importance = {}
        if hasattr(gdp_model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, gdp_model.feature_importances_))
            feature_importance = dict(
                sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )

        # Train separate XGBoost forecasters for each feature (recursive forecasting)
        self._train_feature_forecasters(features_df)

        # Multi-year GDP prediction using recursive forecasting
        predictions = self._make_future_predictions(
            gdp_model,
            features_df,
            prediction_years
        )

        return {
            'predictions': predictions,
            'performance': performance,
            'feature_importance': feature_importance
        }

    def _train_feature_forecasters(self, df):
        """
        For each feature: train an XGBoost model to predict feature(t+1)
        based on all features at time t.
        """
        features = df[self.feature_names].copy()
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        self.feature_models = {}

        for feature in self.feature_names:
            # Shift 1 year ahead
            y = features[feature].shift(-1)
            X = features.copy()

            valid = y.notnull()
            X = X[valid]
            y = y[valid]

            if len(X) < 5:
                continue

            model = XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X, y)

            self.feature_models[feature] = model

    def _make_future_predictions(self, gdp_model, features_df, prediction_years):
        """
        Predict next-year features using XGBoost forecasters, feed them back in,
        then compute GDP using the GDP model. Repeat recursively.
        """
        if features_df.empty:
            return []

        # Start from the last known feature row
        last_row = features_df[self.feature_names].iloc[-1].copy()
        
        # Get baseline GDP and calculate average growth rate
        last_gdp = features_df[self.target].iloc[-1]
        baseline_growth_rate = 0.025  # Default 2.5% annual growth

        predictions = []
        current_features = last_row.copy()
        previous_gdp = last_gdp

        for step in range(1, prediction_years + 1):
            next_features = current_features.copy()

            # Predict each feature(t+1) using its own XGBoost model
            for feature, model in self.feature_models.items():
                try:
                    pred_val = model.predict(current_features.values.reshape(1, -1))[0]
                    next_features[feature] = pred_val
                except:
                    # If prediction fails, carry forward the current value
                    next_features[feature] = current_features[feature]

            # Now predict GDP using updated features
            try:
                predicted_gdp = gdp_model.predict(next_features.values.reshape(1, -1))[0]
            except:
                predicted_gdp = float("nan")

            # Apply minimum growth constraint to prevent unrealistic decline
            if not np.isnan(predicted_gdp):
                min_gdp = previous_gdp * (1 + baseline_growth_rate)
                predicted_gdp = max(predicted_gdp, min_gdp)
            else:
                predicted_gdp = previous_gdp * (1 + baseline_growth_rate)

            predictions.append({
                "year": int(features_df["year"].max() + step),
                "predicted_gdp": float(predicted_gdp),
                "confidence": max(0.5, 0.95 - 0.08 * (step - 1))
            })

            # Feed predictions into next iteration
            current_features = next_features.copy()
            previous_gdp = predicted_gdp

        return predictions
ml = MLModels()
print(ml.feature_names)