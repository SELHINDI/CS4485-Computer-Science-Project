import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """
    Handles machine learning models for GDP prediction
    """
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        self.trained_models = {}
        self.scalers = {}
    
    def train_and_predict(self, features_df, prediction_years=5, model_type='random_forest'):
        """Train models and make predictions"""
        from feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer()
        X, y, feature_names = feature_engineer.prepare_features_for_training(features_df, 'gdp')
        
        if X is None or len(X) < 5:
            return {
                'predictions': [],
                'performance': {'error': 'Insufficient data for training'},
                'feature_importance': {}
            }
        
        # Train model
        if model_type not in self.models:
            model_type = 'random_forest'
        
        model = self.models[model_type]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train
        model.fit(X_train, y_train)
        self.trained_models[model_type] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        performance = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Make future predictions
        predictions = self._make_future_predictions(model, features_df, prediction_years)
        
        return {
            'predictions': predictions,
            'performance': performance,
            'feature_importance': feature_importance
        }
    
    def _make_future_predictions(self, model, features_df, prediction_years):
        """Make future predictions"""
        predictions = []
        
        if features_df.empty:
            return predictions
        
        last_year = features_df['year'].max()
        
        for i in range(prediction_years):
            future_year = last_year + i + 1
            
            # Simple prediction using last known values with trend
            # In a real implementation, this would be more sophisticated
            if not features_df.empty and 'gdp' in features_df.columns:
                recent_gdp = features_df['gdp'].dropna().iloc[-1] if not features_df['gdp'].dropna().empty else 1000000000000
                growth_rate = 0.02  # Assume 2% growth
                predicted_gdp = recent_gdp * (1 + growth_rate) ** (i + 1)
                
                predictions.append({
                    'year': future_year,
                    'predicted_gdp': float(predicted_gdp),
                    'confidence': max(0.5, 0.9 - i * 0.1)
                })
        
        return predictions
