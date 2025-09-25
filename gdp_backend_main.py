from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModels

app = Flask(__name__)
CORS(app)

# Global variables
data_collector = DataCollector()
feature_engineer = FeatureEngineer()
ml_models = MLModels()

# Cache for storing data and models
cache = {
    'data': {},
    'models': {},
    'last_updated': {}
}

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get list of available countries"""
    try:
        countries = data_collector.get_available_countries()
        return jsonify({
            'status': 'success',
            'data': countries
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/data/<country>', methods=['GET'])
def get_country_data(country):
    """Get historical data for a specific country"""
    try:
        start_year = request.args.get('start_year', 1990, type=int)
        end_year = request.args.get('end_year', 2023, type=int)
        
        # Check cache
        cache_key = f"{country}_{start_year}_{end_year}"
        if cache_key in cache['data']:
            if datetime.now() - cache['last_updated'].get(cache_key, datetime.min) < timedelta(hours=24):
                return jsonify({
                    'status': 'success',
                    'data': cache['data'][cache_key],
                    'cached': True
                })
        
        # Fetch fresh data
        data = data_collector.get_country_data(country, start_year, end_year)
        
        # Cache the data
        cache['data'][cache_key] = data
        cache['last_updated'][cache_key] = datetime.now()
        
        return jsonify({
            'status': 'success',
            'data': data,
            'cached': False
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/analyze/<country>', methods=['POST'])
def analyze_country(country):
    """Perform comprehensive analysis for a country"""
    try:
        request_data = request.get_json()
        start_year = request_data.get('start_year', 1990)
        end_year = request_data.get('end_year', 2023)
        prediction_years = request_data.get('prediction_years', 5)
        model_type = request_data.get('model_type', 'ensemble')
        
        # Get historical data
        historical_data = data_collector.get_country_data(country, start_year, end_year)
        
        # Engineer features
        features_data = feature_engineer.create_features(historical_data, country)
        
        # Train model and make predictions
        model_key = f"{country}_{model_type}_{start_year}_{end_year}"
        
        # Check if we have a cached model
        if model_key not in cache['models'] or datetime.now() - cache['last_updated'].get(model_key, datetime.min) > timedelta(hours=6):
            model_results = ml_models.train_and_predict(
                features_data, 
                prediction_years, 
                model_type
            )
            cache['models'][model_key] = model_results
            cache['last_updated'][model_key] = datetime.now()
        else:
            model_results = cache['models'][model_key]
        
        # Prepare response
        response = {
            'historical_data': historical_data,
            'features': features_data,
            'predictions': model_results['predictions'],
            'model_performance': model_results['performance'],
            'feature_importance': model_results.get('feature_importance', {}),
            'analysis_metadata': {
                'country': country,
                'start_year': start_year,
                'end_year': end_year,
                'prediction_years': prediction_years,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': response
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_countries():
    """Compare multiple countries"""
    try:
        request_data = request.get_json()
        countries = request_data.get('countries', [])
        start_year = request_data.get('start_year', 1990)
        end_year = request_data.get('end_year', 2023)
        
        if len(countries) < 2:
            return jsonify({
                'status': 'error',
                'message': 'At least 2 countries required for comparison'
            }), 400
        
        comparison_data = {}
        
        for country in countries:
            try:
                data = data_collector.get_country_data(country, start_year, end_year)
                comparison_data[country] = data
            except Exception as e:
                comparison_data[country] = {'error': str(e)}
        
        return jsonify({
            'status': 'success',
            'data': comparison_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/validate/<country>', methods=['POST'])
def validate_model(country):
    """Validate model against historical data"""
    try:
        request_data = request.get_json()
        train_end_year = request_data.get('train_end_year', 2018)
        test_end_year = request_data.get('test_end_year', 2023)
        model_type = request_data.get('model_type', 'ensemble')
        
        # Get historical data
        full_data = data_collector.get_country_data(country, 1990, test_end_year)
        
        # Split data
        train_data = {k: [item for item in v if item.get('year', 0) <= train_end_year] 
                     for k, v in full_data.items()}
        test_data = {k: [item for item in v if item.get('year', 0) > train_end_year] 
                    for k, v in full_data.items()}
        
        # Engineer features for training data
        train_features = feature_engineer.create_features(train_data, country)
        
        # Train model
        validation_results = ml_models.validate_model(
            train_features, 
            test_data, 
            model_type,
            country
        )
        
        return jsonify({
            'status': 'success',
            'data': validation_results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': {
            'data': len(cache['data']),
            'models': len(cache['models'])
        }
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache"""
    cache['data'].clear()
    cache['models'].clear()
    cache['last_updated'].clear()
    
    return jsonify({
        'status': 'success',
        'message': 'Cache cleared successfully'
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)