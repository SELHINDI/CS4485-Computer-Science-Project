from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModels

app = Flask(__name__)
CORS(app)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

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

def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get list of available countries"""
    try:
        countries = data_collector.get_available_countries()
        return jsonify({
            'status': 'success',
            'data': convert_numpy_types(countries)
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
        start_year = request.args.get('start_year', 2010, type=int)
        end_year = request.args.get('end_year', 2023, type=int)
        
        data = data_collector.get_country_data(country, start_year, end_year)
        
        return jsonify({
            'status': 'success',
            'data': convert_numpy_types(data)
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
        request_data = request.get_json() or {}
        start_year = request_data.get('start_year', 2010)
        end_year = request_data.get('end_year', 2023)
        prediction_years = request_data.get('prediction_years', 5)
        model_type = request_data.get('model_type', 'random_forest')
        
        print(f"Analyzing {country} from {start_year} to {end_year}")
        
        # Get historical data
        historical_data = data_collector.get_country_data(country, start_year, end_year)
        print(f"Retrieved historical data with {len(historical_data)} indicators")
        
        # Engineer features
        features_data = feature_engineer.create_features(historical_data, country)
        print(f"Created features DataFrame with shape: {features_data.shape}")
        
        # Train model and make predictions
        model_results = ml_models.train_and_predict(features_data, prediction_years, model_type)
        print(f"Model training completed")
        
        response = {
            'historical_data': historical_data,
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
            'data': convert_numpy_types(response)
        })
        
    except Exception as e:
        print(f"Error in analyze_country: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_countries():
    """Compare multiple countries"""
    try:
        request_data = request.get_json() or {}
        countries = request_data.get('countries', [])
        start_year = request_data.get('start_year', 2010)
        end_year = request_data.get('end_year', 2023)
        
        if len(countries) < 2:
            return jsonify({
                'status': 'error',
                'message': 'At least 2 countries required for comparison'
            }), 400
        
        comparison_data = {}
        
        for country in countries[:5]:  # Limit to 5 countries
            try:
                data = data_collector.get_country_data(country, start_year, end_year)
                comparison_data[country] = data
            except Exception as e:
                comparison_data[country] = {'error': str(e)}
        
        return jsonify({
            'status': 'success',
            'data': convert_numpy_types(comparison_data)
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
        },
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("Starting GDP Analysis Backend...")
    print("Available at: http://localhost:5001")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)
