from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import json
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    # Create countries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS countries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            current_gdp REAL,
            growth_rate REAL,
            per_capita REAL,
            world_rank INTEGER
        )
    ''')
    
    # Create historical_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code TEXT,
            year INTEGER,
            gdp REAL,
            growth_rate REAL,
            FOREIGN KEY (country_code) REFERENCES countries (code)
        )
    ''')
    
    # Create quarterly_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quarterly_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code TEXT,
            quarter TEXT,
            gdp REAL,
            growth_rate REAL,
            inflation REAL,
            unemployment REAL,
            FOREIGN KEY (country_code) REFERENCES countries (code)
        )
    ''')
    
    # Create sector_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sector_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code TEXT,
            sector TEXT,
            percentage REAL,
            value REAL,
            FOREIGN KEY (country_code) REFERENCES countries (code)
        )
    ''')
    
    conn.commit()
    conn.close()

def populate_sample_data():
    """Populate database with sample GDP data"""
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) FROM countries')
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # Sample data
    countries_data = [
        ('USA', 'United States', 25400000000000, 3.2, 76400, 1),
        ('CHN', 'China', 17700000000000, 5.1, 12500, 2),
        ('JPN', 'Japan', 4200000000000, 1.0, 33400, 3),
        ('DEU', 'Germany', 4200000000000, 0.2, 50400, 4),
        ('IND', 'India', 3700000000000, 6.9, 2600, 5)
    ]
    
    # Insert countries
    cursor.executemany('''
        INSERT INTO countries (code, name, current_gdp, growth_rate, per_capita, world_rank)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', countries_data)
    
    # Historical data for all countries
    historical_data = [
        # USA
        ('USA', 2015, 18.2, 2.9),
        ('USA', 2016, 18.7, 1.7),
        ('USA', 2017, 19.5, 2.3),
        ('USA', 2018, 20.5, 2.9),
        ('USA', 2019, 21.4, 2.2),
        ('USA', 2020, 20.9, -2.2),
        ('USA', 2021, 23.3, 5.9),
        ('USA', 2022, 25.5, 2.1),
        ('USA', 2023, 25.0, 2.5),
        ('USA', 2024, 25.4, 3.2),
        
        # China
        ('CHN', 2015, 11.1, 6.9),
        ('CHN', 2016, 11.2, 6.7),
        ('CHN', 2017, 12.3, 6.8),
        ('CHN', 2018, 13.9, 6.7),
        ('CHN', 2019, 14.3, 6.0),
        ('CHN', 2020, 14.7, 2.2),
        ('CHN', 2021, 17.7, 8.1),
        ('CHN', 2022, 17.9, 3.0),
        ('CHN', 2023, 17.4, 5.2),
        ('CHN', 2024, 17.7, 5.1),
        
        # Japan
        ('JPN', 2015, 4.4, 0.4),
        ('JPN', 2016, 4.9, 0.5),
        ('JPN', 2017, 4.9, 1.7),
        ('JPN', 2018, 5.0, 0.3),
        ('JPN', 2019, 5.1, 0.7),
        ('JPN', 2020, 4.9, -4.3),
        ('JPN', 2021, 4.9, 1.6),
        ('JPN', 2022, 4.2, 1.0),
        ('JPN', 2023, 4.2, 1.9),
        ('JPN', 2024, 4.2, 1.0),
        
        # Germany
        ('DEU', 2015, 3.4, 1.5),
        ('DEU', 2016, 3.5, 2.2),
        ('DEU', 2017, 3.7, 2.2),
        ('DEU', 2018, 3.9, 1.1),
        ('DEU', 2019, 3.9, 0.6),
        ('DEU', 2020, 3.8, -4.6),
        ('DEU', 2021, 4.3, 2.6),
        ('DEU', 2022, 4.3, 1.8),
        ('DEU', 2023, 4.2, -0.1),
        ('DEU', 2024, 4.2, 0.2),
        
        # India
        ('IND', 2015, 2.1, 7.4),
        ('IND', 2016, 2.3, 8.0),
        ('IND', 2017, 2.7, 6.8),
        ('IND', 2018, 2.7, 6.5),
        ('IND', 2019, 2.9, 4.0),
        ('IND', 2020, 2.7, -6.6),
        ('IND', 2021, 3.2, 8.7),
        ('IND', 2022, 3.4, 7.0),
        ('IND', 2023, 3.5, 7.2),
        ('IND', 2024, 3.7, 6.9)
    ]
    
    cursor.executemany('''
        INSERT INTO historical_data (country_code, year, gdp, growth_rate)
        VALUES (?, ?, ?, ?)
    ''', historical_data)
    
    # Quarterly data for all countries
    quarterly_data = [
        # USA
        ('USA', 'Q1 2024', 25.1, 2.8, 3.1, 3.8),
        ('USA', 'Q2 2024', 25.3, 3.2, 2.9, 3.7),
        ('USA', 'Q3 2024', 25.4, 3.5, 2.7, 3.6),
        ('USA', 'Q4 2024', 25.4, 3.2, 2.8, 3.5),
        
        # China
        ('CHN', 'Q1 2024', 17.2, 5.3, 2.1, 5.2),
        ('CHN', 'Q2 2024', 17.4, 5.1, 1.9, 5.1),
        ('CHN', 'Q3 2024', 17.6, 5.0, 1.8, 5.0),
        ('CHN', 'Q4 2024', 17.7, 5.1, 1.9, 4.9),
        
        # Japan
        ('JPN', 'Q1 2024', 4.1, 1.2, 2.8, 2.6),
        ('JPN', 'Q2 2024', 4.2, 1.0, 2.6, 2.5),
        ('JPN', 'Q3 2024', 4.2, 0.8, 2.4, 2.4),
        ('JPN', 'Q4 2024', 4.2, 1.0, 2.5, 2.3),
        
        # Germany
        ('DEU', 'Q1 2024', 4.1, 0.1, 3.2, 3.1),
        ('DEU', 'Q2 2024', 4.2, 0.2, 3.0, 3.0),
        ('DEU', 'Q3 2024', 4.2, 0.3, 2.8, 2.9),
        ('DEU', 'Q4 2024', 4.2, 0.2, 2.9, 2.8),
        
        # India
        ('IND', 'Q1 2024', 3.6, 7.1, 4.8, 3.2),
        ('IND', 'Q2 2024', 3.7, 6.9, 4.6, 3.1),
        ('IND', 'Q3 2024', 3.7, 6.8, 4.4, 3.0),
        ('IND', 'Q4 2024', 3.7, 6.9, 4.5, 2.9)
    ]
    
    cursor.executemany('''
        INSERT INTO quarterly_data (country_code, quarter, gdp, growth_rate, inflation, unemployment)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', quarterly_data)
    
    # Sector data for all countries
    sector_data = [
        # USA
        ('USA', 'Services', 78.2, 19.8),
        ('USA', 'Manufacturing', 11.1, 2.8),
        ('USA', 'Agriculture', 0.9, 0.23),
        ('USA', 'Construction', 4.1, 1.04),
        ('USA', 'Other', 5.7, 1.45),
        
        # China
        ('CHN', 'Services', 54.5, 9.6),
        ('CHN', 'Manufacturing', 28.8, 5.1),
        ('CHN', 'Agriculture', 7.1, 1.3),
        ('CHN', 'Construction', 6.8, 1.2),
        ('CHN', 'Other', 2.8, 0.5),
        
        # Japan
        ('JPN', 'Services', 71.4, 3.0),
        ('JPN', 'Manufacturing', 20.1, 0.8),
        ('JPN', 'Agriculture', 1.2, 0.05),
        ('JPN', 'Construction', 5.1, 0.21),
        ('JPN', 'Other', 2.2, 0.09),
        
        # Germany
        ('DEU', 'Services', 68.5, 2.9),
        ('DEU', 'Manufacturing', 22.1, 0.93),
        ('DEU', 'Agriculture', 0.7, 0.03),
        ('DEU', 'Construction', 5.8, 0.24),
        ('DEU', 'Other', 2.9, 0.12),
        
        # India
        ('IND', 'Services', 54.3, 2.0),
        ('IND', 'Manufacturing', 16.3, 0.6),
        ('IND', 'Agriculture', 17.8, 0.66),
        ('IND', 'Construction', 8.2, 0.3),
        ('IND', 'Other', 3.4, 0.13)
    ]
    
    cursor.executemany('''
        INSERT INTO sector_data (country_code, sector, percentage, value)
        VALUES (?, ?, ?, ?)
    ''', sector_data)
    
    conn.commit()
    conn.close()

class GDPPredictor:
    """GDP prediction using machine learning models"""
    
    def __init__(self):
        self.models = {}
    
    def prepare_data(self, historical_data):
        """Prepare data for machine learning"""
        df = pd.DataFrame(historical_data)
        X = df[['year']].values
        y = df['gdp'].values
        return X, y
    
    def linear_regression(self, X, y, future_years):
        """Linear regression prediction"""
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future years
        future_X = np.array([[year] for year in future_years])
        predictions = model.predict(future_X)
        
        # Calculate R² score
        r2 = r2_score(y, model.predict(X))
        
        return predictions, r2
    
    def polynomial_regression(self, X, y, future_years, degree=2):
        """Polynomial regression prediction"""
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict future years
        future_X = np.array([[year] for year in future_years])
        future_X_poly = poly_features.transform(future_X)
        predictions = model.predict(future_X_poly)
        
        # Calculate R² score
        r2 = r2_score(y, model.predict(X_poly))
        
        return predictions, r2
    
    def exponential_growth(self, historical_data, future_years):
        """Exponential growth model"""
        df = pd.DataFrame(historical_data)
        df = df.sort_values('year')
        
        # Calculate average growth rate
        growth_rates = df['gdp'].pct_change().dropna()
        avg_growth_rate = growth_rates.mean()
        
        # Predict using exponential growth
        last_gdp = df['gdp'].iloc[-1]
        predictions = []
        
        for i, year in enumerate(future_years):
            years_ahead = i + 1
            predicted_gdp = last_gdp * (1 + avg_growth_rate) ** years_ahead
            predictions.append(predicted_gdp)
        
        # Calculate R² score (simplified)
        r2 = 0.8  # Placeholder for exponential model
        
        return np.array(predictions), r2

# Initialize database and sample data
init_db()
populate_sample_data()

# Initialize predictor
predictor = GDPPredictor()

@app.route('/')
def index():
    """Serve the main application"""
    return render_template('index.html')

@app.route('/api/countries')
def get_countries():
    """Get all countries"""
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT code, name, current_gdp, growth_rate, per_capita, world_rank
        FROM countries
        ORDER BY world_rank
    ''')
    
    countries = []
    for row in cursor.fetchall():
        countries.append({
            'code': row[0],
            'name': row[1],
            'current_gdp': row[2],
            'growth_rate': row[3],
            'per_capita': row[4],
            'world_rank': row[5]
        })
    
    conn.close()
    return jsonify(countries)

@app.route('/api/countries/<country_code>')
def get_country(country_code):
    """Get specific country data"""
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    # Get country info
    cursor.execute('''
        SELECT code, name, current_gdp, growth_rate, per_capita, world_rank
        FROM countries WHERE code = ?
    ''', (country_code,))
    
    country_row = cursor.fetchone()
    if not country_row:
        conn.close()
        return jsonify({'error': 'Country not found'}), 404
    
    country = {
        'code': country_row[0],
        'name': country_row[1],
        'current_gdp': country_row[2],
        'growth_rate': country_row[3],
        'per_capita': country_row[4],
        'world_rank': country_row[5]
    }
    
    # Get historical data
    cursor.execute('''
        SELECT year, gdp, growth_rate
        FROM historical_data
        WHERE country_code = ?
        ORDER BY year
    ''', (country_code,))
    
    historical_data = []
    for row in cursor.fetchall():
        historical_data.append({
            'year': row[0],
            'gdp': row[1],
            'growth_rate': row[2]
        })
    
    # Get quarterly data
    cursor.execute('''
        SELECT quarter, gdp, growth_rate, inflation, unemployment
        FROM quarterly_data
        WHERE country_code = ?
        ORDER BY quarter
    ''', (country_code,))
    
    quarterly_data = []
    for row in cursor.fetchall():
        quarterly_data.append({
            'quarter': row[0],
            'gdp': row[1],
            'growth_rate': row[2],
            'inflation': row[3],
            'unemployment': row[4]
        })
    
    # Get sector data
    cursor.execute('''
        SELECT sector, percentage, value
        FROM sector_data
        WHERE country_code = ?
        ORDER BY percentage DESC
    ''', (country_code,))
    
    sector_data = []
    for row in cursor.fetchall():
        sector_data.append({
            'sector': row[0],
            'percentage': row[1],
            'value': row[2]
        })
    
    country['historical_data'] = historical_data
    country['quarterly_data'] = quarterly_data
    country['sector_data'] = sector_data
    
    conn.close()
    return jsonify(country)

@app.route('/api/predict', methods=['POST'])
def predict_gdp():
    """Generate GDP predictions"""
    data = request.get_json()
    country_code = data.get('country_code', 'USA')
    years = data.get('years', 5)
    model_type = data.get('model', 'linear')
    
    # Get historical data
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT year, gdp, growth_rate
        FROM historical_data
        WHERE country_code = ?
        ORDER BY year
    ''', (country_code,))
    
    historical_data = []
    for row in cursor.fetchall():
        historical_data.append({
            'year': row[0],
            'gdp': row[1],
            'growth_rate': row[2]
        })
    
    conn.close()
    
    if len(historical_data) < 3:
        return jsonify({'error': 'Insufficient historical data'}), 400
    
    # Prepare data for prediction
    X, y = predictor.prepare_data(historical_data)
    
    # Generate future years
    last_year = max([d['year'] for d in historical_data])
    future_years = list(range(last_year + 1, last_year + years + 1))
    
    # Generate predictions based on model type
    if model_type == 'linear':
        predictions, r2_score = predictor.linear_regression(X, y, future_years)
    elif model_type == 'polynomial':
        predictions, r2_score = predictor.polynomial_regression(X, y, future_years)
    elif model_type == 'exponential':
        predictions, r2_score = predictor.exponential_growth(historical_data, future_years)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    # Format predictions
    prediction_data = []
    for i, year in enumerate(future_years):
        prediction_data.append({
            'year': year,
            'gdp': float(predictions[i]),
            'type': 'prediction'
        })
    
    # Calculate confidence level based on R² score
    confidence_level = min(95, max(60, r2_score * 100))
    
    return jsonify({
        'predictions': prediction_data,
        'historical_data': historical_data,
        'model_type': model_type,
        'r2_score': float(r2_score),
        'confidence_level': confidence_level,
        'years': years
    })

@app.route('/api/compare', methods=['POST'])
def compare_countries():
    """Compare two countries"""
    data = request.get_json()
    country1_code = data.get('country1')
    country2_code = data.get('country2')
    
    if not country1_code or not country2_code:
        return jsonify({'error': 'Both countries must be specified'}), 400
    
    # Get both countries' data
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    countries_data = {}
    for code in [country1_code, country2_code]:
        cursor.execute('''
            SELECT code, name, current_gdp, growth_rate, per_capita, world_rank
            FROM countries WHERE code = ?
        ''', (code,))
        
        row = cursor.fetchone()
        if row:
            countries_data[code] = {
                'code': row[0],
                'name': row[1],
                'current_gdp': row[2],
                'growth_rate': row[3],
                'per_capita': row[4],
                'world_rank': row[5]
            }
            
            # Get historical data for comparison chart
            cursor.execute('''
                SELECT year, gdp
                FROM historical_data
                WHERE country_code = ?
                ORDER BY year
            ''', (code,))
            
            historical = []
            for hist_row in cursor.fetchall():
                historical.append({
                    'year': hist_row[0],
                    'gdp': hist_row[1]
                })
            
            countries_data[code]['historical_data'] = historical
    
    conn.close()
    
    if len(countries_data) != 2:
        return jsonify({'error': 'One or both countries not found'}), 404
    
    return jsonify({
        'country1': countries_data[country1_code],
        'country2': countries_data[country2_code]
    })

@app.route('/api/analysis/<country_code>')
def get_analysis(country_code):
    """Get detailed analysis for a country"""
    conn = sqlite3.connect('gdp_data.db')
    cursor = conn.cursor()
    
    # Get country data
    cursor.execute('''
        SELECT code, name, current_gdp, growth_rate, per_capita, world_rank
        FROM countries WHERE code = ?
    ''', (country_code,))
    
    country_row = cursor.fetchone()
    if not country_row:
        conn.close()
        return jsonify({'error': 'Country not found'}), 404
    
    # Get historical data for analysis
    cursor.execute('''
        SELECT year, gdp, growth_rate
        FROM historical_data
        WHERE country_code = ?
        ORDER BY year
    ''', (country_code,))
    
    historical_data = []
    for row in cursor.fetchall():
        historical_data.append({
            'year': row[0],
            'gdp': row[1],
            'growth_rate': row[2]
        })
    
    # Calculate analysis metrics
    gdp_values = [d['gdp'] for d in historical_data]
    growth_rates = [d['growth_rate'] for d in historical_data]
    
    analysis = {
        'country': {
            'code': country_row[0],
            'name': country_row[1],
            'current_gdp': country_row[2],
            'growth_rate': country_row[3],
            'per_capita': country_row[4],
            'world_rank': country_row[5]
        },
        'historical_data': historical_data,
        'metrics': {
            'average_growth_rate': np.mean(growth_rates),
            'gdp_volatility': np.std(gdp_values),
            'growth_trend': 'positive' if np.mean(growth_rates[-3:]) > np.mean(growth_rates[:3]) else 'negative',
            'peak_gdp': max(gdp_values),
            'peak_year': historical_data[gdp_values.index(max(gdp_values))]['year']
        }
    }
    
    conn.close()
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)