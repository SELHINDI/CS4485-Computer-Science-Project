# GDP Analysis and Prediction Backend

A comprehensive backend system for analyzing and predicting GDP trends using machine learning techniques. This system supports real-time data collection, feature engineering, and multiple prediction models with validation capabilities.

## 🚀 Features

- **Real-time Data Collection**: Automated data gathering from World Bank and other economic APIs
- **Advanced Feature Engineering**: Creates 50+ economic indicators and features
- **Multiple ML Models**: Support for Linear Regression, Random Forest, Gradient Boosting, and Ensemble methods
- **Time Series Analysis**: Specialized features for economic time series prediction
- **Model Validation**: Historical backtesting and performance metrics
- **RESTful API**: Clean API endpoints for frontend integration
- **Caching System**: Intelligent caching for improved performance
- **Docker Support**: Full containerization for easy deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Features](#features-detailed)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-team/gdp-analysis-backend.git
   cd gdp-analysis-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - API: `http://localhost:5000`
   - Health check: `http://localhost:5000/api/health`

## 🚀 Quick Start

### Basic Usage

1. **Check available countries**
   ```bash
   curl http://localhost:5000/api/countries
   ```

2. **Get country data**
   ```bash
   curl "http://localhost:5000/api/data/US?start_year=2010&end_year=2023"
   ```

3. **Analyze and predict**
   ```bash
   curl -X POST http://localhost:5000/api/analyze/US \
     -H "Content-Type: application/json" \
     -d '{
       "start_year": 2010,
       "end_year": 2023,
       "prediction_years": 5,
       "model_type": "ensemble"
     }'
   ```

### Python Client Example

```python
import requests
import json

# Initialize API client
BASE_URL = "http://localhost:5000/api"

# Analyze a country
def analyze_country(country, start_year=2010, end_year=2023, prediction_years=5):
    response = requests.post(f"{BASE_URL}/analyze/{country}", json={
        "start_year": start_year,
        "end_year": end_year,
        "prediction_years": prediction_years,
        "model_type": "ensemble"
    })
    
    if response.status_code == 200:
        return response.json()['data']
    else:
        print(f"Error: {response.json()}")
        return None

# Example usage
result = analyze_country('US', prediction_years=3)
if result:
    print("Predictions:")
    for pred in result['predictions']:
        print(f"Year {pred['year']}: ${pred['predicted_gdp']:,.0f} (confidence: {pred['confidence']:.2f})")
```

## 📚 API Documentation

### Core Endpoints

#### GET /api/health
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "cache_size": {"data": 5, "models": 3}
}
```

#### GET /api/countries
Get list of available countries
```json
{
  "status": "success",
  "data": [
    {
      "id": "US",
      "name": "United States",
      "region": "North America",
      "income_level": "High income"
    }
  ]
}
```

#### GET /api/data/{country}
Get historical data for a country

**Parameters:**
- `start_year` (optional): Start year (default: 1990)
- `end_year` (optional): End year (default: 2023)

**Response:**
```json
{
  "status": "success",
  "data": {
    "gdp": [...],
    "gdp_growth": [...],
    "exports": [...],
    "major_events": [...]
  }
}
```

#### POST /api/analyze/{country}
Perform comprehensive analysis and prediction

**Request Body:**
```json
{
  "start_year": 2010,
  "end_year": 2023,
  "prediction_years": 5,
  "model_type": "ensemble"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "historical_data": {...},
    "predictions": [
      {
        "year": 2024,
        "predicted_gdp": 25000000000000,
        "confidence": 0.85
      }
    ],
    "model_performance": {
      "mae": 50000000000,
      "mse": 75000000000000,
      "r2": 0.95,
      "mape": 2.1
    },
    "feature_importance": {
      "gdp_lag_1": 0.35,
      "exports": 0.18,
      "gdp_growth": 0.12
    }
  }
}
```

#### POST /api/compare
Compare multiple countries

**Request Body:**
```json
{
  "countries": ["US", "CN", "DE"],
  "start_year": 2010,
  "end_year": 2023
}
```

#### POST /api/validate/{country}
Validate model against historical data

**Request Body:**
```json
{
  "train_end_year": 2018,
  "test_end_year": 2023,
  "model_type": "ensemble"
}
```

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│   Flask API     │────│  Data Collector │
│   (Your App)    │    │   (app.py)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Feature Engineer│    │  External APIs  │
                       │                 │    │ (World Bank,    │
                       └─────────────────┘    │  IMF, etc.)     │
                                │             └─────────────────┘
                                ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │ (Random Forest, │
                       │  Ensemble, etc.)│
                       └─────────────────┘
```

### Data Flow

1. **Data Collection**: Fetch economic indicators from multiple sources
2. **Feature Engineering**: Create 50+ engineered features
3. **Model Training**: Train multiple ML models with cross-validation
4. **Prediction**: Generate future GDP predictions with confidence intervals
5. **Validation**: Backtest models against historical data

### Key Modules

- **`app.py`**: Main Flask application with API endpoints
- **`data_collector.py`**: Handles data fetching from external APIs
- **`feature_engineer.py`**: Creates and transforms features for ML models
- **`ml_models.py`**: Implements various machine learning models
- **`config.py`**: Configuration management

## 🔬 Features (Detailed)

### Data Sources

- **World Bank API**: GDP, trade, demographic data
- **IMF Data**: International financial statistics
- **Custom Events**: Major economic events and crises
- **Commodity Prices**: Oil prices and other commodities

### Feature Engineering

#### Time-Based Features
- Linear and quadratic time trends
- Business cycle indicators
- Seasonal patterns
- Decade classification

#### Economic Indicators
- GDP growth rates and momentum
- Trade ratios and balances
- Investment indicators
- Government spending ratios

#### Lag Features
- 1-3 year lags for key indicators
- Rolling averages (3, 5 year windows)
- Volatility measures

#### External Factors
- Crisis impact indicators
- Oil price trends
- Global recession flags
- Country-specific events

### Machine Learning Models

#### Supported Models
1. **Linear Regression**: Baseline model
2. **Ridge Regression**: Regularized linear model
3. **Random Forest**: Ensemble tree-based model
4. **Gradient Boosting**: Advanced ensemble method
5. **Support Vector Regression**: Non-linear kernel methods
6. **Ensemble**: Weighted combination of best models

#### Model Selection
- Automatic hyperparameter tuning
- Cross-validation with time series splits
- Performance-based model weighting
- Feature importance analysis

### Performance Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python test_app.py

# Run with coverage
make test

# Run specific test class
python -m unittest test_app.TestDataCollector

# Quick test without coverage
make test-quick
```

### Test Structure

```
tests/
├── test_app.py              # API endpoint tests
├── test_data_collector.py   # Data collection tests
├── test_feature_engineer.py # Feature engineering tests
├── test_ml_models.py        # ML model tests
└── test_integration.py      # End-to-end tests
```

### Continuous Integration

GitHub Actions workflow automatically:
- Runs tests on multiple Python versions
- Checks code formatting and linting
- Generates coverage reports
- Builds Docker images

## 🚢 Deployment

### Local Development

```bash
# Development server
python app.py

# Production server
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Docker Deployment

```bash
# Build and deploy
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale gdp-backend=3
```

### Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI, then:
heroku create your-gdp-api
git push heroku main
```

#### Railway
```bash
# Install Railway CLI, then:
railway login
railway init
railway up
```

#### AWS/GCP/Azure
Use the provided Dockerfile with your preferred container service.

### Environment Variables

```bash
# Required
SECRET_KEY=your-secret-key
FLASK_ENV=production

# Optional API Keys
WORLD_BANK_API_KEY=your-key
ALPHA_VANTAGE_API_KEY=your-key
FRED_API_KEY=your-key

# Database (if using)
DATABASE_URL=your-database-url
```

## 📊 Performance Optimization

### Caching Strategy
- **Data Cache**: 24-hour TTL for API data
- **Model Cache**: 6-hour TTL for trained models
- **Feature Cache**: 1-hour TTL for engineered features

### Scaling Considerations
- Horizontal scaling with multiple workers
- Database connection pooling
- Redis for distributed caching
- Load balancing with nginx

## 🔧 Configuration

### Model Configuration

```python
# config.py
class Config:
    # Model parameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 3
    
    # Feature engineering
    MAX_LAG_FEATURES = 3
    ROLLING_WINDOWS = [3, 5]
    
    # Prediction limits
    MAX_PREDICTION_YEARS = 10
    MIN_DATA_POINTS = 10
```

### API Configuration

```python
# Rate limiting and timeouts
REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.1
MAX_RETRIES = 3

# Cache settings
CACHE_TIMEOUT = timedelta(hours=24)
MODEL_CACHE_TIMEOUT = timedelta(hours=6)
```

## 📈 Usage Examples

### Business Use Cases

#### Economic Research
```python
# Compare GDP trends across countries
countries = ['US', 'CN', 'DE', 'JP']
comparison = requests.post(f'{BASE_URL}/compare', json={
    'countries': countries,
    'start_year': 2000,
    'end_year': 2023
})
```

#### Investment Analysis
```python
# Predict economic growth for investment decisions
prediction = requests.post(f'{BASE_URL}/analyze/IN', json={
    'start_year': 2010,
    'end_year': 2023,
    'prediction_years': 5,
    'model_type': 'ensemble'
})
```

#### Policy Impact Assessment
```python
# Validate model performance during crisis periods
validation = requests.post(f'{BASE_URL}/validate/US', json={
    'train_end_year': 2007,  # Before 2008 crisis
    'test_end_year': 2012,   # Including crisis period
    'model_type': 'random_forest'
})
```

## 🐛 Troubleshooting

### Common Issues

#### API Connection Errors
```bash
# Check if World Bank API is accessible
curl "https://api.worldbank.org/v2/country?format=json"

# Test local health endpoint
curl http://localhost:5000/api/health
```

#### Memory Issues
```bash
# Increase Docker memory limits
docker-compose up -d --scale gdp-backend=2

# Clear cache
curl -X POST http://localhost:5000/api/clear-cache
```

#### Model Training Failures
- Ensure minimum 10 data points
- Check for missing data in time series
- Verify feature engineering pipeline
- Review logs for specific errors

### Logging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Workflow

1. **Fork and clone** the repository
2. **Create feature branch**: `git checkout -b feature/new-indicator`
3. **Make changes** with tests
4. **Run tests**: `make test`
5. **Format code**: `make format`
6. **Submit pull request**

### Code Standards

- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** for all classes and methods
- **Unit tests** for new features
- **Integration tests** for API endpoints

### Adding New Features

#### New Economic Indicators
1. Add to `data_collector.py` indicators dictionary
2. Update feature engineering in `feature_engineer.py`
3. Add tests for new indicators
4. Update documentation

#### New ML Models
1. Add model to `ml_models.py`
2. Implement feature importance extraction
3. Add model-specific preprocessing
4. Update ensemble weighting logic

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **World Bank** for providing comprehensive economic data APIs
- **IMF** for international financial statistics
- **Scikit-learn** community for excellent ML tools
- **Flask** community for web framework excellence

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-team/gdp-analysis-backend/issues)
- **Documentation**: [Wiki](https://github.com/your-team/gdp-analysis-backend/wiki)
- **Email**: team@yourproject.com

---

## 🔄 Changelog

### v1.0.0 (Current)
- Initial release
- Complete API implementation
- Multiple ML models
- Docker support
- Comprehensive testing

### Planned Features
- Real-time streaming data
- Additional economic indicators
- Advanced ensemble methods
- Database persistence
- Web dashboard integration

---

*Built with ❤️ for economic analysis and prediction*

hello