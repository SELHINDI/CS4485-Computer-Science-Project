# GDP Analysis & Predictor Tool - Full Stack Application

A comprehensive full-stack web application for GDP analysis and prediction, featuring a Python Flask backend with machine learning algorithms and a beautiful, modern frontend.

## 🌟 Features

### 🚀 **Backend (Python Flask)**
- **RESTful API** with comprehensive endpoints
- **SQLite Database** with sample GDP data
- **Machine Learning Models** (Linear, Polynomial, Exponential regression)
- **Statistical Analysis** with scikit-learn
- **CORS Support** for frontend integration

### 🎨 **Frontend (Modern Web)**
- **Responsive Design** with dark/light theme support
- **Interactive Charts** using Chart.js
- **Real-time Data** visualization
- **AI-Powered Predictions** with confidence levels
- **Country Comparison** tools
- **Beautiful Animations** and transitions

### 📊 **Data & Analysis**
- **5 Major Economies** (USA, China, Japan, Germany, India)
- **Historical Data** (2015-2024)
- **Quarterly Performance** tracking
- **Sector Analysis** and breakdowns
- **Growth Rate Analysis** with trends

## 🚀 Quick Start

### **Option 1: Python Environment (Recommended)**

```bash
# Navigate to project directory
cd "C:\Users\alpba\OneDrive\Masaüstü\senior project alp"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### **Option 2: Docker (Easy Deployment)**

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t gdp-analyzer .
docker run -p 5000:5000 gdp-analyzer
```

### **Option 3: Direct Python**

```bash
# Install dependencies
pip install Flask Flask-CORS pandas numpy scikit-learn

# Run the application
python app.py
```

## 🌐 **Access the Application**

Once running, open your browser and visit:
- **Local**: http://localhost:5000
- **Network**: http://your-ip:5000

## 📁 **Project Structure**

```
GDP-Analysis-Tool/
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── gdp_data.db          # SQLite database (auto-created)
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── styles.css        # Enhanced CSS styling
│   └── script.js         # Frontend JavaScript
└── README.md             # This documentation
```

## 🔧 **API Endpoints**

### **Countries**
- `GET /api/countries` - Get all countries
- `GET /api/countries/{code}` - Get specific country data

### **Predictions**
- `POST /api/predict` - Generate GDP predictions
  ```json
  {
    "country_code": "USA",
    "years": 5,
    "model": "linear"
  }
  ```

### **Comparison**
- `POST /api/compare` - Compare two countries
  ```json
  {
    "country1": "USA",
    "country2": "CHN"
  }
  ```

### **Analysis**
- `GET /api/analysis/{code}` - Get detailed country analysis

## 🤖 **Machine Learning Models**

### **Linear Regression**
- **Best for**: Stable, consistent growth trends
- **Algorithm**: `y = mx + b`
- **Use case**: Mature economies

### **Polynomial Regression**
- **Best for**: Non-linear growth patterns
- **Algorithm**: `y = ax² + bx + c`
- **Use case**: Accelerating/decelerating growth

### **Exponential Growth**
- **Best for**: Compound growth scenarios
- **Algorithm**: `y = a(1 + r)ˣ`
- **Use case**: Emerging economies

## 🎨 **Frontend Features**

### **Interactive Dashboard**
- Real-time GDP statistics
- Animated charts and graphs
- Responsive design for all devices
- Dark/light theme toggle

### **Data Visualization**
- Historical GDP trends
- Growth rate analysis
- Sector contribution charts
- Quarterly performance tables

### **AI Predictions**
- Multiple prediction models
- Confidence level indicators
- R² score accuracy metrics
- Interactive forecast charts

### **Country Comparison**
- Side-by-side analysis
- Growth rate comparisons
- Per capita GDP analysis
- Historical trend overlays

## 🗄️ **Database Schema**

### **Tables**
- `countries` - Country information and current stats
- `historical_data` - Yearly GDP and growth data
- `quarterly_data` - Quarterly performance metrics
- `sector_data` - Economic sector breakdowns

### **Sample Data**
- **5 Countries**: USA, China, Japan, Germany, India
- **10 Years**: 2015-2024 historical data
- **Quarterly**: Q1-Q4 2024 performance
- **Sectors**: Services, Manufacturing, Agriculture, etc.

## 🔧 **Configuration**

### **Environment Variables**
```bash
FLASK_APP=app.py
FLASK_ENV=production
```

### **Database**
- SQLite database auto-created on first run
- Sample data automatically populated
- Persistent storage for data updates

## 📊 **Usage Examples**

### **Generate Prediction**
1. Navigate to "Prediction" section
2. Select country and forecast period
3. Choose AI model (Linear/Polynomial/Exponential)
4. Click "Generate Prediction"
5. View forecast with confidence levels

### **Compare Countries**
1. Go to "Comparison" section
2. Select two countries from dropdowns
3. Click "Compare Countries"
4. View side-by-side analysis
5. Review growth trends and metrics

### **Analyze Data**
1. Visit "Analysis" section
2. View growth rate charts
3. Examine sector contributions
4. Review quarterly performance
5. Get statistical insights

## 🚀 **Deployment Options**

### **Local Development**
```bash
python app.py
```

### **Production Server**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Docker Deployment**
```bash
docker-compose up -d
```

### **Cloud Deployment**
- **Heroku**: Add Procfile and deploy
- **AWS**: Use Elastic Beanstalk or ECS
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Use App Service

## 🔒 **Security Features**

- **CORS Protection** for API security
- **Input Validation** on all endpoints
- **Error Handling** with proper HTTP status codes
- **SQL Injection Protection** with parameterized queries

## 📈 **Performance**

- **Fast API Responses** with optimized queries
- **Efficient Data Processing** with pandas/numpy
- **Caching** for frequently accessed data
- **Optimized Charts** with Chart.js

## 🛠️ **Development**

### **Adding New Countries**
Edit the database or add new entries via API:

```python
# Add new country data
cursor.execute('''
    INSERT INTO countries (code, name, current_gdp, growth_rate, per_capita, world_rank)
    VALUES (?, ?, ?, ?, ?, ?)
''', (code, name, gdp, growth, per_capita, rank))
```

### **Customizing Models**
Extend the `GDPPredictor` class:

```python
def custom_model(self, X, y, future_years):
    # Your custom prediction algorithm
    return predictions, r2_score
```

### **Adding New Endpoints**
```python
@app.route('/api/custom')
def custom_endpoint():
    # Your custom logic
    return jsonify(data)
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Port Already in Use**
   ```bash
   # Kill process on port 5000
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```

2. **Database Issues**
   ```bash
   # Delete and recreate database
   rm gdp_data.db
   python app.py
   ```

3. **Dependencies Issues**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **CORS Issues**
   - Ensure Flask-CORS is installed
   - Check browser console for errors

## 📝 **API Documentation**

### **Response Formats**

**Country Data:**
```json
{
  "code": "USA",
  "name": "United States",
  "current_gdp": 25400000000000,
  "growth_rate": 3.2,
  "per_capita": 76400,
  "world_rank": 1,
  "historical_data": [...],
  "quarterly_data": [...],
  "sector_data": [...]
}
```

**Prediction Response:**
```json
{
  "predictions": [...],
  "historical_data": [...],
  "model_type": "linear",
  "r2_score": 0.95,
  "confidence_level": 87,
  "years": 5
}
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 **License**

This project is open source and available under the MIT License.

## 🆘 **Support**

For issues or questions:

1. Check the browser console for errors
2. Verify all dependencies are installed
3. Ensure the Flask server is running
4. Check database connectivity

## 🔄 **Updates**

### **Version 2.0.0**
- Full-stack Python Flask backend
- Machine learning prediction models
- Enhanced frontend with animations
- Docker support for easy deployment
- Comprehensive API documentation

---

**Built with ❤️ using Python Flask, Chart.js, and modern web technologies**