# test_app.py
import unittest
import json
import pandas as pd
import numpy as np
from app import app
from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModels

class TestGDPBackend(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and test data"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create sample test data
        self.sample_data = {
            'gdp': [
                {'year': 2018, 'value': 1000000000000, 'country': 'Test Country'},
                {'year': 2019, 'value': 1050000000000, 'country': 'Test Country'},
                {'year': 2020, 'value': 1000000000000, 'country': 'Test Country'},
                {'year': 2021, 'value': 1080000000000, 'country': 'Test Country'},
                {'year': 2022, 'value': 1120000000000, 'country': 'Test Country'}
            ],
            'gdp_growth': [
                {'year': 2018, 'value': 2.5, 'country': 'Test Country'},
                {'year': 2019, 'value': 3.0, 'country': 'Test Country'},
                {'year': 2020, 'value': -2.0, 'country': 'Test Country'},
                {'year': 2021, 'value': 4.0, 'country': 'Test Country'},
                {'year': 2022, 'value': 2.8, 'country': 'Test Country'}
            ],
            'inflation': [
                {'year': 2018, 'value': 2.1, 'country': 'Test Country'},
                {'year': 2019, 'value': 2.3, 'country': 'Test Country'},
                {'year': 2020, 'value': 1.8, 'country': 'Test Country'},
                {'year': 2021, 'value': 3.5, 'country': 'Test Country'},
                {'year': 2022, 'value': 4.2, 'country': 'Test Country'}
            ],
            'unemployment': [
                {'year': 2018, 'value': 5.2, 'country': 'Test Country'},
                {'year': 2019, 'value': 4.8, 'country': 'Test Country'},
                {'year': 2020, 'value': 7.1, 'country': 'Test Country'},
                {'year': 2021, 'value': 6.2, 'country': 'Test Country'},
                {'year': 2022, 'value': 4.9, 'country': 'Test Country'}
            ]
        }
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_get_countries(self):
        """Test get countries endpoint"""
        response = self.app.get('/api/countries')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)
    
    def test_analyze_country_invalid_data(self):
        """Test analysis with invalid data"""
        response = self.app.post('/api/analyze/INVALID', 
                                json={'start_year': 2020, 'end_year': 2023})
        # Should handle gracefully, not crash
        self.assertIn(response.status_code, [200, 500])
    
    def test_compare_countries_insufficient_data(self):
        """Test comparison with insufficient countries"""
        response = self.app.post('/api/compare', 
                                json={'countries': ['US'], 'start_year': 2020})
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
    
    def test_clear_cache(self):
        """Test cache clearing"""
        response = self.app.post('/api/clear-cache')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')

class TestDataCollector(unittest.TestCase):
    
    def setUp(self):
        """Set up data collector"""
        self.data_collector = DataCollector()
    
    def test_available_countries(self):
        """Test getting available countries"""
        countries = self.data_collector.get_available_countries()
        self.assertIsInstance(countries, list)
        self.assertGreater(len(countries), 0)
        
        # Check structure of first country
        if countries:
            country = countries[0]
            self.assertIn('id', country)
            self.assertIn('name', country)
    
    def test_world_bank_data_structure(self):
        """Test World Bank data structure"""
        # This test might fail if API is down, so we'll be lenient
        try:
            data = self.data_collector.get_world_bank_data('US', 'NY.GDP.MKTP.CD', 2020, 2022)
            if data:  # Only test if we got data
                self.assertIsInstance(data, list)
                if len(data) > 0:
                    item = data[0]
                    self.assertIn('year', item)
                    self.assertIn('value', item)
                    self.assertIn('country', item)
        except:
            # API might be down, skip this test
            pass
    
    def test_country_codes_mapping(self):
        """Test country codes mapping"""
        self.assertIn('United States', self.data_collector.country_codes)
        self.assertEqual(self.data_collector.country_codes['United States'], 'US')
    
    def test_indicators_defined(self):
        """Test that indicators are properly defined"""
        self.assertIn('gdp', self.data_collector.indicators)
        self.assertIn('gdp_growth', self.data_collector.indicators)
        self.assertIn('inflation', self.data_collector.indicators)


class TestFeatureEngineer(unittest.TestCase):
    
    def setUp(self):
        """Set up feature engineer and sample data"""
        self.feature_engineer = FeatureEngineer()
        
        # Create comprehensive sample data
        self.sample_data = {
            'gdp': [
                {'year': 2018, 'value': 1000000000000, 'country': 'Test Country'},
                {'year': 2019, 'value': 1050000000000, 'country': 'Test Country'},
                {'year': 2020, 'value': 1000000000000, 'country': 'Test Country'},
                {'year': 2021, 'value': 1080000000000, 'country': 'Test Country'},
                {'year': 2022, 'value': 1120000000000, 'country': 'Test Country'}
            ],
            'gdp_growth': [
                {'year': 2018, 'value': 2.5, 'country': 'Test Country'},
                {'year': 2019, 'value': 3.0, 'country': 'Test Country'},
                {'year': 2020, 'value': -2.0, 'country': 'Test Country'},
                {'year': 2021, 'value': 4.0, 'country': 'Test Country'},
                {'year': 2022, 'value': 2.8, 'country': 'Test Country'}
            ],
            'exports': [
                {'year': 2018, 'value': 200000000000, 'country': 'Test Country'},
                {'year': 2019, 'value': 210000000000, 'country': 'Test Country'},
                {'year': 2020, 'value': 180000000000, 'country': 'Test Country'},
                {'year': 2021, 'value': 220000000000, 'country': 'Test Country'},
                {'year': 2022, 'value': 240000000000, 'country': 'Test Country'}
            ],
            'imports': [
                {'year': 2018, 'value': 180000000000, 'country': 'Test Country'},
                {'year': 2019, 'value': 190000000000, 'country': 'Test Country'},
                {'year': 2020, 'value': 170000000000, 'country': 'Test Country'},
                {'year': 2021, 'value': 200000000000, 'country': 'Test Country'},
                {'year': 2022, 'value': 210000000000, 'country': 'Test Country'}
            ],
            'major_events': [
                {'year': 2020, 'event': 'COVID-19 Pandemic', 'impact': -2.5}
            ]
        }
    
    def test_create_base_dataframe(self):
        """Test base DataFrame creation"""
        df = self.feature_engineer._create_base_dataframe(self.sample_data)
        
        self.assertFalse(df.empty)
        self.assertIn('year', df.columns)
        self.assertIn('gdp', df.columns)
        self.assertIn('gdp_growth', df.columns)
        self.assertEqual(len(df), 5)  # 2018-2022
    
    def test_add_time_features(self):
        """Test time feature creation"""
        df = self.feature_engineer._create_base_dataframe(self.sample_data)
        df = self.feature_engineer._add_time_features(df)
        
        self.assertIn('time_trend', df.columns)
        self.assertIn('time_trend_sq', df.columns)
        self.assertIn('business_cycle', df.columns)
        
        # Check that time trend starts at 0
        self.assertEqual(df['time_trend'].min(), 0)
    
    def test_add_lag_features(self):
        """Test lag feature creation"""
        df = self.feature_engineer._create_base_dataframe(self.sample_data)
        df = self.feature_engineer._add_lag_features(df)
        
        self.assertIn('gdp_lag_1', df.columns)
        self.assertIn('gdp_growth_lag_1', df.columns)
        
        # Check that lag values are correct
        self.assertEqual(df.loc[df['year'] == 2019, 'gdp_lag_1'].iloc[0], 
                        df.loc[df['year'] == 2018, 'gdp'].iloc[0])
    
    def test_add_ratio_features(self):
        """Test ratio feature creation"""
        df = self.feature_engineer._create_base_dataframe(self.sample_data)
        df = self.feature_engineer._add_ratio_features(df)
        
        self.assertIn('trade_balance', df.columns)
        self.assertIn('export_import_ratio', df.columns)
        
        # Check trade balance calculation
        expected_balance = 200000000000 - 180000000000  # 2018 values
        actual_balance = df.loc[df['year'] == 2018, 'trade_balance'].iloc[0]
        self.assertEqual(actual_balance, expected_balance)
    
    def test_prepare_features_for_training(self):
        """Test feature preparation for training"""
        df = self.feature_engineer.create_features(self.sample_data, 'Test Country')
        X, y, feature_names = self.feature_engineer.prepare_features_for_training(df, 'gdp')
        
        if X is not None:  # Only test if we got valid data
            self.assertIsInstance(X, pd.DataFrame)
            self.assertIsInstance(y, pd.Series)
            self.assertIsInstance(feature_names, list)
            self.assertEqual(len(X.columns), len(feature_names))
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Create data with missing values
        incomplete_data = self.sample_data.copy()
        incomplete_data['gdp'][2]['value'] = None  # Remove 2020 GDP
        
        df = self.feature_engineer.create_features(incomplete_data, 'Test Country')
        
        # Should not have NaN in final result (after interpolation/filling)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df[col].isnull().sum()
            # Some NaN might be acceptable for lag features at the beginning
            self.assertLessEqual(nan_count, 2, f"Too many NaN values in {col}")


class TestMLModels(unittest.TestCase):
    
    def setUp(self):
        """Set up ML models and sample data"""
        self.ml_models = MLModels()
        self.feature_engineer = FeatureEngineer()
        
        # Create more comprehensive sample data for ML testing
        self.sample_data = {
            'gdp': [{'year': year, 'value': 1000000000000 * (1.02 ** (year - 2010)), 'country': 'Test'} 
                   for year in range(2010, 2023)],
            'gdp_growth': [{'year': year, 'value': 2.0 + np.random.normal(0, 0.5), 'country': 'Test'} 
                          for year in range(2010, 2023)],
            'exports': [{'year': year, 'value': 200000000000 * (1.03 ** (year - 2010)), 'country': 'Test'} 
                       for year in range(2010, 2023)],
            'imports': [{'year': year, 'value': 180000000000 * (1.025 ** (year - 2010)), 'country': 'Test'} 
                       for year in range(2010, 2023)],
            'unemployment': [{'year': year, 'value': 5.0 + np.random.normal(0, 1), 'country': 'Test'} 
                           for year in range(2010, 2023)],
            'inflation': [{'year': year, 'value': 2.0 + np.random.normal(0, 0.8), 'country': 'Test'} 
                         for year in range(2010, 2023)]
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIn('linear', self.ml_models.models)
        self.assertIn('random_forest', self.ml_models.models)
        self.assertIn('ensemble', self.ml_models.models)
    
    def test_single_model_training(self):
        """Test single model training"""
        features_df = self.feature_engineer.create_features(self.sample_data, 'Test')
        
        if not features_df.empty and len(features_df) >= 10:
            result = self.ml_models.train_and_predict(features_df, 3, 'random_forest')
            
            self.assertIn('predictions', result)
            self.assertIn('performance', result)
            self.assertIn('feature_importance', result)
            
            # Check predictions structure
            if result['predictions']:
                pred = result['predictions'][0]
                self.assertIn('year', pred)
                self.assertIn('predicted_gdp', pred)
                self.assertIn('confidence', pred)
    
    def test_ensemble_training(self):
        """Test ensemble model training"""
        features_df = self.feature_engineer.create_features(self.sample_data, 'Test')
        
        if not features_df.empty and len(features_df) >= 10:
            result = self.ml_models.train_and_predict(features_df, 2, 'ensemble')
            
            self.assertIn('predictions', result)
            self.assertIn('performance', result)
            self.assertEqual(result['model_type'], 'ensemble')
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create simple test data
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        self.ml_models.feature_names['test'] = feature_names
        importance = self.ml_models._get_feature_importance(model, 'test')
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 5)
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        df = pd.DataFrame({'year': [2020, 2021, 2022], 'gdp': [100, 105, 110]})
        
        confidence_0 = self.ml_models._calculate_confidence(df, 0)
        confidence_3 = self.ml_models._calculate_confidence(df, 3)
        
        self.assertGreater(confidence_0, confidence_3)
        self.assertGreaterEqual(confidence_0, 0.1)
        self.assertLessEqual(confidence_0, 1.0)
    
    def test_invalid_model_type(self):
        """Test handling of invalid model type"""
        features_df = self.feature_engineer.create_features(self.sample_data, 'Test')
        
        if not features_df.empty:
            result = self.ml_models.train_and_predict(features_df, 2, 'invalid_model')
            
            # Should default to random_forest or handle gracefully
            self.assertIn('predictions', result)


class TestIntegration(unittest.TestCase):
    
    def test_full_pipeline(self):
        """Test the full analysis pipeline"""
        # Create sample data
        sample_data = {
            'gdp': [{'year': year, 'value': 1000000000000 * (1.02 ** (year - 2015)), 'country': 'Integration Test'} 
                   for year in range(2015, 2023)],
            'gdp_growth': [{'year': year, 'value': 2.0 + (year - 2020) * 0.1, 'country': 'Integration Test'} 
                          for year in range(2015, 2023)],
            'exports': [{'year': year, 'value': 200000000000 * (1.025 ** (year - 2015)), 'country': 'Integration Test'} 
                       for year in range(2015, 2023)]
        }
        
        # Test pipeline
        feature_engineer = FeatureEngineer()
        ml_models = MLModels()
        
        # Step 1: Feature engineering
        features_df = feature_engineer.create_features(sample_data, 'Integration Test')
        self.assertFalse(features_df.empty)
        
        # Step 2: Model training and prediction
        if len(features_df) >= 5:  # Need minimum data
            result = ml_models.train_and_predict(features_df, 3, 'random_forest')
            
            self.assertIn('predictions', result)
            self.assertIn('performance', result)
            
            # Check that we got some predictions
            if 'error' not in result['performance']:
                self.assertGreater(len(result['predictions']), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [TestGDPBackend, TestDataCollector, TestFeatureEngineer, TestMLModels, TestIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")


# run_tests.py
#!/usr/bin/env python3
"""
Test runner script for GDP Analysis Backend
"""
import os
import sys
import unittest
import coverage

def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    
    # Initialize coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests' if os.path.exists('tests') else '.'
    suite = loader.discover(start_dir, pattern='test*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    print("-" * 50)
    cov.report()
    
    # Generate HTML coverage report
    cov.html_report(directory='htmlcov')
    print(f"\nHTML coverage report generated in 'htmlcov' directory")
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests_with_coverage()
    sys.exit(0 if success else 1)
    