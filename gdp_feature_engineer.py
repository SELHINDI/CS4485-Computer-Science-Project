import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Handles feature engineering for GDP prediction models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_features(self, raw_data, country):
        """
        Create engineered features from raw economic data
        """
        # Convert raw data to DataFrame
        df = self._create_base_dataframe(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by year
        df = df.sort_values('year').reset_index(drop=True)
        
        # Create time-based features
        df = self._add_time_features(df)
        
        # Create lag features
        df = self._add_lag_features(df)
        
        # Create rolling window features
        df = self._add_rolling_features(df)
        
        # Create ratio and interaction features
        df = self._add_ratio_features(df)
        
        # Create external factor features
        df = self._add_external_features(df, raw_data, country)
        
        # Create trend features
        df = self._add_trend_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _create_base_dataframe(self, raw_data):
        """Create base DataFrame from raw data"""
        try:
            # Get all years from any non-empty dataset
            all_years = set()
            for indicator, data_list in raw_data.items():
                if data_list and isinstance(data_list, list):
                    years = {item['year'] for item in data_list if 'year' in item}
                    all_years.update(years)
            
            if not all_years:
                return pd.DataFrame()
            
            # Create base DataFrame with all years
            df = pd.DataFrame({'year': sorted(all_years)})
            
            # Add each indicator
            for indicator, data_list in raw_data.items():
                if data_list and isinstance(data_list, list) and indicator != 'major_events':
                    # Create a dictionary for fast lookup
                    value_dict = {item['year']: item['value'] for item in data_list 
                                 if 'year' in item and 'value' in item}
                    
                    # Map values to years
                    df[indicator] = df['year'].map(value_dict)
            
            return df
            
        except Exception as e:
            print(f"Error creating base DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _add_time_features(self, df):
        """Add time-based features"""
        if 'year' not in df.columns:
            return df
            
        # Linear time trend
        df['time_trend'] = df['year'] - df['year'].min()
        
        # Quadratic time trend
        df['time_trend_sq'] = df['time_trend'] ** 2
        
        # Decade features
        df['decade'] = (df['year'] // 10) * 10
        
        # Business cycle approximation (simplified 10-year cycle)
        df['business_cycle'] = np.sin(2 * np.pi * df['time_trend'] / 10)
        
        return df
    
    def _add_lag_features(self, df, max_lags=3):
        """Add lagged features for key indicators"""
        lag_indicators = ['gdp', 'gdp_growth', 'inflation', 'unemployment', 
                         'exports', 'imports', 'fdi']
        
        for indicator in lag_indicators:
            if indicator in df.columns:
                for lag in range(1, max_lags + 1):
                    df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df, windows=[3, 5]):
        """Add rolling window features"""
        rolling_indicators = ['gdp_growth', 'inflation', 'unemployment']
        
        for indicator in rolling_indicators:
            if indicator in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{indicator}_ma_{window}'] = df[indicator].rolling(window=window).mean()
                    
                    # Rolling standard deviation
                    df[f'{indicator}_std_{window}'] = df[indicator].rolling(window=window).std()
                    
                    # Rolling min/max
                    df[f'{indicator}_min_{window}'] = df[indicator].rolling(window=window).min()
                    df[f'{indicator}_max_{window}'] = df[indicator].rolling(window=window).max()
        
        return df
    
    def _add_ratio_features(self, df):
        """Add ratio and interaction features"""
        
        # Trade ratios
        if 'exports' in df.columns and 'imports' in df.columns:
            df['trade_balance'] = df['exports'] - df['imports']
            df['export_import_ratio'] = np.where(df['imports'] != 0, 
                                               df['exports'] / df['imports'], np.nan)
        
        # Economic efficiency ratios
        if 'gdp' in df.columns and 'population' in df.columns:
            df['gdp_per_capita_calc'] = np.where(df['population'] != 0,
                                               df['gdp'] / df['population'], np.nan)
        
        if 'exports' in df.columns and 'gdp' in df.columns:
            df['export_gdp_ratio'] = np.where(df['gdp'] != 0,
                                            df['exports'] / df['gdp'], np.nan)
        
        if 'government_expenditure' in df.columns and 'gdp' in df.columns:
            df['govt_spending_ratio'] = np.where(df['gdp'] != 0,
                                               df['government_expenditure'] / df['gdp'], np.nan)
        
        # Investment ratios
        if 'fdi' in df.columns and 'gdp' in df.columns:
            df['fdi_gdp_ratio'] = np.where(df['gdp'] != 0,
                                         df['fdi'] / df['gdp'], np.nan)
        
        return df
    
    def _add_external_features(self, df, raw_data, country):
        """Add external factor features"""
        
        # Major events impact
        if 'major_events' in raw_data and raw_data['major_events']:
            df['crisis_impact'] = 0.0
            for event in raw_data['major_events']:
                year_idx = df['year'] == event['year']
                if year_idx.any():
                    df.loc[year_idx, 'crisis_impact'] = event['impact']
        else:
            df['crisis_impact'] = 0.0
        
        # Oil price features
        if 'oil_price_trend' in raw_data and raw_data['oil_price_trend']:
            oil_dict = {item['year']: item['price'] for item in raw_data['oil_price_trend']}
            df['oil_price'] = df['year'].map(oil_dict)
            
            # Oil price changes
            df['oil_price_change'] = df['oil_price'].pct_change()
            df['oil_price_volatility'] = df['oil_price'].rolling(window=3).std()
        
        # Global economic indicators (simplified)
        df['global_recession'] = 0
        recession_years = [1991, 2001, 2008, 2009, 2020]
        df.loc[df['year'].isin(recession_years), 'global_recession'] = 1
        
        return df
    
    def _add_trend_features(self, df):
        """Add trend and momentum features"""
        
        # GDP growth momentum
        if 'gdp_growth' in df.columns:
            df['gdp_growth_momentum'] = df['gdp_growth'].diff()
            df['gdp_growth_acceleration'] = df['gdp_growth_momentum'].diff()
        
        # Economic sentiment (based on multiple indicators)
        sentiment_indicators = ['gdp_growth', 'unemployment', 'inflation']
        available_indicators = [col for col in sentiment_indicators if col in df.columns]
        
        if available_indicators:
            # Create a simple economic sentiment score
            temp_df = df[available_indicators].copy()
            
            # Normalize unemployment and inflation (lower is better)
            if 'unemployment' in temp_df.columns:
                temp_df['unemployment'] = -temp_df['unemployment']  # Invert
            if 'inflation' in temp_df.columns:
                temp_df['inflation'] = -np.abs(temp_df['inflation'] - 2)  # Target 2% inflation
            
            # Calculate sentiment as average z-score
            sentiment_scores = []
            for _, row in temp_df.iterrows():
                valid_values = [v for v in row.values if not np.isnan(v)]
                if valid_values:
                    sentiment_scores.append(np.mean(valid_values))
                else:
                    sentiment_scores.append(np.nan)
            
            df['economic_sentiment'] = sentiment_scores
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Forward fill then backward fill for time series data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For remaining missing values, use interpolation
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear')
        
        # Fill any remaining missing values with column mean
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        return df
    
    def prepare_features_for_training(self, df, target_column='gdp'):
        """Prepare features for model training"""
        
        if df.empty or target_column not in df.columns:
            return None, None, []
        
        # Remove non-predictive columns
        exclude_cols = ['year', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get features and target
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Remove rows where target is missing
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if X.empty:
            return None, None, []
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return X, y, feature_cols
    
    def get_feature_importance_explanation(self, feature_names, importances):
        """Get human-readable explanation of feature importance"""
        
        feature_explanations = {
            'gdp_lag_1': 'Previous year GDP (strong economic continuity indicator)',
            'gdp_growth': 'Annual GDP growth rate',
            'gdp_growth_lag_1': 'Previous year GDP growth rate',
            'exports': 'Total exports value',
            'imports': 'Total imports value',
            'trade_balance': 'Export-import balance',
            'export_import_ratio': 'Ratio of exports to imports',
            'unemployment': 'Unemployment rate',
            'inflation': 'Inflation rate',
            'population': 'Total population',
            'fdi': 'Foreign direct investment',
            'government_expenditure': 'Government spending',
            'time_trend': 'Linear time trend',
            'business_cycle': 'Business cycle indicator',
            'crisis_impact': 'Major crisis/event impact',
            'oil_price': 'Oil price level',
            'economic_sentiment': 'Overall economic sentiment score',
            'trade_openness': 'Trade openness ratio',
            'export_gdp_ratio': 'Exports as percentage of GDP',
            'fdi_gdp_ratio': 'FDI as percentage of GDP'
        }
        
        explained_features = []
        
        for feature, importance in zip(feature_names, importances):
            explanation = feature_explanations.get(feature, f'Economic indicator: {feature}')
            explained_features.append({
                'feature': feature,
                'importance': float(importance),
                'explanation': explanation
            })
        
        return sorted(explained_features, key=lambda x: x['importance'], reverse=True)