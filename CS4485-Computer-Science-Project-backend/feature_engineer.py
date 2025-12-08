import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Handles feature engineering for GDP prediction models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_features(self, raw_data, country):
        """Create engineered features from raw economic data"""
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
        
        # Create ratio features
        df = self._add_ratio_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _create_base_dataframe(self, raw_data):
        """Create base DataFrame from raw data"""
        try:
            all_years = set()
            for indicator, data_list in raw_data.items():
                if data_list and isinstance(data_list, list) and indicator != 'major_events':
                    years = {item['year'] for item in data_list if 'year' in item}
                    all_years.update(years)
            
            if not all_years:
                return pd.DataFrame()
            
            df = pd.DataFrame({'year': sorted(all_years)})
            
            for indicator, data_list in raw_data.items():
                if data_list and isinstance(data_list, list) and indicator != 'major_events':
                    value_dict = {item['year']: item['value'] for item in data_list 
                                 if 'year' in item and 'value' in item}
                    df[indicator] = df['year'].map(value_dict)
            
            return df
            
        except Exception as e:
            print(f"Error creating base DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _add_time_features(self, df):
        """Add time-based features"""
        if 'year' not in df.columns:
            return df
            
        df['time_trend'] = df['year'] - df['year'].min()
        df['time_trend_sq'] = df['time_trend'] ** 2
        df['business_cycle'] = np.sin(2 * np.pi * df['time_trend'] / 10)
        
        return df
    
    def _add_lag_features(self, df, max_lags=2):
        """Add lagged features for key indicators"""
        lag_indicators = ['gdp', 'gdp_growth', 'exports', 'imports']
        
        for indicator in lag_indicators:
            if indicator in df.columns:
                for lag in range(1, max_lags + 1):
                    df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df, windows=[3]):
        """Add rolling window features"""
        rolling_indicators = ['gdp_growth', 'inflation']
        
        for indicator in rolling_indicators:
            if indicator in df.columns:
                for window in windows:
                    df[f'{indicator}_ma_{window}'] = df[indicator].rolling(window=window).mean()
        
        return df
    
    def _add_ratio_features(self, df):
        """Add ratio features"""
        if 'exports' in df.columns and 'imports' in df.columns:
            df['trade_balance'] = df['exports'] - df['imports']
            df['export_import_ratio'] = np.where(df['imports'] != 0, 
                                               df['exports'] / df['imports'], np.nan)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Use forward fill and backward fill
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Fill remaining missing values with column mean
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def prepare_features_for_training(self, df, target_column='gdp'):
        """Prepare features for model training"""
        if df.empty or target_column not in df.columns:
            return None, None, []
        
        exclude_cols = ['year', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if X.empty:
            return None, None, []
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return X, y, feature_cols
