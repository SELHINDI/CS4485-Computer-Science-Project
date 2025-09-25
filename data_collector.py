import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os

class DataCollector:
    """
    Handles data collection from various sources including World Bank API
    """
    
    def __init__(self):
        self.world_bank_base = "https://api.worldbank.org/v2"
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Common country codes mapping
        self.country_codes = {
            'United States': 'US',
            'China': 'CN', 
            'Germany': 'DE',
            'Japan': 'JP',
            'India': 'IN',
            'United Kingdom': 'GB',
            'France': 'FR',
            'Italy': 'IT',
            'Brazil': 'BR',
            'Canada': 'CA'
        }
        
        # GDP and related indicators from World Bank
        self.indicators = {
            'gdp': 'NY.GDP.MKTP.CD',
            'gdp_per_capita': 'NY.GDP.PCAP.CD',
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
            'inflation': 'FP.CPI.TOTL.ZG',
            'unemployment': 'SL.UEM.TOTL.ZS',
            'exports': 'NE.EXP.GNFS.CD',
            'imports': 'NE.IMP.GNFS.CD',
            'population': 'SP.POP.TOTL'
        }
    
    def get_available_countries(self):
        """Get list of available countries"""
        return [{'id': code, 'name': name, 'region': 'Unknown', 'income_level': 'Unknown'} 
                for name, code in self.country_codes.items()]
    
    def get_world_bank_data(self, country_code, indicator, start_year, end_year):
        """Fetch data from World Bank API"""
        try:
            url = f"{self.world_bank_base}/country/{country_code}/indicator/{indicator}"
            params = {
                'format': 'json',
                'date': f"{start_year}:{end_year}",
                'per_page': 1000
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) > 1 and data[1]:
                result = []
                for item in data[1]:
                    if item['value'] is not None:
                        result.append({
                            'year': int(item['date']),
                            'value': float(item['value']),
                            'country': item['country']['value']
                        })
                return result
            return []
            
        except Exception as e:
            print(f"Error fetching {indicator} for {country_code}: {str(e)}")
            return []
    
    def get_country_data(self, country, start_year=1990, end_year=2023):
        """Get comprehensive country data"""
        # Convert country name to code if needed
        if country in self.country_codes:
            country_code = self.country_codes[country]
        else:
            country_code = country
        
        data = {}
        
        # Fetch all indicators
        for name, indicator in self.indicators.items():
            data[name] = self.get_world_bank_data(country_code, indicator, start_year, end_year)
            time.sleep(0.1)  # Rate limiting
        
        # Add major events (simplified)
        data['major_events'] = [
            {'year': 2008, 'event': 'Global Financial Crisis', 'impact': -2.0},
            {'year': 2020, 'event': 'COVID-19 Pandemic', 'impact': -2.5}
        ]
        
        return data
