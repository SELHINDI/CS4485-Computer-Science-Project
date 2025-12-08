import sys
import os
import shutil

# Add current directory to path
sys.path.append(os.getcwd())

from data_collector import DataCollector

def test_caching():
    print("Initializing DataCollector...")
    # Point to a dummy offline dir so we force API usage (unless we pick a country not in the real offline dir)
    # But simpler: let's pick a country/indicator combo that we know is NOT in the offline CSVs.
    # The current offline CSVs seem to cover all indicators but maybe not all countries?
    # Actually, the offline CSVs are global (World Bank format usually contains all countries).
    # So to test caching, we might need to temporarily rename the offline folder or use a fake indicator code if we can mock the API.
    # 
    # Better approach: The offline CSVs in `data/offline/indicators` are specific files.
    # Let's try to fetch data for a country, but first ensure we are NOT using the offline CSVs for this test.
    # We can do this by initializing DataCollector with offline=False, BUT `_get_indicator_series` checks `self.offline`?
    # Wait, `_get_indicator_series` calls `_find_indicator_csv`.
    # If we rename `data/offline/indicators` to `data/offline/indicators_backup`, `_find_indicator_csv` will return None.
    
    offline_dir = "data/offline/indicators"
    backup_dir = "data/offline/indicators_backup"
    
    if os.path.exists(offline_dir):
        print(f"Temporarily renaming {offline_dir} to {backup_dir} to force API and Cache usage...")
        os.rename(offline_dir, backup_dir)
    
    try:
        collector = DataCollector(offline=True) 
        
        # Monkeypatch get_world_bank_data to avoid network calls and timeouts
        original_get_wb_data = collector.get_world_bank_data
        def mock_get_world_bank_data(country_code, indicator, start_year, end_year):
            print(f"[MOCK] Returning dummy data for {indicator}...")
            return [
                {"year": 2020, "value": 100.0, "country": "United States"},
                {"year": 2021, "value": 110.0, "country": "United States"},
                {"year": 2022, "value": 120.0, "country": "United States"},
            ]
        collector.get_world_bank_data = mock_get_world_bank_data
        
        country = "United States"
        indicator = "gdp" # NY.GDP.MKTP.CD
        
        # Clear any existing cache for this
        cache_path = collector._get_cache_path(collector.country_codes[country], collector.indicators[indicator])
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Cleared existing cache: {cache_path}")
            
        print("\n--- Step 1: Fetch from API (Mocked) -> Cache Miss ---")
        # This should hit our mock and save to cache
        data1 = collector._get_indicator_series(
            collector.country_codes[country], 
            collector.indicators[indicator], 
            2020, 2022
        )
        print(f"Fetched {len(data1)} points.")
        
        if not os.path.exists(cache_path):
            print("FAILURE: Cache file was NOT created.")
        else:
            print(f"SUCCESS: Cache file created at {cache_path}")
            
        print("\n--- Step 2: Fetch from Cache (Cache Hit) ---")
        # Restore original method to prove we DON'T call it (if we did, it would print [MOCK] or fail if we didn't mock it)
        # But better: modify the cache file to prove we read from disk
        
        import json
        with open(cache_path, 'r') as f:
            cached_content = json.load(f)
        
        # Modify a value in the file
        if cached_content:
            cached_content[0]['value'] = 999999999
            with open(cache_path, 'w') as f:
                json.dump(cached_content, f)
            print("Modified cache file to verify read...")
            
            # We can also un-monkeypatch to ensure we don't hit the mock
            collector.get_world_bank_data = original_get_wb_data
            
            data2 = collector._get_indicator_series(
                collector.country_codes[country], 
                collector.indicators[indicator], 
                2020, 2022
            )
            
            if data2 and data2[0]['value'] == 999999999:
                print("SUCCESS: Read from cache (saw modified value).")
            else:
                print(f"FAILURE: Did not read modified value. Got {data2[0]['value'] if data2 else 'None'}")
        
    finally:
        # Restore offline dir
        if os.path.exists(backup_dir):
            print(f"Restoring {offline_dir}...")
            os.rename(backup_dir, offline_dir)

if __name__ == "__main__":
    test_caching()
