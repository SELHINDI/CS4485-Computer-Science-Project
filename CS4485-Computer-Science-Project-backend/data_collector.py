import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import requests


class DataCollector:
    """
    Collects macro data for a country.

    IMPORTANT:
    - By default this uses OFFLINE World Bank CSVs from:
        data/offline/indicators/
      e.g.
        API_NY.GDP.MKTP.CD_DS2_en_csv_v2_xxxxx.csv
        API_NY.GDP.PCAP.CD_DS2_en_csv_v2_xxxxx.csv
        API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_xxxxx.csv
        API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_xxxxx.csv
        API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_xxxxx.csv
        API_NE.EXP.GNFS.CD_DS2_en_csv_v2_xxxxx.csv
        API_NE.IMP.GNFS.CD_DS2_en_csv_v2_xxxxx.csv
        (optionally) API_SP.POP.TOTL_DS2_en_csv_....csv

    - If a CSV is missing for an indicator, we automatically fall back
      to the online World Bank API for that indicator.
    """

    def __init__(
        self,
        offline: bool = True,
        offline_dir: str = "data/offline/indicators",
    ) -> None:
        self.offline = offline
        self.offline_dir = offline_dir

        # Base URL for World Bank API (used as fallback)
        self.world_bank_base = "https://api.worldbank.org/v2"

        # Simple cache directory (mostly unused but kept for compatibility)
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Countries available in the dropdown
        # UPDATED: Using 3-letter ISO codes to match World Bank CSVs
        self.country_codes: Dict[str, str] = {
            "United States": "USA",
            "China": "CHN",
            "Germany": "DEU",
            "Japan": "JPN",
            "India": "IND",
            "United Kingdom": "GBR",
            "France": "FRA",
            "Italy": "ITA",
            "Brazil": "BRA",
            "Canada": "CAN",
            "European Union": "EUU",
        }

        # Indicator mapping that the FRONTEND EXPECTS
        self.indicators: Dict[str, str] = {
            "gdp": "NY.GDP.MKTP.CD",           # GDP (current US$)
            "gdp_per_capita": "NY.GDP.PCAP.CD",
            "gdp_growth": "NY.GDP.MKTP.KD.ZG",
            "inflation": "FP.CPI.TOTL.ZG",
            "unemployment": "SL.UEM.TOTL.ZS",
            "exports": "NE.EXP.GNFS.CD",
            "imports": "NE.IMP.GNFS.CD",
            "population": "SP.POP.TOTL",       # CSV optional; API fallback works
        }

    # ------------------------------------------------------------------
    # Public API used by app.py
    # ------------------------------------------------------------------

    def get_available_countries(self) -> List[Dict[str, str]]:
        """Return list of countries as [{'name': ..., 'code': ...}, ...]."""
        # Frontend might expect 2-letter codes for flags/UI, but backend needs 3-letter for data.
        # For now, let's expose the 3-letter codes or map them back if needed.
        # Actually, the frontend sends whatever we return here as 'code'.
        # If we change this to 3-letter, frontend might break if it relies on 2-letter for flags.
        # Let's keep returning what we have, but ensure we handle inputs correctly.
        
        # To be safe, let's return the keys (names) and values (3-letter codes).
        # If frontend sends "US", we need to handle that.
        return [
            {"name": name, "code": code}
            for name, code in self.country_codes.items()
        ]

    def _normalize_country_code(self, code: str) -> str:
        """Convert 2-letter code to 3-letter code if needed."""
        mapping = {
            "US": "USA", "CN": "CHN", "DE": "DEU", "JP": "JPN", 
            "IN": "IND", "GB": "GBR", "FR": "FRA", "IT": "ITA", 
            "BR": "BRA", "CA": "CAN", "EU": "EUU"
        }
        return mapping.get(code.upper(), code.upper())

    def get_country_data(
        self,
        country: str,
        start_year: int = 1990,
        end_year: int = 2023,
    ) -> Dict[str, Any]:
        """
        Main method used by the backend.
        """
        # Convert country name to code if needed
        if country in self.country_codes:
            country_code = self.country_codes[country]
        else:
            # It might be a code already (e.g. "US" or "USA")
            country_code = self._normalize_country_code(country)

        data: Dict[str, Any] = {}

        data: Dict[str, Any] = {}

        for name, indicator_code in self.indicators.items():
            series = self._get_indicator_series(
                country_code, indicator_code, start_year, end_year
            )
            data[name] = series
            # Tiny delay to be gentle if we fall back to the API
            time.sleep(0.05)

        # Add a couple of major events so charts have something to show
        data["major_events"] = [
            {
                "year": 2008,
                "event": "Global Financial Crisis",
                "impact": -2.0,
            },
            {
                "year": 2020,
                "event": "COVID-19 Pandemic",
                "impact": -2.5,
            },
        ]

        return data

    # ------------------------------------------------------------------
    # Offline CSV helpers
    # ------------------------------------------------------------------

    def _find_indicator_csv(self, indicator_code: str) -> Optional[str]:
        """
        Look in self.offline_dir for a CSV whose filename contains
        the indicator code (case-insensitive).
        """
        if not self.offline:
            return None
        if not os.path.isdir(self.offline_dir):
            return None

        for fname in os.listdir(self.offline_dir):
            if indicator_code.lower() in fname.lower():
                return os.path.join(self.offline_dir, fname)
        return None

    def _detect_header_index(self, csv_path: str) -> int:
        """
        World Bank CSVs have a few metadata rows at the top.
        Find the row that actually contains:
        Country Name, Country Code, Indicator Name, Indicator Code, 1960, 1961, ...
        """
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for idx, line in enumerate(f):
                lower = line.lower()
                if (
                    "country name" in lower
                    and "country code" in lower
                    and "indicator name" in lower
                    and "indicator code" in lower
                ):
                    return idx
        # If we don't find it, assume standard World Bank layout with header at line 4
        return 4

    def _read_indicator_csv(
        self,
        csv_path: str,
        country_code: str,
        start_year: int,
        end_year: int,
    ) -> List[Dict[str, Any]]:
        """Read one World Bank CSV and return [{year, value}, ...] for the country."""
        header_idx = self._detect_header_index(csv_path)

        df = pd.read_csv(
            csv_path,
            skiprows=header_idx,
            header=0,
            encoding="utf-8-sig",
        )

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Find the Country Code column
        cc_col = None
        for c in df.columns:
            if c.lower() == "country code":
                cc_col = c
                break

        if cc_col is None:
            raise KeyError(f"'Country Code' column not found in {csv_path}")

        # Filter to our country
        mask = df[cc_col].astype(str).str.upper() == country_code.upper()
        country_rows = df[mask]

        if country_rows.empty:
            # No data for this country in this file
            return []

        row = country_rows.iloc[0]

        # Year columns are numeric column names like "1960", "2015", etc.
        year_values: List[Dict[str, Any]] = []
        for col in df.columns:
            col_str = str(col).strip()
            if col_str.isdigit():
                year = int(col_str)
                if start_year <= year <= end_year:
                    val = row[col_str]
                    if pd.isna(val):
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    year_values.append({"year": year, "value": v})

        # Sort by year to avoid surprises
        year_values.sort(key=lambda x: x["year"])
        print(
            f"[OFFLINE] {os.path.basename(csv_path)} "
            f"{country_code}: {len(year_values)} points from {start_year}-{end_year}"
        )
        return year_values

    def _get_indicator_series(
        self,
        country_code: str,
        indicator_code: str,
        start_year: int,
        end_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Decide whether to read from local CSV, local cache, or from the World Bank API.
        Always returns a list (possibly empty) and never raises to the caller.
        """
        # 1) Try offline CSV
        csv_path = self._find_indicator_csv(indicator_code)
        if csv_path:
            try:
                return self._read_indicator_csv(
                    csv_path, country_code, start_year, end_year
                )
            except Exception as e:
                print(
                    f"Error reading offline CSV for {indicator_code} "
                    f"({country_code}): {e}"
                )

        # 2) Try local JSON cache
        cache_path = self._get_cache_path(country_code, indicator_code)
        cached_data = self._read_from_cache(cache_path)
        if cached_data is not None:
            print(f"[CACHE] Hit for {indicator_code} ({country_code})")
            # Filter by year range
            return [
                d for d in cached_data 
                if start_year <= d["year"] <= end_year
            ]

        # 3) Fallback to online World Bank API
        data = self.get_world_bank_data(
            country_code, indicator_code, start_year, end_year
        )

        # 4) Save to cache if we got data
        if data:
            self._save_to_cache(cache_path, data)
        
        return data

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_path(self, country_code: str, indicator_code: str) -> str:
        """Generate a filename for the cache."""
        safe_country = "".join(c for c in country_code if c.isalnum() or c in ('-', '_'))
        safe_indicator = "".join(c for c in indicator_code if c.isalnum() or c in ('-', '_', '.'))
        filename = f"{safe_country}_{safe_indicator}.json"
        return os.path.join(self.cache_dir, filename)

    def _read_from_cache(self, cache_path: str) -> Optional[List[Dict[str, Any]]]:
        """Read data from a JSON cache file."""
        import json
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache {cache_path}: {e}")
            return None

    def _save_to_cache(self, cache_path: str, data: List[Dict[str, Any]]) -> None:
        """Save data to a JSON cache file."""
        import json
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[CACHE] Saved {len(data)} points to {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"Error saving cache {cache_path}: {e}")

    # ------------------------------------------------------------------
    # Online World Bank fallback
    # ------------------------------------------------------------------

    def get_world_bank_data(
        self,
        country_code: str,
        indicator: str,
        start_year: int,
        end_year: int,
    ) -> List[Dict[str, Any]]:
        """Fetch data from World Bank API (used as fallback)."""
        try:
            url = f"{self.world_bank_base}/country/{country_code}/indicator/{indicator}"
            params = {
                "format": "json",
                "date": f"{start_year}:{end_year}",
                "per_page": 1000,
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if len(data) > 1 and data[1]:
                result: List[Dict[str, Any]] = []
                for item in data[1]:
                    if item["value"] is not None:
                        result.append(
                            {
                                "year": int(item["date"]),
                                "value": float(item["value"]),
                                "country": item["country"]["value"],
                            }
                        )
                print(
                    f"[ONLINE] {indicator} {country_code}: "
                    f"{len(result)} points from {start_year}-{end_year}"
                )
                return sorted(result, key=lambda x: x["year"])
            return []

        except Exception as e:
            print(
                f"Error fetching {indicator} for {country_code} "
                f"from World Bank: {str(e)}"
            )
            return []
