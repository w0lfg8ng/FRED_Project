from datetime import datetime, timedelta
import pandas as pd
from fredapi import Fred
from typing import Dict, List, Optional
import os
import json

class FREDCollector:
    """Handles data collection and caching from FRED API."""
    
    # Key economic indicators we'll track
    # Replace the existing INDICATORS dictionary
    INDICATORS = {
        'GDPC1': 'Real GDP',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'FEDFUNDS': 'Federal Funds Rate',
        'T10Y2Y': 'Treasury Yield Spread',
        'INDPRO': 'Industrial Production',
        'RSAFS': 'Retail Sales',
        'M2SL': 'Money Supply',          # Added: M2 Money Supply
        'NFCI': 'Financial Conditions',  # Added: Chicago Fed National Financial Conditions Index
        'HOUST': 'Housing Starts',       # Added: Housing Starts
        'TOTALSA': 'Vehicle Sales'       # Added: Total Vehicle Sales
    }

    def __init__(self, config):
        """Initialize collector with configuration."""
        self.config = config
        self.fred = Fred(api_key=config.api_key)
        self.cache_dir = os.path.join('data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_indicator_data(self, series_id: str, indicator_name: str) -> pd.DataFrame:
        """Fetch data for a single indicator, using cache if available."""
        cache_file = os.path.join(self.cache_dir, f"{series_id}.json")
    
        try:
            # Check cache first
            if self._is_cache_valid(cache_file):
                try:
                    return self._load_from_cache(cache_file)
                except (json.JSONDecodeError, FileNotFoundError):
                    # If cache is corrupt or missing, proceed to fetch new data
                    pass
                
            # Fetch from FRED using series_id
            series = self.fred.get_series(series_id)
            # Convert the series to a DataFrame, preserving the date index
            df = pd.DataFrame(series)
            df.columns = ['value']  # Name the column
            df.index.name = 'date'  # Name the index
        
            # Save to cache
            self._save_to_cache(cache_file, df)
        
            return df
        except Exception as e:
            print(f"Error fetching data for {series_id}: {str(e)}")
            raise

    def collect_all_indicators(self) -> Dict[str, pd.DataFrame]:
        """Collect all configured indicators."""
        data = {}
    
        for series_id, indicator_name in self.INDICATORS.items():
            try:
                data[indicator_name] = self.get_indicator_data(series_id, indicator_name)
                print(f"Successfully collected {indicator_name} data")
            except Exception as e:
                print(f"Error collecting {indicator_name} data: {str(e)}")
                
        return data

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(cache_file):
            return False
            
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age = datetime.now() - file_time
        
        return age.total_seconds() < self.config.cache_duration
        
    def _save_to_cache(self, cache_file: str, df: pd.DataFrame) -> None:
        """Save data to cache file."""
        # Convert timestamps to string format before saving to JSON
        df_copy = df.copy()
        df_copy.index = df_copy.index.strftime('%Y-%m-%d')
        data = df_copy.reset_index().to_dict(orient='records')
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
    def _load_from_cache(self, cache_file: str) -> pd.DataFrame:
        """Load data from cache file."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
            # Convert the loaded data to a DataFrame and parse dates
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived economic features to the dataset."""
        # Calculate year-over-year changes with explicit NaN handling
        for col in df.columns:
            if col not in ['date']:
                df[f'{col}_yoy_change'] = df[col].pct_change(periods=12, fill_method=None) * 100

        # Calculate rolling averages with minimum periods
        for col in [c for c in df.columns if not c.endswith('_yoy_change')]:
            df[f'{col}_6m_avg'] = df[col].rolling(window=6, min_periods=3).mean()
                
        return df
            
    def prepare_model_data(self) -> pd.DataFrame:
        """Prepare consolidated dataset for modeling."""
        data = self.collect_all_indicators()
    
        print("\nInitial data collection complete...")
        print("\nInitial data shapes:")
        for name, df in data.items():
            print(f"{name}: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
    
        # Inside prepare_model_data, replace the data merging section with this:
        # Create master dataframe with proper date range
        min_date = max(df.index.min() for df in data.values())  # Use max of min dates
        max_date = min(df.index.max() for df in data.values())  # Use min of max dates
        date_range = pd.date_range(start=min_date, end=max_date, freq='M')
        master_df = pd.DataFrame(index=date_range)
    
        # Add each indicator
        for name, df in data.items():
            print(f"\nProcessing {name}...")
            print(f"Before merge: {df.shape}")
        
            # Convert to monthly end frequency
            if name == 'Real GDP':  # Handle quarterly GDP data
                # Resample to monthly and forward fill up to 3 months
                df = df.resample('M').ffill(limit=2)
            else:
                # For other indicators, resample to month-end
                df = df.resample('M').last()
        
            # Merge data on dates
            master_df[name] = df['value']
            print(f"After merge: {master_df[name].notna().sum()} non-null values")
       
        # Special handling for specific indicators - using column names directly
        if 'Financial Conditions' in master_df.columns:
            master_df['Financial Conditions'] = master_df['Financial Conditions'].ffill(limit=3)

        if 'Money Supply' in master_df.columns:
            master_df['Money Supply'] = master_df['Money Supply'].ffill(limit=2)
            master_df['Money Supply'] = master_df['Money Supply'].rolling(window=3, min_periods=1).mean()

        if 'Housing Starts' in master_df.columns:
            master_df['Housing Starts'] = master_df['Housing Starts'].ffill(limit=2)
            master_df['Housing Starts'] = master_df['Housing Starts'].rolling(window=3, min_periods=1).mean()

        if 'Vehicle Sales' in master_df.columns:
            master_df['Vehicle Sales'] = master_df['Vehicle Sales'].ffill(limit=2)
            master_df['Vehicle Sales'] = master_df['Vehicle Sales'].rolling(window=3, min_periods=1).mean()
        
        # Handle missing values for base indicators
        master_df = master_df.ffill(limit=2)
        master_df = master_df.interpolate(method='linear', limit=3)
    
        print("\nMissing values after filling:")
        print(master_df.isnull().sum())
    
        # Add derived features
        print("\nAdding derived features...")
    
        # Year-over-year changes
        for col in master_df.columns:
            print(f"Processing YoY change for {col}")
            master_df[f'{col}_yoy_change'] = master_df[col].ffill().pct_change(periods=12) * 100

        # 6-month moving averages
        for col in master_df.columns:
            if not col.endswith('_yoy_change'):
                print(f"Processing 6m avg for {col}")
                master_df[f'{col}_6m_avg'] = master_df[col].rolling(window=6, min_periods=3).mean()
    
        print("\nShape after adding derived features:", master_df.shape)
    
        # Drop rows where all base indicators are NaN
        base_columns = [col for col in master_df.columns if not (col.endswith('_yoy_change') or col.endswith('_6m_avg'))]
        print("\nBase columns:", base_columns)
    
        # Only drop if ALL base indicators are missing
        master_df = master_df.dropna(subset=base_columns, how='all')
        
        # Verify we have actual data
        if master_df.empty or master_df.shape[0] == 0:
            raise ValueError("No valid data after processing. Please check the input data and processing steps.")
        
        # Drop any future dates
        master_df = master_df[master_df.index <= pd.Timestamp.now()]
    
        print("\nFinal shape:", master_df.shape)
        print("Final missing values count:")
        print(master_df.isnull().sum())
    
        return master_df

    def get_latest_values(self) -> Dict[str, float]:
        """Get the most recent values for all indicators."""
        data = self.collect_all_indicators()
        latest_values = {}
        
        for name, df in data.items():
            latest_values[name] = df['value'].iloc[-1]
            
        return latest_values
    
    def get_latest_data(self):
        """Get the most recent model data."""
        try:
            # Use the existing prepare_model_data method and get the last row
            model_data = self.prepare_model_data()
            if model_data is not None and not model_data.empty:
                return model_data.iloc[[-1]]  # Return last row
            return None
        except Exception as e:
            print(f"Error getting latest data: {str(e)}")
            return None

    def get_historical_data(self):
        """Get historical model data."""
        try:
            # Return all prepared model data
            return self.prepare_model_data()
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return None