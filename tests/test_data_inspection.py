# test_data_inspection.py
import sys
from pathlib import Path
import pandas as pd

print("Script starting...")

try:
    from config.fred_config import FREDConfig
    from src.data_collection.fred_collector import FREDDataCollector
    print("Successfully imported required modules")
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)

def inspect_data():
    print("\n1. Initializing components...")
    config = FREDConfig()
    collector = FREDDataCollector(config)
    print("Collector initialized successfully")

    # Test individual indicators first
    print("\nTesting individual indicators:")
    for series_id, name in collector.INDICATORS.items():
        # Note the order: series_id first, then name
        df = collector.get_indicator_data(series_id, name)
        print(f"\n{name} ({series_id}):")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Sample values:\n{df.head()}")

    print("\n2. Getting prepared model data...")
    try:
        data = collector.prepare_model_data()
        print("\nFirst 5 rows of prepared data:")
        print(data.head())
        print("\nData description:")
        print(data.describe())
        
        # Additional useful information
        print("\nData shape:", data.shape)
        print("\nColumns:", list(data.columns))
        print("\nDate range:", f"From {data.index.min()} to {data.index.max()}")
        print("\nMissing values:")
        print(data.isnull().sum())
        
    except Exception as e:
        print(f"Error inspecting data: {str(e)}")

if __name__ == "__main__":
    inspect_data()
    print("\nInspection completed!")