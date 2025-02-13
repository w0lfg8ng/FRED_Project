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

def test_fred_collector():
    print("\n1. Initializing components...")
    config = FREDConfig()
    collector = FREDDataCollector(config)
    print("Collector initialized successfully")

    print("\n2. Testing single indicator collection (UNRATE)...")
    try:
        df = collector.get_indicator_data("Unemployment Rate", "UNRATE")
        print(f"✓ Successfully fetched UNRATE data")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Latest value: {df['value'].iloc[-1]:.1f}%")
    except Exception as e:
        print(f"✗ Error fetching UNRATE: {str(e)}")

    print("\n3. Testing cache functionality...")
    try:
        # Fetch again - should use cache
        df_cached = collector.get_indicator_data("Unemployment Rate", "UNRATE")
        print("✓ Successfully retrieved from cache")
        print(f"  Cached data shape matches: {df.shape == df_cached.shape}")
    except Exception as e:
        print(f"✗ Error testing cache: {str(e)}")

    print("\n4. Testing multiple indicators collection...")
    try:
        data_dict = collector.collect_all_indicators()
        print(f"Retrieved {len(data_dict)} indicators:")
        for name, df in data_dict.items():
            print(f"  ✓ {name}: {df.shape[0]} rows")
    except Exception as e:
        print(f"✗ Error collecting all indicators: {str(e)}")

    print("\n5. Testing model data preparation...")
    try:
        model_data = collector.prepare_model_data()
        print("✓ Successfully prepared model data")
        print(f"  Shape: {model_data.shape}")
        print("\n  Features:")
        for col in model_data.columns:
            print(f"    - {col}")
    except Exception as e:
        print(f"✗ Error preparing model data: {str(e)}")

    print("\n6. Testing latest values retrieval...")
    try:
        latest = collector.get_latest_values()
        print("Latest values for all indicators:")
        for indicator, value in latest.items():
            print(f"  ✓ {indicator}: {value:.2f}")
    except Exception as e:
        print(f"✗ Error getting latest values: {str(e)}")

if __name__ == "__main__":
    test_fred_collector()
    print("\nTesting completed!")