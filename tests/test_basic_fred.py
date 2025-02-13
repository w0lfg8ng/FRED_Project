import sys
print("0. Starting script...")

from pathlib import Path
print("1. Imported Path")

import pandas as pd
print("2. Imported pandas")

print("3. About to import FREDConfig...")
try:
    from config.fred_config import FREDConfig
    print("4. Successfully imported FREDConfig")
except Exception as e:
    print(f"Error importing FREDConfig: {str(e)}")
    sys.exit(1)

print("5. About to import FREDDataCollector...")
try:
    from src.data_collection.fred_collector import FREDDataCollector
    print("6. Successfully imported FREDDataCollector")
except Exception as e:
    print(f"Error importing FREDDataCollector: {str(e)}")
    sys.exit(1)

def test_fred_collector():
    print("7. Inside test_fred_collector function")
    
    print("8. Creating config...")
    config = FREDConfig()
    print(f"9. API Key from config: {config.api_key}")
    
    print("10. Creating collector...")
    collector = FREDDataCollector(config)
    print("11. Collector created")

    print("\n12. Testing direct FRED API call...")
    try:
        print("13. About to call FRED API...")
        raw_series = collector.fred.get_series('UNRATE')
        print("14. Got response from FRED")
        print("Raw series type:", type(raw_series))
        print("First few values:")
        print(raw_series.head())
        
        print("\n15. Converting to DataFrame...")
        df = pd.DataFrame(raw_series)
        print("16. Conversion successful")
        print("\nDataFrame after conversion:")
        print(df.head())
        
    except Exception as e:
        print(f"Error with direct API call: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Main block starting...")
    test_fred_collector()
    print("\nTesting completed!")