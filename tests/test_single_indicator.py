# test_single_indicator.py
from config.fred_config import FREDConfig
from src.data_collection.fred_collector import FREDDataCollector

config = FREDConfig()
collector = FREDDataCollector(config)

# Test single indicator
print("\nTesting UNRATE collection:")
df = collector.get_indicator_data("Unemployment Rate", "UNRATE")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nData types:", df.dtypes)
print("\nMissing values:", df.isnull().sum())