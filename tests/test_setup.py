from config.fred_config import FREDConfig
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import fredapi
import joblib

print("All imports successful!")

# Test FRED config
config = FREDConfig()
print(f"API Key loaded: {'*' * len(config.api_key)}")