# test_basic.py

print("Script starting...")

from config.fred_config import FREDConfig
print("Imported FREDConfig")

try:
    config = FREDConfig()
    print(f"Created config with API key: {config.api_key}")
except Exception as e:
    print(f"Error creating config: {str(e)}")

print("Script finished")

