import os
from typing import Dict, List, Optional
import yaml
from dotenv import load_dotenv

class FREDConfig:
    """Configuration manager for FRED API and economic indicators."""
    
    # Key economic indicators with their FRED series IDs
    DEFAULT_INDICATORS = {
        'gdp_growth': 'GDPC1',  # Real GDP
        'unemployment': 'UNRATE',  # Unemployment Rate
        'cpi': 'CPIAUCSL',  # Consumer Price Index
        'treasury_spread': 'T10Y2Y',  # 10-Year Treasury Constant Maturity Minus 2-Year
        'industrial_prod': 'INDPRO',  # Industrial Production Index
        'money_supply': 'M2SL',  # M2 Money Supply
        'financial_conditions': 'NFCI',  # Chicago Fed National Financial Conditions Index
        'housing_starts': 'HOUST',  # Housing Starts
        'vehicle_sales': 'TOTALSA',  # Total Vehicle Sales
        'retail_sales': 'RSAFS'  # Retail Sales
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from environment and YAML file."""
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not found in environment variables")
            
        self.config_data = self._load_config(config_path)
        self.indicators = self.config_data.get('indicators', self.DEFAULT_INDICATORS)
        self.update_frequency = self.config_data.get('update_frequency', 'monthly')
        self.cache_duration = self.config_data.get('cache_duration', 86400)  # Default 24 hours

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        if not config_path:
            config_path = os.path.join('config', 'fred_config.yaml')
            
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default values.")
            return {}

    def get_indicator_series(self) -> Dict[str, str]:
        """Get the configured economic indicators and their series IDs."""
        return self.indicators

    def get_model_parameters(self) -> Dict:
        """Get model-specific parameters from config."""
        return self.config_data.get('model_parameters', {
            'forecast_horizons': [1, 3, 6, 12, 24, 60],  # months
            'training_window': 120,  # months
            'feature_window': 12,  # months for rolling features
        })

    @staticmethod
    def generate_example_config() -> None:
        """Generate example configuration file."""
        example_config = {
            'indicators': FREDConfig.DEFAULT_INDICATORS,
            'update_frequency': 'monthly',
            'cache_duration': 86400,
            'model_parameters': {
                'forecast_horizons': [1, 3, 6, 12, 24, 60],
                'training_window': 120,
                'feature_window': 12,
            }
        }
        with open('config/fred_config.yaml.example', 'w') as file:
            yaml.dump(example_config, file, default_flow_style=False)