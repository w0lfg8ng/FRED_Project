import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
import os

class EconomicModel:
    """Economic prediction model using machine learning."""
    
    def __init__(self, config):
        """Initialize model with configuration."""
        self.config = config
        self.models_dir = os.path.join('models', 'saved')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Economic condition thresholds
        self.thresholds = {
            'overheated': {
                'cpi_growth': 3.5,      # Inflation threshold
                'unemployment': 4.0,     # Low unemployment threshold
                'gdp_growth': 3.0       # High growth threshold
            },
            'depressed': {
                'gdp_growth': 1.0,      # Low growth threshold
                'unemployment': 6.0,     # High unemployment threshold
                'financial_conditions': 0.5  # Tight financial conditions threshold
            }
        }
        
        self.trained_models = {}
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for model training."""
        features = pd.DataFrame()
        
        # Year-over-year changes
        for col in data.columns:
            features[f'{col}_yoy'] = data[col].pct_change(periods=12)
        
        # Moving averages (3-month and 12-month)
        for col in data.columns:
            features[f'{col}_ma3'] = data[col].rolling(window=3).mean()
            features[f'{col}_ma12'] = data[col].rolling(window=12).mean()
        
        # Volatility (12-month rolling standard deviation)
        for col in data.columns:
            features[f'{col}_vol'] = data[col].rolling(window=12).std()
        
        # Trend indicators
        for col in data.columns:
            # Rate of change of moving averages
            features[f'{col}_trend'] = features[f'{col}_ma3'] - features[f'{col}_ma12']
            
            # Acceleration
            features[f'{col}_accel'] = features[f'{col}_yoy'].diff()
        
        return features
    
    def classify_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify current economic conditions."""
        conditions = pd.DataFrame(index=data.index)
        
        # Calculate year-over-year growth rates
        cpi_growth = data['cpi'].pct_change(periods=12) * 100
        gdp_growth = data['gdp_growth'].pct_change(periods=12) * 100
        
        # Classify conditions
        conditions['overheated'] = (
            (cpi_growth > self.thresholds['overheated']['cpi_growth']) &
            (data['unemployment'] < self.thresholds['overheated']['unemployment']) &
            (gdp_growth > self.thresholds['overheated']['gdp_growth'])
        )
        
        conditions['depressed'] = (
            (gdp_growth < self.thresholds['depressed']['gdp_growth']) |
            (data['unemployment'] > self.thresholds['depressed']['unemployment']) |
            (data['financial_conditions'] > self.thresholds['depressed']['financial_conditions'])
        )
        
        conditions['goldilocks'] = ~(conditions['overheated'] | conditions['depressed'])
        
        return conditions
    
    def train(self, features: pd.DataFrame, conditions: pd.DataFrame) -> Dict:
        """Train prediction models for different time horizons."""
        model_params = self.config.get_model_parameters()
        forecast_horizons = model_params['forecast_horizons']
        
        # Prepare features
        X = features.fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for months in forecast_horizons:
            print(f"Training models for {months}-month horizon...")
            
            # Prepare target variables
            y_overheated = conditions['overheated'].shift(-months).fillna(False)
            y_depressed = conditions['depressed'].shift(-months).fillna(False)
            
            # Train models
            model_overheated = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            model_depressed = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            # Fit models
            valid_idx = ~y_overheated.isna()
            model_overheated.fit(X_scaled[valid_idx], y_overheated[valid_idx])
            model_depressed.fit(X_scaled[valid_idx], y_depressed[valid_idx])
            
            # Store models
            self.trained_models[months] = {
                'overheated': model_overheated,
                'depressed': model_depressed,
                'scaler': scaler,
                'feature_cols': features.columns.tolist()
            }
            
            # Save models to disk
            self._save_models(months)
        
        return self.trained_models
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate predictions for all time horizons."""
        predictions = {}
        
        if not self.trained_models:
            self._load_models()
        
        for months, model_dict in self.trained_models.items():
            # Prepare features
            X = features[model_dict['feature_cols']].fillna(0)
            X_scaled = model_dict['scaler'].transform(X)
            
            # Get probabilities
            prob_overheated = model_dict['overheated'].predict_proba(X_scaled)[-1][1]
            prob_depressed = model_dict['depressed'].predict_proba(X_scaled)[-1][1]
            prob_goldilocks = 1 - max(prob_overheated, prob_depressed)
            
            # Calculate confidence score
            feature_imp_oh = model_dict['overheated'].feature_importances_
            feature_imp_dep = model_dict['depressed'].feature_importances_
            confidence = np.mean([feature_imp_oh.max(), feature_imp_dep.max()])
            
            predictions[f'{months}_months'] = {
                'overheated': float(prob_overheated),
                'depressed': float(prob_depressed),
                'goldilocks': float(prob_goldilocks),
                'confidence': float(confidence)
            }
        
        return predictions
    
    def _save_models(self, horizon: int) -> None:
        """Save trained models to disk."""
        model_path = os.path.join(self.models_dir, f'models_{horizon}months.joblib')
        joblib.dump(self.trained_models[horizon], model_path)
    
    def _load_models(self) -> None:
        """Load all saved models from disk."""
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                horizon = int(filename.split('_')[1].split('months')[0])
                model_path = os.path.join(self.models_dir, filename)
                self.trained_models[horizon] = joblib.load(model_path)