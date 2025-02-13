import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, 
    TimeSeriesSplit, 
    cross_val_score,
    GridSearchCV,
    ParameterGrid
)
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.impute import SimpleImputer
import joblib
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from imblearn.over_sampling import SMOTE
from collections import Counter
from data_collection.fred_collector import FREDCollector
from config.fred_config import FREDConfig as Config


class EconomicState(Enum):
    DEPRESSED = -1
    NORMAL = 0
    OVERHEATED = 1

class MLStateClassifier:
    """ML-based economic state classifier using ensemble methods with enhanced features."""
    
    def __init__(self):
        """Initialize with default parameters that will be tuned."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Keep your existing thresholds
        self.thresholds = {
            'overheated': {
                'gdp_growth': 3.5,
                'unemployment': 3.5,
                'inflation': 2.5,
                'fed_rate': 3.0,
                'yield_spread': 2.0,
            },
            'depressed': {
                'gdp_growth': 1.0,
                'unemployment': 6.0,
                'inflation': 1.0,
                'fed_rate': 1.0,
                'yield_spread': 0.0,
            }
        }   
    
        # Initialize models with broader parameter search space
        # In __init__ method, update RF classifier initialization
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42
        )
        self.gb_regressor = GradientBoostingRegressor(random_state=42)
    
        # Initialize scalers and feature selection
        # Replace StandardScaler with RobustScaler in __init__
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
        self.feature_importances = {}
        self.selected_features = []
        self.is_trained = False
        self.feature_names = []

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare exact features needed for the model."""
        try:
            # Create new DataFrame with original index
            features = pd.DataFrame(index=data.index)
            
            # Define exact feature names in the order expected by the model
            feature_order = [
                'Real GDP_yoy',
                'Unemployment Rate_raw',
                'Unemployment Rate_yoy',
                'Consumer Price Index_yoy',
                'Federal Funds Rate_yoy',
                'Industrial Production_raw',
                'Industrial Production_yoy',
                'Financial Conditions_6m',
                'Housing Starts_6m',
                'Vehicle Sales_yoy',
                'Retail Sales_yoy',
                'growth_inflation'
            ]
            
            # Define mapping from data columns to feature names
            column_mapping = {
                'Real GDP_yoy': 'Real GDP_yoy_change',
                'Unemployment Rate_raw': 'Unemployment Rate',
                'Unemployment Rate_yoy': 'Unemployment Rate_yoy_change',
                'Consumer Price Index_yoy': 'Consumer Price Index_yoy_change',
                'Federal Funds Rate_yoy': 'Federal Funds Rate_yoy_change',
                'Industrial Production_raw': 'Industrial Production',
                'Industrial Production_yoy': 'Industrial Production_yoy_change',
                'Financial Conditions_6m': 'Financial Conditions_6m_avg',
                'Housing Starts_6m': 'Housing Starts_6m_avg',
                'Vehicle Sales_yoy': 'Vehicle Sales_yoy_change',
                'Retail Sales_yoy': 'Retail Sales_yoy_change'
            }
            
            # Check if all required columns are present
            missing_columns = [col for col in column_mapping.values()
                             if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Map the columns in the correct order
            for feature_name in feature_order[:-1]:  # Exclude growth_inflation
                features[feature_name] = data[column_mapping[feature_name]]
            
            # Add computed feature last
            features['growth_inflation'] = (
                features['Real GDP_yoy'] *
                features['Consumer Price Index_yoy']
            ).rank(pct=True)
            
            # Ensure columns are in exact order
            return features[feature_order]
        
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            raise

    def _tune_parameters(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Tune model parameters using grid search."""
        print("\nTuning Random Forest parameters...")
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        print(f"Total RF combinations to try: {len(list(ParameterGrid(rf_param_grid)))}")
        rf_grid = GridSearchCV(
            self.rf_classifier,
            rf_param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=2  # Add verbosity
        )
        rf_grid.fit(X, y)

        print("\nTuning Gradient Boosting parameters...")
        gb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [2, 5, 10]
        }
        
        print(f"Total GB combinations to try: {len(list(ParameterGrid(gb_param_grid)))}")
        gb_grid = GridSearchCV(
            self.gb_regressor,
            gb_param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2  # Add verbosity
        )
        gb_grid.fit(X, y.astype(float))

        # Update models with best parameters
        self.rf_classifier = rf_grid.best_estimator_
        self.gb_regressor = gb_grid.best_estimator_

        print("\nBest Random Forest parameters:", rf_grid.best_params_)
        print("Best Gradient Boosting parameters:", gb_grid.best_params_)

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features using RFECV."""
        # Initialize RFECV with RandomForest
        selector = RFECV(
            estimator=self.rf_classifier,
            step=1,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='balanced_accuracy',
            min_features_to_select=5
        )
        
        # Fit selector
        selector.fit(X, y)
        
        # Get selected features
        self.selected_features = X.columns[selector.support_].tolist()
        
        print("Selected feature names:", self.selected_features)
        print("Selected features shape:", X[self.selected_features].shape)
        print("\nSelected features:", len(self.selected_features))
        print("Top 5 features:", self.selected_features[:5])
        
        return X[self.selected_features]

    def _balance_dataset(self, X: pd.DataFrame, y: pd.Series):
        """Balance dataset using SMOTE."""
        print("\nOriginal class distribution:", Counter(y))
        
        # Initialize SMOTE
        smote = SMOTE(random_state=42)
        
        # Fit and apply SMOTE
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print("Balanced class distribution:", Counter(y_balanced))
        
        return X_balanced, y_balanced

    def _add_early_stopping(self, estimator, X: pd.DataFrame, y: pd.Series) -> None:
        """Implement early stopping using built-in sklearn parameters."""
        # Create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep time order
        )
        
        # Configure early stopping parameters
        estimator_params = {
            'warm_start': True,
            'n_iter_no_change': 5,
            'validation_fraction': 0.2,
            'tol': 1e-3
        }
        
        # Only set supported parameters
        valid_params = estimator.get_params().keys()
        supported_params = {k: v for k, v in estimator_params.items() if k in valid_params}
        estimator.set_params(**supported_params)
        
        # Fit the model
        return estimator.fit(X_train, y_train)

    def _analyze_validation_curve(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Analyze validation curve to find optimal parameters."""
        param_range = np.linspace(0.1, 1.0, 5)
        train_scores, val_scores = validation_curve(
            self.rf_classifier, X, y,
            param_name="max_features",
            param_range=param_range,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='balanced_accuracy'
        )
        
        return {
            'param_range': param_range,
            'train_scores': train_scores,
            'val_scores': val_scores
        }

    def _plot_validation_curves(
        self, 
        validation_results: Dict[str, np.ndarray],
        save_path: str = 'visualizations'
    ) -> None:
        """Plot validation curves and training metrics."""
        # Create visualizations directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set plotting style
        plt.style.use('seaborn')
        
        # Plot 1: Validation Curve
        plt.figure(figsize=(12, 6))
        param_range = validation_results['param_range']
        train_scores_mean = np.mean(validation_results['train_scores'], axis=1)
        train_scores_std = np.std(validation_results['train_scores'], axis=1)
        val_scores_mean = np.mean(validation_results['val_scores'], axis=1)
        val_scores_std = np.std(validation_results['val_scores'], axis=1)
        
        plt.plot(param_range, train_scores_mean, label='Training Score', color='blue')
        plt.fill_between(param_range, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color='blue')
        plt.plot(param_range, val_scores_mean, label='Validation Score', color='red')
        plt.fill_between(param_range, 
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Max Features Parameter')
        plt.ylabel('Balanced Accuracy Score')
        plt.title('Validation Curve - Random Forest')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{save_path}/validation_curve_{timestamp}.png')
        plt.close()
        
        # Plot 2: Feature Importance
        if self.feature_importances:
            plt.figure(figsize=(12, 6))
            importances = pd.Series(self.feature_importances).sort_values(ascending=True)
            importances.plot(kind='barh')
            plt.title('Feature Importance')
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f'{save_path}/feature_importance_{timestamp}.png')
            plt.close()

    def _plot_prediction_analysis(
        self, 
        scores: np.ndarray,
        save_path: str = 'visualizations'
    ) -> None:
        """Plot prediction analysis."""
        # Create visualizations directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(scores, bins=30, kde=True)
        plt.title('Distribution of Prediction Scores')
        plt.xlabel('Score')
        plt.ylabel('Count')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{save_path}/prediction_distribution_{timestamp}.png')
        plt.close()

    def generate_labels(self, data: pd.DataFrame) -> pd.Series:
        """Enhanced label generation incorporating all major indicators."""
        conditions = []
        weights = {
            'gdp': 0.25,
            'unemployment': 0.20,
            'inflation': 0.15,
            'financial': 0.15,  # New weight for financial conditions
            'housing': 0.15,    # New weight for housing/consumption indicators
            'industrial': 0.10  # New weight for industrial/production indicators
        }
        
        for _, row in data.iterrows():
            # Core economic indicators
            gdp_score = 1 if row['Real GDP_yoy_change'] > self.thresholds['overheated']['gdp_growth'] else \
                       -1 if row['Real GDP_yoy_change'] < self.thresholds['depressed']['gdp_growth'] else 0
            
            unemp_score = -1 if row['Unemployment Rate'] < self.thresholds['overheated']['unemployment'] else \
                          1 if row['Unemployment Rate'] > self.thresholds['depressed']['unemployment'] else 0
            
            infl_score = 1 if row['Consumer Price Index_yoy_change'] > self.thresholds['overheated']['inflation'] else \
                       -1 if row['Consumer Price Index_yoy_change'] < self.thresholds['depressed']['inflation'] else 0
            
            # Financial conditions score
            financial_score = -1 if row['Financial Conditions'] < -0.5 else \
                             1 if row['Financial Conditions'] > 0.5 else 0
            
            # Housing and consumption score (average of housing starts and retail sales)
            housing_retail_score = (
                (1 if row['Housing Starts_yoy_change'] > 15 else -1 if row['Housing Starts_yoy_change'] < -10 else 0) +
                (1 if row['Retail Sales_yoy_change'] > 5 else -1 if row['Retail Sales_yoy_change'] < -3 else 0)
            ) / 2
            
            # Industrial activity score
            industrial_score = 1 if row['Industrial Production_yoy_change'] > 5 else \
                             -1 if row['Industrial Production_yoy_change'] < -3 else 0
            
            # Weighted average of all components
            total_score = (
                weights['gdp'] * gdp_score +
                weights['unemployment'] * unemp_score +
                weights['inflation'] * infl_score +
                weights['financial'] * financial_score +
                weights['housing'] * housing_retail_score +
                weights['industrial'] * industrial_score
            )
            
            # Convert to state with enhanced thresholds
            if total_score > 0.25:
                conditions.append(EconomicState.OVERHEATED.value)
            elif total_score < -0.35:
                conditions.append(EconomicState.DEPRESSED.value)
            else:
                conditions.append(EconomicState.NORMAL.value)
        
        return pd.Series(conditions, index=data.index)
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the model with proper feature name handling."""
        print("\nPreparing features and labels...")
        X = self.prepare_features(data)
        y = self.generate_labels(data)
    
        print("Original feature shape:", X.shape)
        print("Original labels distribution:", y.value_counts())
    
        # Handle NaN values before feature selection
        print("\nHandling missing values...")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
        print("\nSelecting features...")
        X_selected = self._select_features(X_imputed, y)
    
        # Store feature names
        self.feature_names = X_selected.columns.tolist()
    
        # Scale features before SMOTE
        print("\nScaling features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=self.feature_names,
            index=X_selected.index
        )
    
        # Apply SMOTE
        print("\nBalancing dataset using SMOTE...")
        X_balanced, y_balanced = self._balance_dataset(X_scaled, y)
        X_balanced = pd.DataFrame(X_balanced, columns=self.feature_names)
    
        # Split the balanced data
        print("\nSplitting data for validation...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, val_idx in sss.split(X_balanced, y_balanced):
            X_train = X_balanced.iloc[train_idx]
            X_val = X_balanced.iloc[val_idx]
            y_train = pd.Series(y_balanced).iloc[train_idx]
            y_val = pd.Series(y_balanced).iloc[val_idx]
    
        print("Training set class distribution:", Counter(y_train))
        print("Validation set class distribution:", Counter(y_val))
    
        # Parameter tuning with progress
        print("\nTuning model parameters...")
        self._tune_parameters(X_train, y_train)
    
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        self.rf_classifier.set_params(class_weight=class_weight_dict)
    
        # Final training with early stopping
        print("\nTraining final models with early stopping...")
        rf_params = self.rf_classifier.get_params()
        gb_params = self.gb_regressor.get_params()
    
        self.rf_classifier = RandomForestClassifier(**rf_params)
        self.gb_regressor = GradientBoostingRegressor(**gb_params)
    
        self.rf_classifier = self._add_early_stopping(self.rf_classifier, X_train, y_train)
        self.gb_regressor = self._add_early_stopping(self.gb_regressor, X_train.astype(float), y_train.astype(float))
    
        # Store feature names in models
        self.rf_classifier.feature_names_in_ = self.feature_names
        self.gb_regressor.feature_names_in_ = self.feature_names
    
        # Validation analysis
        print("\nAnalyzing validation curves...")
        validation_results = self._analyze_validation_curve(X_train, y_train)
    
        print("\nGenerating validation curve visualizations...")
        self._plot_validation_curves(validation_results)
    
        # Calculate metrics
        print("\nCalculating evaluation metrics...")
        y_pred_rf = self.rf_classifier.predict(X_val)
        y_pred_proba_rf = self.rf_classifier.predict_proba(X_val)
        y_pred_gb = self.gb_regressor.predict(X_val)
    
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba_rf[:, 1])
        except:
            roc_auc = np.nan
    
        metrics = {
            'rf_accuracy': balanced_accuracy_score(y_val, y_pred_rf),
            'rf_precision': precision_score(y_val, y_pred_rf, average='weighted', zero_division=0),
            'rf_recall': recall_score(y_val, y_pred_rf, average='weighted'),
            'rf_f1': f1_score(y_val, y_pred_rf, average='weighted'),
            'rf_roc_auc': roc_auc,
            'gb_rmse': np.sqrt(np.mean((y_val.astype(float) - y_pred_gb) ** 2)),
            'gb_mae': np.mean(np.abs(y_val.astype(float) - y_pred_gb)),
            'confusion_matrix': confusion_matrix(y_val, y_pred_rf).tolist(),
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'class_distribution': pd.Series(y_val).value_counts().to_dict()
        }
    
        # Print metrics
        print("\nDetailed Evaluation Metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, (float, int)):
                print(f"{metric_name}: {value:.3f}")
    
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
    
        print("\nClass Distribution:")
        for state, count in metrics['class_distribution'].items():
            print(f"State {state}: {count} samples")
    
        feature_importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importances:")
        print(feature_importance_df)

        print("\nDetailed Model Performance:")
        print(f"Balanced Accuracy: {metrics['rf_accuracy']:.3f}")
        print(f"Precision: {metrics['rf_precision']:.3f}")
        print(f"Recall: {metrics['rf_recall']:.3f}")
        print(f"F1 Score: {metrics['rf_f1']:.3f}")
        print(f"ROC AUC: {metrics['rf_roc_auc']:.3f}")

        # Store feature importances for later use
        self.feature_importances = dict(zip(self.selected_features, 
                                          self.rf_classifier.feature_importances_))

        self.is_trained = True
        return metrics

    def predict_state(self, data: pd.DataFrame) -> Tuple[EconomicState, float, Dict[str, float]]:
        """Enhanced state prediction with proper feature name handling."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        # Prepare features
        X = self.prepare_features(data.tail(1))  # Use most recent data point
    
        # Handle NaN values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
    
        # Select only the features used during training
        if self.feature_names:
            X_imputed = X_imputed[self.feature_names]
    
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=self.feature_names,
            index=X_imputed.index
        )
    
        # Get predictions from both models with consistent feature names
        rf_pred_proba = self.rf_classifier.predict_proba(X_scaled)[0]
        gb_pred = self.gb_regressor.predict(X_scaled)[0]
    
        # Calculate confidence intervals more efficiently
        n_estimators = len(self.rf_classifier.estimators_)
        subset_size = min(20, n_estimators // 2)  # Use smaller subset
        subset_indices = np.random.choice(n_estimators, subset_size, replace=False)
    
        predictions = []
        for idx in subset_indices:
            pred = self.rf_classifier.estimators_[idx].predict_proba(X_scaled)[0]
            predictions.append(pred[1] - pred[0])
    
        confidence_interval = np.percentile(predictions, [2.5, 97.5])
        print(f"\nPrediction Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
    
        # Enhanced ensemble prediction
        rf_score = rf_pred_proba[1] - rf_pred_proba[0]
        confidence = np.max(rf_pred_proba)
    
        # Weighted average
        final_score = (0.6 * rf_score + 0.4 * gb_pred) * confidence
    
        # Get feature contributions using consistent feature names
        feature_contributions = {}
        for idx, feature in enumerate(self.feature_names):
            contribution = self.rf_classifier.feature_importances_[idx] * X_imputed.iloc[0][feature]
            feature_contributions[feature] = float(contribution)
    
        # Map score to state with interpolation
        state = self._determine_state(final_score, confidence)
    
        return state, final_score, feature_contributions

    def _determine_state(self, score: float, confidence: float) -> EconomicState:
        """Determine economic state with smoother transitions."""
        threshold = 0.3 * confidence
        
        if score > threshold:
            return EconomicState.OVERHEATED
        elif score < -threshold:
            return EconomicState.DEPRESSED
        else:
            return EconomicState.NORMAL

    def predict_state_forward(self, data: pd.DataFrame, months_ahead: int = 3) -> Tuple[EconomicState, float, float]:
        """Predict economic state for future months with uncertainty scaling."""
        base_state, base_score, base_contributions = self.predict_state(data)
    
        # Increase uncertainty with time
        uncertainty_factor = 1.0 + (months_ahead / 12.0) * 0.5  # 50% more uncertainty per year
    
        # Adjust score based on time horizon
        # Add mean reversion tendency
        reversion_strength = months_ahead / 24.0  # Stronger mean reversion for longer horizons
        mean_reversion = -base_score * reversion_strength
    
        # Add time-based uncertainty
        noise = np.random.normal(0, 0.1 * uncertainty_factor)
    
        adjusted_score = base_score + mean_reversion + noise
        adjusted_confidence = max(0.1, base_contributions.get('confidence', 1.0) / uncertainty_factor)
    
        # Determine state with wider thresholds for future predictions
        if adjusted_score > 0.3 * uncertainty_factor:
            state = EconomicState.OVERHEATED
        elif adjusted_score < -0.3 * uncertainty_factor:
            state = EconomicState.DEPRESSED
        else:
            state = EconomicState.NORMAL
    
        return state, adjusted_score, adjusted_confidence
    
    def save_models(self, path: str) -> None:
        """Save models with feature names."""
        try:
            model_data = {
                'rf_classifier': self.rf_classifier,
                'gb_regressor': self.gb_regressor,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_importances': self.feature_importances,
                'is_trained': self.is_trained,
                'thresholds': self.thresholds,
                'selected_features': self.selected_features,
                'feature_names': self.feature_names  # Add feature names to saved data
            }
            joblib.dump(model_data, path)
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise

    def load_models(self, path: str) -> None:
        """Load models with feature names."""
        try:
            model_data = joblib.load(path)
            self.rf_classifier = model_data['rf_classifier']
            self.gb_regressor = model_data['gb_regressor']
            self.scaler = model_data['scaler']
            self.imputer = model_data.get('imputer', SimpleImputer(strategy='mean'))
            self.feature_importances = model_data['feature_importances']
            self.is_trained = model_data['is_trained']
            self.thresholds = model_data['thresholds']
            self.selected_features = model_data.get('selected_features', [])
            self.feature_names = model_data.get('feature_names', [])  # Load feature names

            # Print feature names and their order from the loaded model
            if self.feature_names:
                print("\nModel's expected features in order:")
                for i, feature in enumerate(self.feature_names):
                    print(f"{i+1}. {feature}")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def _load_historical_data(self):
        """Load historical economic state data with batched processing."""
        try:
            # Initialize FRED collector
            config = Config()
            collector = FREDCollector(config)
    
            # Get historical data
            historical_data = collector.get_historical_data()
    
            if historical_data is None or historical_data.empty:
                return []
        
            # Process data through classifier in batches
            results = []
            batch_size = 50  # Process 50 rows at a time
        
            for start_idx in range(0, len(historical_data), batch_size):
                end_idx = min(start_idx + batch_size, len(historical_data))
                batch = historical_data.iloc[start_idx:end_idx]
            
                # Prepare features for entire batch at once
                try:
                    features = self.prepare_features(batch)
                
                    # Skip if features contain NaN values
                    if features.isna().any().any():
                        continue
                
                    # Process each row in the batch
                    for date, row in features.iterrows():
                        try:
                            # Get prediction
                            state, score, _ = self.predict_state(row.to_frame().T)
                            results.append((date, score))
                        
                            # Print progress every 100 records
                            if len(results) % 100 == 0:
                                print(f"Processed {len(results)} records...")
                            
                        except Exception as e:
                            print(f"Warning: Skipping data point {date} due to error: {str(e)}")
                            continue
                        
                except Exception as e:
                    print(f"Warning: Skipping batch due to error: {str(e)}")
                    continue
        
            print(f"Completed processing {len(results)} historical records")
            return sorted(results)  # Sort by date
    
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            return []