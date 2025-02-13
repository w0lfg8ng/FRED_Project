import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Create models/saved directory if it doesn't exist
os.makedirs('models/saved', exist_ok=True)

print("Script starting...")

try:
    from config.fred_config import FREDConfig
    from src.data_collection.fred_collector import FREDDataCollector
    from src.models.state_classifier import MLStateClassifier
    print("Successfully imported required modules")
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)

def test_state_classifier():
    print("\n1. Initializing components...")
    config = FREDConfig()
    collector = FREDDataCollector(config)
    classifier = MLStateClassifier()
    print("Components initialized successfully")

    print("\n2. Collecting and preparing data...")
    try:
        data = collector.prepare_model_data()
        print(f"Data collected. Shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"Error collecting data: {str(e)}")
        return

    print("\n3. Training ML models...")
    try:
        metrics = classifier.train(data)
        print("\nTraining metrics:")
        print(f"Random Forest Accuracy: {metrics['rf_accuracy']:.3f}")
        print(f"Gradient Boosting RMSE: {metrics['gb_rmse']:.3f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    print("\n4. Testing predictions...")
    try:
        # Get prediction for most recent data
        state, score, contributions = classifier.predict_state(data)
        
        print("\nCurrent Economic State:")
        print(f"State: {state}")
        print(f"Score: {score:.3f}")
        
        print("\nTop Feature Contributions:")
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for feature, contribution in sorted_contributions[:5]:
            print(f"{feature}: {contribution:.3f}")
            
        # Save models for future use
        print("\n5. Testing model persistence...")
        classifier.save_models('models/saved/state_classifier.joblib')
        print("Models saved successfully")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return

if __name__ == "__main__":
    test_state_classifier()
    print("\nTesting completed!")