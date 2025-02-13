# Update inspect_model.py with this content
import joblib
import numpy as np

model = joblib.load('models/saved/state_classifier.joblib')
print("\nModel Keys:")
print(model.keys())

rf_classifier = model['rf_classifier']

print("\nFeature Names from RF Classifier:")
if hasattr(rf_classifier, 'feature_names_in_'):
    print(rf_classifier.feature_names_in_)
else:
    print("No feature_names_in_ found")

print("\nNumber of features used in training:")
print(rf_classifier.n_features_in_)

print("\nFeature importances:")
importances = rf_classifier.feature_importances_
if importances is not None:
    # Sort features by importance
    feature_names = getattr(rf_classifier, 'feature_names_in_', [f"feature_{i}" for i in range(len(importances))])
    sorted_idx = np.argsort(importances)
    for idx in sorted_idx:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")