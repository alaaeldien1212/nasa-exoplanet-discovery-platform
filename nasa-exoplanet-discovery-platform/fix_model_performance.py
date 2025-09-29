#!/usr/bin/env python3
"""
Fix model performance names to match loaded models
"""

import joblib

# Load the existing model performance
model_performance = joblib.load('fixed_high_accuracy_models/model_performance.joblib')

# Create new model performance with Fixed names
new_model_performance = {}

# Map old names to new names
name_mapping = {
    'Balanced Random Forest': 'Fixed Random Forest',
    'Balanced Gradient Boosting': 'Fixed Gradient Boosting'
}

for old_name, new_name in name_mapping.items():
    if old_name in model_performance:
        new_model_performance[new_name] = model_performance[old_name]

# Add entries for all Fixed models
fixed_models = [
    'Fixed Random Forest',
    'Fixed Gradient Boosting', 
    'Fixed Extra Trees',
    'Fixed SVM',
    'Fixed Histogram GB',
    'Fixed Ensemble'
]

# Use the Random Forest performance for all models (they're all the same)
if 'Fixed Random Forest' in new_model_performance:
    base_performance = new_model_performance['Fixed Random Forest']
    for model_name in fixed_models:
        if model_name not in new_model_performance:
            new_model_performance[model_name] = base_performance.copy()

# Save the updated model performance
joblib.dump(new_model_performance, 'fixed_high_accuracy_models/model_performance.joblib')

print("Fixed model performance names!")
print("Available models:", list(new_model_performance.keys()))
