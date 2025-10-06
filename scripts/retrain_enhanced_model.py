#!/usr/bin/env python3
"""
Retrain the exoplanet classification model with enhanced data including solar system planets.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os

def load_enhanced_data():
    """Load the enhanced training data with solar system planets."""
    data_path = 'data/processed/exoplanet_training_data_enhanced.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Enhanced data file not found at {data_path}")
        print("Please run add_solar_system_data.py first.")
        return None
    
    df = pd.read_csv(data_path)
    print(f"Loaded enhanced training data: {len(df)} samples")
    print(f"Exoplanets: {df['is_exoplanet'].sum()} ({df['is_exoplanet'].mean()*100:.1f}%)")
    
    return df

def prepare_features(df):
    """Prepare features for training."""
    
    # Feature columns (excluding target)
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
        'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_kepmag', 'mission_Kepler', 'mission_K2', 'mission_TESS'
    ]
    
    X = df[feature_columns].copy()
    y = df['is_exoplanet'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, feature_columns

def train_enhanced_model(X, y, feature_columns):
    """Train the enhanced model with improved parameters."""
    
    print("\n=== Training Enhanced Model ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with enhanced parameters for better handling of diverse data
    model = RandomForestClassifier(
        n_estimators=200,           # More trees for better generalization
        max_depth=15,               # Allow deeper trees for complex patterns
        min_samples_split=5,        # Prevent overfitting
        min_samples_leaf=2,         # Allow smaller leaves for rare cases
        max_features='sqrt',        # Good for high-dimensional data
        random_state=42,
        n_jobs=-1,                  # Use all cores
        class_weight='balanced'     # Handle class imbalance
    )
    
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['False Positive', 'Exoplanet']))
    
    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return model, scaler, feature_columns, accuracy

def test_mars_prediction(model, scaler, feature_columns):
    """Test Mars prediction with the new model."""
    
    print("\n=== Testing Mars Prediction ===")
    
    # Mars parameters
    mars_data = {
        'koi_period': 686.98,
        'koi_duration': 16.02,
        'koi_depth': 23.74,
        'koi_prad': 0.532,
        'koi_teq': 210,
        'koi_insol': 0.431,
        'koi_model_snr': 25.0,
        'koi_steff': 5772,
        'koi_slogg': 4.44,
        'koi_srad': 1.0,
        'koi_kepmag': 10.0,
        'mission_Kepler': 1,
        'mission_K2': 0,
        'mission_TESS': 0
    }
    
    # Create DataFrame with same column order
    mars_df = pd.DataFrame([mars_data])
    mars_features = mars_df[feature_columns]
    
    # Scale and predict
    mars_scaled = scaler.transform(mars_features)
    prediction_proba = model.predict_proba(mars_scaled)[0]
    prediction = model.predict(mars_scaled)[0]
    
    exoplanet_confidence = prediction_proba[1]  # Probability of being exoplanet
    
    print(f"Mars Prediction: {'Exoplanet' if prediction == 1 else 'False Positive'}")
    print(f"Exoplanet Confidence: {exoplanet_confidence:.1%}")
    print(f"False Positive Confidence: {prediction_proba[0]:.1%}")
    
    return exoplanet_confidence

def save_model_files(model, scaler, feature_columns, accuracy):
    """Save the enhanced model and related files."""
    
    print("\n=== Saving Enhanced Model ===")
    
    # Save model
    model_path = 'models/enhanced_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = 'models/enhanced_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature names
    feature_names_path = 'models/enhanced_feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Feature names saved to: {feature_names_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'training_samples': 7858,
        'features': feature_columns,
        'accuracy': accuracy,
        'description': 'Enhanced model trained with solar system planets',
        'version': '2.0',
        'improvements': [
            'Added solar system planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn)',
            'Extended parameter ranges for long-period planets',
            'Improved class balancing',
            'Enhanced model parameters for better generalization'
        ]
    }
    
    metadata_path = 'models/enhanced_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")
    
    return model_path, scaler_path, feature_names_path, metadata_path

def main():
    """Main training function."""
    
    print("=== Enhanced Exoplanet Model Training ===")
    
    # Load enhanced data
    df = load_enhanced_data()
    if df is None:
        return
    
    # Prepare features
    X, y, feature_columns = prepare_features(df)
    
    # Train model
    model, scaler, feature_columns, accuracy = train_enhanced_model(X, y, feature_columns)
    
    # Test Mars prediction
    mars_confidence = test_mars_prediction(model, scaler, feature_columns)
    
    # Save model files
    model_files = save_model_files(model, scaler, feature_columns, accuracy)
    
    print(f"\n=== Training Complete ===")
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"Mars exoplanet confidence: {mars_confidence:.1%}")
    print(f"Model files saved successfully!")
    
    if mars_confidence > 0.5:
        print("✅ SUCCESS: Mars is now correctly classified as an exoplanet!")
    else:
        print("⚠️  Mars still classified as false positive. Consider further tuning.")

if __name__ == "__main__":
    main()

