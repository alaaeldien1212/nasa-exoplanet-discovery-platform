#!/usr/bin/env python3
"""
Enhanced predictor using the retrained model with solar system planets.
"""

import joblib
import json
import pandas as pd
import numpy as np
import os

class EnhancedPredictor:
    """Enhanced exoplanet predictor with improved model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self._api_mode = False
        
        # Try to load enhanced model first, fallback to original
        self._load_model()
    
    def _load_model(self):
        """Load the enhanced model or fallback to original."""
        
        # Paths for enhanced model
        enhanced_model_path = 'models/enhanced_model.pkl'
        enhanced_scaler_path = 'models/enhanced_scaler.pkl'
        enhanced_features_path = 'models/enhanced_feature_names.json'
        enhanced_metadata_path = 'models/enhanced_model_metadata.json'
        
        # Paths for original model (fallback)
        original_model_path = 'models/best_model.pkl'
        original_scaler_path = 'models/scaler.pkl'
        original_features_path = 'models/feature_names.json'
        original_metadata_path = 'models/model_metadata.json'
        
        try:
            # Try to load enhanced model
            if (os.path.exists(enhanced_model_path) and 
                os.path.exists(enhanced_scaler_path) and 
                os.path.exists(enhanced_features_path)):
                
                self.model = joblib.load(enhanced_model_path)
                self.scaler = joblib.load(enhanced_scaler_path)
                
                with open(enhanced_features_path, 'r') as f:
                    self.feature_names = json.load(f)
                
                if os.path.exists(enhanced_metadata_path):
                    with open(enhanced_metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                
                if not self._api_mode:
                    print("✅ Loaded enhanced model with solar system planets")
                    if self.metadata:
                        print(f"   Model version: {self.metadata.get('version', 'unknown')}")
                        print(f"   Training samples: {self.metadata.get('training_samples', 'unknown')}")
                
            else:
                # Fallback to original model
                self.model = joblib.load(original_model_path)
                self.scaler = joblib.load(original_scaler_path)
                
                with open(original_features_path, 'r') as f:
                    self.feature_names = json.load(f)
                
                if os.path.exists(original_metadata_path):
                    with open(original_metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                
                if not self._api_mode:
                    print("⚠️  Using original model (enhanced model not found)")
                    print("   Run retrain_enhanced_model.py to get improved Mars classification")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, planet_data):
        """Predict exoplanet classification."""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert input data to DataFrame
        df = pd.DataFrame([planet_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            # Add missing features with default values
            for feature in missing_features:
                if 'mission_' in feature:
                    df[feature] = 0  # Default to 0 for mission features
                else:
                    df[feature] = 0.0  # Default to 0.0 for other features
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get confidence
        if prediction == 1:
            confidence = prediction_proba[1]
            classification = 'exoplanet'
        else:
            confidence = prediction_proba[0]
            classification = 'false positive'
        
        # Calculate derived parameters for display
        period = planet_data.get('koi_period', 0)
        radius = planet_data.get('koi_prad', 0)
        depth = planet_data.get('koi_depth', 0)
        duration = planet_data.get('koi_duration', 0)
        temperature = planet_data.get('koi_teq', 0)
        insol = planet_data.get('koi_insol', 0)
        snr = planet_data.get('koi_model_snr', 0)
        
        return {
            'classification': classification,
            'confidence': confidence,
            'prediction': prediction,
            'is_exoplanet': bool(prediction),
            'parameters': {
                'period_days': period,
                'radius_earth': radius,
                'transit_depth_ppm': depth,
                'transit_duration_hours': duration,
                'equilibrium_temperature_k': temperature,
                'insolation_earth_units': insol,
                'signal_to_noise_ratio': snr
            },
            'model_info': {
                'version': self.metadata.get('version', 'unknown') if self.metadata else 'unknown',
                'enhanced': 'enhanced_model.pkl' in str(self.model) if hasattr(self, 'model') else False
            }
        }

# Create global instance
enhanced_predictor = EnhancedPredictor()

