import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any

class ExoplanetPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing components"""
        model_dir = Path("models")
        
        try:
            # Try to load the best model
            self.model = joblib.load(model_dir / "best_model.pkl")
            self.scaler = joblib.load(model_dir / "scaler.pkl")
            
            with open(model_dir / "feature_names.json", 'r') as f:
                self.feature_names = json.load(f)
            
            with open(model_dir / "model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
                
            # Don't print to stdout when used in API
            if not hasattr(self, '_api_mode'):
                print("Loaded trained model successfully")
            
        except FileNotFoundError:
            if not hasattr(self, '_api_mode'):
                print("Trained model not found, using simple model...")
            self.load_simple_model()
    
    def load_simple_model(self):
        """Load simple model for demonstration"""
        model_dir = Path("models")
        
        try:
            self.model = joblib.load(model_dir / "simple_model.pkl")
            
            with open(model_dir / "feature_names.json", 'r') as f:
                self.feature_names = json.load(f)
            
            with open(model_dir / "model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
                
            if not hasattr(self, '_api_mode'):
                print("Loaded simple model successfully")
            
        except FileNotFoundError:
            if not hasattr(self, '_api_mode'):
                print("No model found. Please train a model first.")
            self.model = None
    
    def preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for prediction"""
        if not self.feature_names:
            raise ValueError("Model not loaded properly")
        
        # Create feature vector
        features = []
        for feature_name in self.feature_names:
            if feature_name in data:
                value = data[feature_name]
                # Handle missing values
                if pd.isna(value) or value is None:
                    value = 0.0
                features.append(float(value))
            else:
                # Default values for missing features
                if 'mission_' in feature_name:
                    features.append(0.0)
                else:
                    features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data"""
        if not self.model:
            return {
                "error": "Model not loaded",
                "prediction": None,
                "confidence": None
            }
        
        try:
            # Preprocess input
            features = self.preprocess_input(data)
            
            # Scale features if scaler is available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.5
            
            # Determine classification
            classification = "exoplanet" if prediction == 1 else "false positive"
            
            return {
                "prediction": int(prediction),
                "classification": classification,
                "confidence": confidence,
                "features_used": self.feature_names,
                "model_info": self.metadata
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "prediction": None,
                "confidence": None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "metadata": self.metadata,
            "feature_names": self.feature_names,
            "model_loaded": self.model is not None
        }

# Global predictor instance
predictor = ExoplanetPredictor()
