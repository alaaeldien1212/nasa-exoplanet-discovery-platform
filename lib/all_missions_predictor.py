
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

class AllMissionsPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = joblib.load('models/best_model_all_missions.pkl')
            self.scaler = joblib.load('models/scaler_all_missions.pkl')
            
            with open('models/feature_names_all_missions.json', 'r') as f:
                self.feature_names = json.load(f)
            
            with open('models/model_metadata_all_missions.json', 'r') as f:
                self.metadata = json.load(f)
                
            print('All-missions model loaded successfully!')
        except Exception as e:
            print(f'Error loading all-missions model: {e}')
    
    def predict(self, data):
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Convert input data to DataFrame
            if isinstance(data, dict):
                # Handle mission encoding
                mission = data.get('mission', 'Kepler')
                data['mission_Kepler'] = 1 if mission == 'Kepler' else 0
                data['mission_K2'] = 1 if mission == 'K2' else 0
                data['mission_TESS'] = 1 if mission == 'TESS' else 0
                
                df = pd.DataFrame([data])
            else:
                df = data
            
            # Select features in correct order
            features = df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Calculate confidence
            confidence = max(probability)
            
            # Determine classification
            classification = 'exoplanet' if prediction == 1 else 'false positive'
            
            return {
                'classification': classification,
                'confidence': confidence,
                'prediction': prediction,
                'probability_exoplanet': probability[1],
                'probability_false_positive': probability[0],
                'model_info': {
                    'type': 'All Missions Model',
                    'missions': self.metadata.get('missions_used', []),
                    'total_samples': self.metadata.get('total_samples', 0),
                    'accuracy': self.metadata.get('best_accuracy', 0)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}

# Create global instance
all_missions_predictor = AllMissionsPredictor()
