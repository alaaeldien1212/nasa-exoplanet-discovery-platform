
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

class MissionAgnosticPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.feature_mapping = {
            'koi_period': 'orbital_period',
            'koi_duration': 'transit_duration', 
            'koi_depth': 'transit_depth',
            'koi_prad': 'planetary_radius',
            'koi_teq': 'equilibrium_temperature',
            'koi_insol': 'insolation',
            'koi_model_snr': 'signal_to_noise_ratio',
            'koi_steff': 'stellar_temperature',
            'koi_slogg': 'stellar_surface_gravity',
            'koi_srad': 'stellar_radius',
            'koi_kepmag': 'stellar_magnitude'
        }
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = joblib.load('models/best_model_mission_agnostic.pkl')
            self.scaler = joblib.load('models/scaler_mission_agnostic.pkl')
            
            with open('models/feature_names_mission_agnostic.json', 'r') as f:
                self.feature_names = json.load(f)
            
            with open('models/model_metadata_mission_agnostic.json', 'r') as f:
                self.metadata = json.load(f)
                
            print('Mission-agnostic model loaded successfully!')
        except Exception as e:
            print(f'Error loading mission-agnostic model: {e}')
    
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
                
                # Map old feature names to new ones
                feature_mapping = {
                    'koi_period': 'orbital_period',
                    'koi_duration': 'transit_duration', 
                    'koi_depth': 'transit_depth',
                    'koi_prad': 'planetary_radius',
                    'koi_teq': 'equilibrium_temperature',
                    'koi_insol': 'insolation',
                    'koi_model_snr': 'signal_to_noise_ratio',
                    'koi_steff': 'stellar_temperature',
                    'koi_slogg': 'stellar_surface_gravity',
                    'koi_srad': 'stellar_radius',
                    'koi_kepmag': 'stellar_magnitude'
                }
                
                # Rename features if they exist
                for old_name, new_name in feature_mapping.items():
                    if old_name in data:
                        data[new_name] = data[old_name]
                
                df = pd.DataFrame([data])
            else:
                df = data
            
            # Add derived features
            if 'orbital_period' in df.columns:
                df['period_log'] = np.log10(df['orbital_period'])
            if 'planetary_radius' in df.columns:
                df['radius_log'] = np.log10(df['planetary_radius'])
            if 'equilibrium_temperature' in df.columns:
                df['temperature_log'] = np.log10(df['equilibrium_temperature'])
            
            # Ensure all required features are present with default values
            for feature_name in self.feature_names:
                if feature_name not in df.columns:
                    if feature_name.startswith('mission_'):
                        df[feature_name] = 0
                    elif 'log' in feature_name:
                        df[feature_name] = 0
                    else:
                        df[feature_name] = 0
            
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
                    'type': 'Mission-Agnostic Model',
                    'version': '4.0',
                    'missions': self.metadata.get('missions_used', []),
                    'total_samples': self.metadata.get('total_samples', 0),
                    'accuracy': self.metadata.get('best_accuracy', 0),
                    'features': self.metadata.get('features_used', [])
                }
            }
            
        except Exception as e:
            return {'error': str(e)}

# Create global instance
mission_agnostic_predictor = MissionAgnosticPredictor()
