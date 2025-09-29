#!/usr/bin/env python3
"""
WORKING EXOPLANET CLASSIFIER - Simple and reliable
This classifier actually works without complex dependencies
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class WorkingExoplanetClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        
    def load_nasa_data(self):
        """Load NASA data with simple approach"""
        print("Loading NASA data...")
        
        try:
            df = pd.read_csv('../nasa data planet/cumulative_2025.09.27_12.55.48.csv', comment='#')
            print(f"Loaded data: {len(df)} records")
            return self.preprocess_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """Preprocess data with robust handling"""
        print("Preprocessing data...")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map column names
        column_mapping = {
            'pl_orbper': 'orbital_period',
            'koi_period': 'orbital_period',
            'pl_rade': 'planetary_radius', 
            'koi_prad': 'planetary_radius',
            'st_teff': 'stellar_temperature',
            'koi_steff': 'stellar_temperature',
            'st_rad': 'stellar_radius',
            'koi_srad': 'stellar_radius',
            'koi_duration': 'transit_duration',
            'pl_disposition': 'disposition',
            'koi_disposition': 'disposition'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Select only the features we need
        required_features = ['orbital_period', 'planetary_radius', 'stellar_temperature', 'stellar_radius']
        
        # Check which features exist
        available_features = [col for col in required_features if col in df.columns]
        print(f"Available features: {available_features}")
        
        if 'disposition' not in df.columns:
            print("No disposition column found!")
            return pd.DataFrame()
        
        # Select data
        df = df[available_features + ['disposition']].copy()
        
        # Convert to numeric
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['disposition'])
        df = df[df['disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
        
        # Fill missing values with median
        for col in available_features:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Final dataset: {len(df)} records")
        print("Class distribution:")
        print(df['disposition'].value_counts())
        
        return df
    
    def train_model(self, df):
        """Train a single robust model"""
        print("Training working model...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'disposition']
        X = df[feature_cols]
        y = df['disposition']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with class balancing
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                print(f"Warning: Missing features {missing_features}, filling with 0")
                for feature in missing_features:
                    data[feature] = 0
            
            X = data[self.feature_names].fillna(0)
        else:
            X = data
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence
        }
    
    def save_model(self):
        """Save the trained model"""
        print("Saving working model...")
        
        import os
        os.makedirs('working_models', exist_ok=True)
        
        joblib.dump(self.model, 'working_models/model.joblib')
        joblib.dump(self.scaler, 'working_models/scaler.joblib')
        joblib.dump(self.feature_names, 'working_models/feature_names.joblib')
        joblib.dump(self.class_names, 'working_models/class_names.joblib')
        
        print("Model saved successfully!")
    
    def load_model(self):
        """Load the trained model"""
        print("Loading working model...")
        
        try:
            self.model = joblib.load('working_models/model.joblib')
            self.scaler = joblib.load('working_models/scaler.joblib')
            self.feature_names = joblib.load('working_models/feature_names.joblib')
            self.class_names = joblib.load('working_models/class_names.joblib')
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main training function"""
    print("=" * 60)
    print("WORKING EXOPLANET CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    classifier = WorkingExoplanetClassifier()
    
    # Load data
    df = classifier.load_nasa_data()
    
    if df.empty:
        print("No data available for training!")
        return
    
    # Train model
    accuracy = classifier.train_model(df)
    
    # Save model
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Final accuracy: {accuracy:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()