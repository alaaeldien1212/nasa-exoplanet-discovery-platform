#!/usr/bin/env python3
"""
FIXED ACCURATE CLASSIFIER - Simple and robust version
This fixes all the data handling issues and creates a properly balanced model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FixedAccurateClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.model_performance = {}
        self.best_model = None
        
    def load_data_simple(self):
        """Load data with simple, robust approach"""
        print("Loading NASA data with simple approach...")
        
        # Load just the cumulative data first (most reliable)
        try:
            df = pd.read_csv('../nasa data planet/cumulative_2025.09.27_12.55.48.csv', comment='#')
            print(f"Loaded cumulative data: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print("Cleaning data...")
        
        # Reset index to avoid duplicate issues
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
            'st_mass': 'stellar_mass',
            'koi_smass': 'stellar_mass',
            'sy_dist': 'distance',
            'st_dist': 'distance',
            'koi_duration': 'transit_duration',
            'pl_disposition': 'disposition',
            'koi_disposition': 'disposition'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select only the columns we need
        required_cols = ['orbital_period', 'planetary_radius', 'stellar_temperature', 
                        'stellar_radius', 'stellar_mass', 'disposition']
        
        # Check which columns exist
        available_cols = [col for col in required_cols if col in df.columns]
        print(f"Available columns: {available_cols}")
        
        if 'disposition' not in df.columns:
            print("No disposition column found!")
            return pd.DataFrame()
        
        # Select only available columns
        df = df[available_cols].copy()
        
        # Convert to numeric, handling errors
        numeric_cols = [col for col in available_cols if col != 'disposition']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing disposition
        df = df.dropna(subset=['disposition'])
        
        # Filter for valid dispositions
        valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        df = df[df['disposition'].isin(valid_dispositions)]
        
        # Fill missing numeric values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Final dataset: {len(df)} records")
        print("Class distribution:")
        print(df['disposition'].value_counts())
        
        return df
    
    def train_models(self, df):
        """Train balanced models"""
        print("Training balanced models...")
        
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
        
        # Apply SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"After SMOTE balancing:")
        print(pd.Series(y_train_balanced).value_counts())
        
        # Define models with class balancing
        models = {
            'Balanced Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ),
            'Balanced Gradient Boosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=3,
                random_state=42
            )
        }
        
        best_accuracy = 0
        best_model_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            print(f"{name} CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
            
            # Store model and performance
            self.models[name] = model
            self.model_performance[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Track best model
            if cv_mean > best_accuracy:
                best_accuracy = cv_mean
                best_model_name = name
                self.best_model = model
        
        print(f"\nBest model: {best_model_name} with CV accuracy: {best_accuracy:.4f}")
        
        # Print detailed results for best model
        if best_model_name:
            print(f"\nDetailed results for {best_model_name}:")
            print(self.model_performance[best_model_name]['classification_report'])
            print("\nConfusion Matrix:")
            print(self.model_performance[best_model_name]['confusion_matrix'])
        
        return self.model_performance
    
    def predict_new_data(self, data, model_name=None):
        """Make predictions on new data"""
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
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
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence
        }
    
    def save_models(self):
        """Save all trained models and components"""
        print("Saving fixed models...")
        
        # Create directory
        import os
        os.makedirs('fixed_accurate_models', exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'fixed_accurate_models/{name.lower().replace(" ", "_")}_model.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, 'fixed_accurate_models/scaler.joblib')
        
        # Save feature names
        joblib.dump(self.feature_names, 'fixed_accurate_models/feature_names.joblib')
        
        # Save performance metrics
        joblib.dump(self.model_performance, 'fixed_accurate_models/model_performance.joblib')
        
        print("Models saved successfully!")

def main():
    """Main training function"""
    print("=" * 60)
    print("FIXED ACCURATE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    classifier = FixedAccurateClassifier()
    
    # Load data
    df = classifier.load_data_simple()
    
    if df.empty:
        print("No data available for training!")
        return
    
    # Clean data
    df = classifier.clean_data(df)
    
    if df.empty:
        print("No valid data after cleaning!")
        return
    
    # Train models
    performance = classifier.train_models(df)
    
    # Save models
    classifier.save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Print summary
    for name, metrics in performance.items():
        print(f"{name}: {metrics['accuracy']:.4f} accuracy, {metrics['cv_mean']:.4f} CV")

if __name__ == "__main__":
    main()
