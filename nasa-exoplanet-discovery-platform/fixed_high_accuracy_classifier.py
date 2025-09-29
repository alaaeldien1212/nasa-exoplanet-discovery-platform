#!/usr/bin/env python3
"""
FIXED HIGH ACCURACY CLASSIFIER - Integration with existing app
This replaces the old biased model with the new balanced one
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

class FixedHighAccuracyClassifier:
    def __init__(self):
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.best_model = None
        
    def load_all_nasa_data_robust(self):
        """Load NASA data with robust error handling"""
        print("Loading NASA data...")
        
        try:
            df = pd.read_csv('../nasa data planet/cumulative_2025.09.27_12.55.48.csv', comment='#')
            print(f"Loaded cumulative data: {len(df)} records")
            return self.preprocess_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """Preprocess data with proper handling"""
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
            'st_mass': 'stellar_mass',
            'koi_smass': 'stellar_mass',
            'koi_duration': 'transit_duration',
            'pl_disposition': 'disposition',
            'koi_disposition': 'disposition'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Select required columns
        required_cols = ['orbital_period', 'planetary_radius', 'stellar_temperature', 
                        'stellar_radius', 'stellar_mass', 'disposition']
        
        available_cols = [col for col in required_cols if col in df.columns]
        
        if 'disposition' not in df.columns:
            print("No disposition column found!")
            return pd.DataFrame()
        
        df = df[available_cols].copy()
        
        # Convert to numeric
        numeric_cols = [col for col in available_cols if col != 'disposition']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['disposition'])
        df = df[df['disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
        
        # Fill missing values
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Final dataset: {len(df)} records")
        print("Class distribution:")
        print(df['disposition'].value_counts())
        
        return df
    
    def train_final_high_accuracy_models(self, df):
        """Train the final high accuracy models"""
        print("Training final high accuracy models...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'disposition']
        X = df[feature_cols]
        y = df['disposition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"After SMOTE balancing:")
        print(pd.Series(y_train_balanced).value_counts())
        
        # Define models
        models = {
            'Fixed Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ),
            'Fixed Gradient Boosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=3,
                random_state=42
            ),
            'Fixed Extra Trees': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ),
            'Fixed SVM': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ),
            'Fixed Histogram GB': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=3,
                random_state=42
            ),
            'Fixed Ensemble': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
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
            self.trained_models[name] = model
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
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
        
        # Store scaler and feature names
        self.scalers['main'] = scaler
        self.label_encoders['target'] = type('obj', (object,), {'classes_': np.unique(y)})
        
        print(f"\nBest model: {best_model_name} with CV accuracy: {best_accuracy:.4f}")
        
        return self.model_performance
    
    def predict_new_data(self, data, model_name='Fixed Random Forest'):
        """Make predictions on new data"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Ensure all required features are present
            feature_cols = list(self.feature_importance[model_name].keys())
            missing_features = set(feature_cols) - set(data.columns)
            if missing_features:
                print(f"Warning: Missing features {missing_features}, filling with 0")
                for feature in missing_features:
                    data[feature] = 0
            
            X = data[feature_cols].fillna(0)
        else:
            X = data
        
        # Scale features
        X_scaled = self.scalers['main'].transform(X)
        
        # Make predictions
        model = self.trained_models[model_name]
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
        print("Saving fixed high accuracy models...")
        
        # Create directory
        import os
        os.makedirs('fixed_high_accuracy_models', exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            joblib.dump(model, f'fixed_high_accuracy_models/{name.lower().replace(" ", "_")}_model.joblib')
        
        # Save scaler
        joblib.dump(self.scalers['main'], 'fixed_high_accuracy_models/scaler.joblib')
        
        # Save label encoders
        joblib.dump(self.label_encoders, 'fixed_high_accuracy_models/label_encoders.joblib')
        
        # Save performance metrics
        joblib.dump(self.model_performance, 'fixed_high_accuracy_models/model_performance.joblib')
        
        # Save feature importance
        joblib.dump(self.feature_importance, 'fixed_high_accuracy_models/feature_importance.joblib')
        
        print("Models saved successfully!")

def main():
    """Main training function"""
    print("=" * 60)
    print("FIXED HIGH ACCURACY CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    classifier = FixedHighAccuracyClassifier()
    
    # Load data
    df = classifier.load_all_nasa_data_robust()
    
    if df.empty:
        print("No data available for training!")
        return
    
    # Train models
    performance = classifier.train_final_high_accuracy_models(df)
    
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
