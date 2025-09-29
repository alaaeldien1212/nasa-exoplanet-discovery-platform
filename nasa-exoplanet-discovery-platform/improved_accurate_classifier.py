#!/usr/bin/env python3
"""
IMPROVED ACCURATE CLASSIFIER - Fixes the biased model issues
This classifier addresses the major problems:
1. Class imbalance bias toward FALSE_POSITIVE
2. Poor CONFIRMED planet detection
3. Low overall accuracy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class ImprovedAccurateClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.best_model = None
        self.feature_names = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess NASA data with proper handling"""
        print("Loading NASA data...")
        
        # Load all NASA datasets
        datasets = []
        
        # Load cumulative data
        try:
            cumulative_df = pd.read_csv('../nasa data planet/cumulative_2025.09.27_12.55.48.csv', comment='#')
            cumulative_df['source'] = 'cumulative'
            datasets.append(cumulative_df)
            print(f"Loaded cumulative data: {len(cumulative_df)} records")
        except Exception as e:
            print(f"Error loading cumulative data: {e}")
        
        # Load TOI data
        try:
            toi_df = pd.read_csv('../nasa data planet/TOI_2025.09.27_12.56.11.csv', comment='#')
            toi_df['source'] = 'toi'
            datasets.append(toi_df)
            print(f"Loaded TOI data: {len(toi_df)} records")
        except Exception as e:
            print(f"Error loading TOI data: {e}")
        
        # Load K2 data
        try:
            k2_df = pd.read_csv('../nasa data planet/k2pandc_2025.09.27_12.56.23.csv', comment='#')
            k2_df['source'] = 'k2'
            datasets.append(k2_df)
            print(f"Loaded K2 data: {len(k2_df)} records")
        except Exception as e:
            print(f"Error loading K2 data: {e}")
        
        if not datasets:
            print("No data loaded!")
            return pd.DataFrame()
        
        # Combine all datasets
        df = pd.concat(datasets, ignore_index=True)
        print(f"Combined dataset: {len(df)} records")
        
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Preprocess data with improved feature engineering"""
        print("Preprocessing data...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map different column names to standard names
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
            'pl_insol': 'stellar_irradiance',
            'koi_insol': 'stellar_irradiance',
            'pl_eqt': 'equilibrium_temperature',
            'koi_teq': 'equilibrium_temperature',
            'koi_duration': 'transit_duration',
            'pl_disposition': 'disposition',
            'koi_disposition': 'disposition',
            'pl_pdisposition': 'disposition',
            'koi_pdisposition': 'disposition'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Handle missing values more intelligently
        print("Handling missing values...")
        
        # Fill missing values with median for numerical columns
        numerical_cols = ['orbital_period', 'planetary_radius', 'stellar_temperature', 
                         'stellar_radius', 'stellar_mass', 'distance', 'transit_duration',
                         'equilibrium_temperature', 'stellar_irradiance']
        
        for col in numerical_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
                except Exception as e:
                    print(f"Error processing column {col}: {e}")
                    df[col] = 0
        
        # Create additional features for better discrimination
        print("Creating additional features...")
        
        # Habitable zone features
        if 'stellar_temperature' in df.columns and 'stellar_radius' in df.columns:
            try:
                df['habitable_zone_inner'] = 0.75 * (df['stellar_temperature'] / 5778) ** 2
                df['habitable_zone_outer'] = 2.0 * (df['stellar_temperature'] / 5778) ** 2
            except:
                df['habitable_zone_inner'] = 0
                df['habitable_zone_outer'] = 0
        
        # Transit depth features
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            try:
                df['transit_depth'] = (df['planetary_radius'] / df['stellar_radius']) ** 2
            except:
                df['transit_depth'] = 0
        
        # Orbital velocity features
        if 'orbital_period' in df.columns and 'stellar_mass' in df.columns:
            try:
                df['orbital_velocity'] = np.sqrt(df['stellar_mass']) / np.sqrt(df['orbital_period'])
            except:
                df['orbital_velocity'] = 0
        
        # Size ratio features
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            try:
                df['size_ratio'] = df['planetary_radius'] / df['stellar_radius']
            except:
                df['size_ratio'] = 0
        
        # Temperature ratio features
        if 'equilibrium_temperature' in df.columns and 'stellar_temperature' in df.columns:
            try:
                df['temp_ratio'] = df['equilibrium_temperature'] / df['stellar_temperature']
            except:
                df['temp_ratio'] = 0
        
        # Select final features
        feature_cols = [
            'orbital_period', 'planetary_radius', 'stellar_temperature',
            'stellar_radius', 'stellar_mass', 'distance', 'transit_duration',
            'equilibrium_temperature', 'stellar_irradiance',
            'habitable_zone_inner', 'habitable_zone_outer', 'transit_depth',
            'orbital_velocity', 'size_ratio', 'temp_ratio'
        ]
        
        # Only use features that exist
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        # Filter for records with target variable
        if 'disposition' in df.columns:
            df = df.dropna(subset=['disposition'])
            df = df[df['disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
        else:
            print("No disposition column found!")
            return pd.DataFrame()
        
        # Select features and target
        X = df[available_features].fillna(0)
        y = df['disposition']
        
        print(f"Final dataset: {len(X)} records with {len(available_features)} features")
        print(f"Class distribution:")
        print(y.value_counts())
        
        return pd.concat([X, y], axis=1)
    
    def train_improved_models(self, df):
        """Train models with proper class balancing and validation"""
        print("Training improved models with class balancing...")
        
        X = df[self.feature_names]
        y = df['disposition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with better parameters
        models = {
            'Balanced Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'Balanced Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            'Balanced Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'Balanced SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'Balanced Logistic Regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
        
        # Train models with SMOTE for additional balancing
        smote = SMOTE(random_state=42)
        
        best_accuracy = 0
        best_model_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with SMOTE
            if hasattr(model, 'fit'):
                pipeline = ImbPipeline([
                    ('smote', smote),
                    ('classifier', model)
                ])
                
                # Train model
                pipeline.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test_scaled)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"{name} Accuracy: {accuracy:.4f}")
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                print(f"{name} CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
                
                # Store model and performance
                self.models[name] = pipeline
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
                    self.best_model = pipeline
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        self.feature_names, 
                        model.feature_importances_
                    ))
        
        # Store scaler and feature names
        self.scalers['main'] = scaler
        self.feature_names = self.feature_names
        
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
        X_scaled = self.scalers['main'].transform(X)
        
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
        print("Saving improved models...")
        
        # Create directory
        import os
        os.makedirs('improved_accurate_models', exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'improved_accurate_models/{name.lower().replace(" ", "_")}_model.joblib')
        
        # Save scaler
        joblib.dump(self.scalers['main'], 'improved_accurate_models/scaler.joblib')
        
        # Save feature names
        joblib.dump(self.feature_names, 'improved_accurate_models/feature_names.joblib')
        
        # Save performance metrics
        joblib.dump(self.model_performance, 'improved_accurate_models/model_performance.joblib')
        
        # Save feature importance
        joblib.dump(self.feature_importance, 'improved_accurate_models/feature_importance.joblib')
        
        print("Models saved successfully!")

def main():
    """Main training function"""
    print("=" * 60)
    print("IMPROVED ACCURATE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    classifier = ImprovedAccurateClassifier()
    
    # Load and preprocess data
    df = classifier.load_and_preprocess_data()
    
    if df.empty:
        print("No data available for training!")
        return
    
    # Train models
    performance = classifier.train_improved_models(df)
    
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
