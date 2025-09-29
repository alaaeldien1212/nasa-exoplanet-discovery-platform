#!/usr/bin/env python3
"""
ENHANCED HIGH-ACCURACY EXOPLANET CLASSIFIER
Advanced ML model with improved accuracy and confidence
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class EnhancedExoplanetClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        self.feature_selector = None
        self.best_model = None
        
    def load_nasa_data(self):
        """Load NASA data with enhanced preprocessing"""
        print("Loading NASA data...")
        
        try:
            df = pd.read_csv('../nasa data planet/cumulative_2025.09.27_12.55.48.csv', comment='#')
            print(f"Loaded data: {len(df)} records")
            return self.preprocess_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """Enhanced data preprocessing with feature engineering"""
        print("Enhanced preprocessing data...")
        
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
        
        # Select features
        feature_cols = ['orbital_period', 'planetary_radius', 'stellar_temperature', 'stellar_radius']
        
        # Check which features exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Available features: {available_features}")
        
        if 'disposition' not in df.columns:
            print("No disposition column found!")
            return pd.DataFrame()
        
        # Select data
        df = df[available_features + ['disposition']].copy()
        
        # Convert to numeric with better error handling
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['disposition'])
        df = df[df['disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
        
        # Enhanced feature engineering
        df = self.create_enhanced_features(df, available_features)
        
        # Fill missing values with median
        for col in df.columns:
            if col != 'disposition' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        print(f"Final dataset: {len(df)} records")
        print("Class distribution:")
        print(df['disposition'].value_counts())
        
        return df
    
    def create_enhanced_features(self, df, base_features):
        """Create enhanced features for better accuracy"""
        print("Creating enhanced features...")
        
        # Basic features
        for feature in base_features:
            if feature in df.columns:
                # Log transform for skewed features
                if feature in ['orbital_period', 'planetary_radius']:
                    df[f'{feature}_log'] = np.log1p(df[feature])
                
                # Square root transform
                df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
                
                # Binned features
                df[f'{feature}_binned'] = pd.cut(df[feature], bins=5, labels=False)
        
        # Interaction features
        if 'orbital_period' in df.columns and 'planetary_radius' in df.columns:
            df['period_radius_ratio'] = df['orbital_period'] / (df['planetary_radius'] + 1e-6)
            df['period_radius_product'] = df['orbital_period'] * df['planetary_radius']
        
        if 'stellar_temperature' in df.columns and 'stellar_radius' in df.columns:
            df['stellar_luminosity'] = df['stellar_radius'] ** 2 * (df['stellar_temperature'] / 5778) ** 4
            df['stellar_density'] = df['stellar_radius'] ** -3
        
        # Physical plausibility features
        if 'planetary_radius' in df.columns:
            df['is_gas_giant'] = (df['planetary_radius'] > 4).astype(int)
            df['is_super_earth'] = ((df['planetary_radius'] > 1) & (df['planetary_radius'] < 2)).astype(int)
            df['is_earth_like'] = ((df['planetary_radius'] > 0.8) & (df['planetary_radius'] < 1.2)).astype(int)
        
        if 'orbital_period' in df.columns:
            df['is_hot_jupiter'] = ((df['orbital_period'] < 10) & (df['planetary_radius'] > 4)).astype(int)
            df['is_close_in'] = (df['orbital_period'] < 10).astype(int)
        
        # Statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'disposition']
        
        for col in numeric_cols:
            if col in df.columns:
                # Z-score normalization
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
                
                # Percentile ranking
                df[f'{col}_percentile'] = df[col].rank(pct=True)
        
        print(f"Enhanced features created. Total features: {len(df.columns) - 1}")
        return df
    
    def train_enhanced_models(self, df):
        """Train multiple enhanced models with hyperparameter tuning"""
        print("Training enhanced models...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'disposition']
        X = df[feature_cols]
        y = df['disposition']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_features = self.feature_selector.get_support(indices=True)
        self.feature_names = [feature_cols[i] for i in selected_features]
        
        print(f"Selected {len(self.feature_names)} best features")
        
        # Scale features
        self.scalers['robust'] = RobustScaler()
        X_train_scaled = self.scalers['robust'].fit_transform(X_train_selected)
        X_test_scaled = self.scalers['robust'].transform(X_test_selected)
        
        # Define models with hyperparameter tuning
        models_config = {
            'Enhanced Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.8]
                }
            },
            'Enhanced Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Enhanced Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.8]
                }
            },
            'Enhanced SVM': {
                'model': SVC(random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly']
                }
            }
        }
        
        best_accuracy = 0
        best_model_name = None
        
        # Train and tune each model
        for model_name, config in models_config.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            self.models[model_name] = best_model
            
            # Evaluate
            y_pred = best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{model_name} - Best params: {grid_search.best_params_}")
            print(f"{model_name} - Accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                self.best_model = best_model
        
        # Train ensemble model
        print(f"\nTraining Enhanced Ensemble...")
        ensemble_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        )
        ensemble_model.fit(X_train_scaled, y_train)
        self.models['Enhanced Ensemble'] = ensemble_model
        
        y_pred_ensemble = ensemble_model.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"Enhanced Ensemble - Accuracy: {ensemble_accuracy:.4f}")
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("FINAL MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        # Detailed evaluation of best model
        y_pred_best = self.best_model.predict(X_test_scaled)
        print(f"\nDetailed Performance:")
        print(classification_report(y_test, y_pred_best))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))
        
        return best_accuracy
    
    def predict(self, data, model_name='Enhanced Ensemble'):
        """Make predictions with enhanced confidence"""
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
        
        # Feature selection
        X_selected = self.feature_selector.transform(X)
        
        # Scale features
        X_scaled = self.scalers['robust'].transform(X_selected)
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'model_used': model_name
        }
    
    def save_enhanced_model(self):
        """Save the enhanced model"""
        print("Saving enhanced model...")
        
        import os
        os.makedirs('enhanced_models', exist_ok=True)
        
        joblib.dump(self.models, 'enhanced_models/models.joblib')
        joblib.dump(self.scalers, 'enhanced_models/scalers.joblib')
        joblib.dump(self.feature_names, 'enhanced_models/feature_names.joblib')
        joblib.dump(self.class_names, 'enhanced_models/class_names.joblib')
        joblib.dump(self.feature_selector, 'enhanced_models/feature_selector.joblib')
        joblib.dump(self.best_model, 'enhanced_models/best_model.joblib')
        
        print("Enhanced model saved successfully!")
    
    def load_enhanced_model(self):
        """Load the enhanced model"""
        print("Loading enhanced model...")
        
        try:
            self.models = joblib.load('enhanced_models/models.joblib')
            self.scalers = joblib.load('enhanced_models/scalers.joblib')
            self.feature_names = joblib.load('enhanced_models/feature_names.joblib')
            self.class_names = joblib.load('enhanced_models/class_names.joblib')
            self.feature_selector = joblib.load('enhanced_models/feature_selector.joblib')
            self.best_model = joblib.load('enhanced_models/best_model.joblib')
            print("Enhanced model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            return False

def main():
    """Main training function"""
    print("=" * 60)
    print("ENHANCED HIGH-ACCURACY EXOPLANET CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    classifier = EnhancedExoplanetClassifier()
    
    # Load data
    df = classifier.load_nasa_data()
    
    if df.empty:
        print("No data available for training!")
        return
    
    # Train enhanced models
    accuracy = classifier.train_enhanced_models(df)
    
    # Save model
    classifier.save_enhanced_model()
    
    print("\n" + "=" * 60)
    print("ENHANCED TRAINING COMPLETE!")
    print(f"Best accuracy achieved: {accuracy:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
