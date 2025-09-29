#!/usr/bin/env python3
"""
Ultra High-Accuracy NASA Exoplanet ML Classifier
Focused approach to achieve 95%+ accuracy with robust data handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import joblib
import logging
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraHighAccuracyClassifier:
    def __init__(self, data_path: str = "../nasa data planet"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        
    def load_optimized_data(self) -> pd.DataFrame:
        """Load and optimize NASA data with focus on high-quality features"""
        logger.info("Loading optimized NASA exoplanet datasets...")
        
        # Focus on the most reliable dataset first (confirmed planets)
        confirmed_file = self.data_path / 'k2pandc_2025.09.27_12.56.23.csv'
        
        if not confirmed_file.exists():
            logger.error(f"Data file not found: {confirmed_file}")
            return pd.DataFrame()
        
        logger.info(f"Loading confirmed planets dataset...")
        df = pd.read_csv(confirmed_file, comment='#')
        logger.info(f"Loaded {len(df)} records")
        
        # Process with focus on quality features
        processed_df = self._optimized_preprocessing(df)
        
        logger.info(f"Final processed dataset: {len(processed_df)} records")
        return processed_df
    
    def _optimized_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized preprocessing focused on high-quality features"""
        logger.info("Optimized preprocessing...")
        
        processed = pd.DataFrame()
        
        # Create target variable with proper mapping
        if 'disposition' in df.columns:
            processed['target'] = df['disposition'].map({
                'CONFIRMED': 'CONFIRMED',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE_POSITIVE',
                'REFUTED': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'CONFIRMED'
        
        # Focus on the most reliable and complete features
        core_features = {
            'orbital_period': 'pl_orbper',
            'planetary_radius': 'pl_rade',
            'planetary_mass': 'pl_bmasse',
            'stellar_temperature': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'distance': 'sy_dist',
            'stellar_irradiance': 'pl_insol',
            'equilibrium_temperature': 'pl_eqt',
            'orbital_eccentricity': 'pl_orbeccen',
            'stellar_metallicity': 'st_met',
            'stellar_surface_gravity': 'st_logg'
        }
        
        # Extract core features with robust handling
        for new_col, old_col in core_features.items():
            if old_col in df.columns:
                # Convert to numeric, replacing errors with NaN
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # Remove rows without target labels
        processed = processed.dropna(subset=['target'])
        processed = processed[processed['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        
        # Create high-quality derived features
        processed = self._create_high_quality_features(processed)
        
        # Robust missing value handling
        processed = self._robust_missing_value_handling(processed)
        
        # Remove any remaining rows with NaN values
        processed = processed.dropna()
        
        logger.info(f"Available features: {len(processed.columns)-1}")
        logger.info(f"Target distribution: {processed['target'].value_counts()}")
        
        return processed
    
    def _create_high_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create high-quality derived features"""
        # Transit probability (key indicator)
        if 'orbital_period' in df.columns and 'stellar_radius' in df.columns:
            df['transit_probability'] = df['stellar_radius'] / df['orbital_period']
        
        # Habitable zone indicators (critical for classification)
        if 'stellar_irradiance' in df.columns:
            df['in_habitable_zone'] = ((df['stellar_irradiance'] >= 0.36) & 
                                     (df['stellar_irradiance'] <= 1.67)).astype(int)
            df['habitable_zone_score'] = 1 - np.abs(np.log(df['stellar_irradiance']) / np.log(1.67))
        
        # Planet-star ratios (important for false positive detection)
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planetary_radius'] / df['stellar_radius']
        
        if 'planetary_mass' in df.columns and 'stellar_mass' in df.columns:
            df['mass_ratio'] = df['planetary_mass'] / df['stellar_mass']
        
        # Temperature-based features
        if 'stellar_temperature' in df.columns:
            df['log_stellar_temp'] = np.log10(df['stellar_temperature'])
            df['stellar_type_numeric'] = np.where(
                df['stellar_temperature'] < 4000, 1,  # M-type
                np.where(df['stellar_temperature'] < 5000, 2,  # K-type
                np.where(df['stellar_temperature'] < 6000, 3,  # G-type
                np.where(df['stellar_temperature'] < 7500, 4, 5))))  # F-type, A-type
        
        # Orbital characteristics
        if 'orbital_period' in df.columns:
            df['log_orbital_period'] = np.log10(df['orbital_period'])
            df['short_period'] = (df['orbital_period'] < 10).astype(int)
            df['long_period'] = (df['orbital_period'] > 365).astype(int)
        
        # Size categories (important for classification)
        if 'planetary_radius' in df.columns:
            df['is_super_earth'] = ((df['planetary_radius'] >= 1.25) & (df['planetary_radius'] <= 2)).astype(int)
            df['is_neptune_like'] = ((df['planetary_radius'] >= 2) & (df['planetary_radius'] <= 4)).astype(int)
            df['is_jupiter_like'] = (df['planetary_radius'] >= 4).astype(int)
        
        # Distance-based features
        if 'distance' in df.columns:
            df['log_distance'] = np.log10(df['distance'])
            df['nearby'] = (df['distance'] < 100).astype(int)
        
        # Density features (if available)
        if 'planetary_mass' in df.columns and 'planetary_radius' in df.columns:
            df['estimated_density'] = df['planetary_mass'] / (df['planetary_radius'] ** 3)
            df['density_category'] = np.where(
                df['estimated_density'] < 2, 1,  # Gas giant
                np.where(df['estimated_density'] < 5, 2,  # Neptune-like
                np.where(df['estimated_density'] < 8, 3, 4)))  # Rocky, Super-dense
        
        return df
    
    def _robust_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust missing value handling"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        # Fill missing values with median for each column
        for col in numeric_columns:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback for all-NaN columns
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def train_ultra_high_accuracy_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train ultra high-accuracy models with advanced techniques"""
        logger.info("Training ultra high-accuracy models...")
        
        # Prepare features and target
        feature_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        if len(feature_columns) == 0:
            logger.error("No valid numeric features found!")
            return {}
        
        X = df[feature_columns]
        y = df['target']
        
        logger.info(f"Training with {len(feature_columns)} features")
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts()}")
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['target'] = le
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )
        
        # Advanced scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Advanced sampling for class imbalance
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Balanced training data shape: {X_train_balanced.shape}")
        
        # Define ultra high-accuracy models
        models = {
            'Random Forest Optimized': RandomForestClassifier(
                n_estimators=1000, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Extra Trees Optimized': ExtraTreesClassifier(
                n_estimators=1000, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting Optimized': GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=10,
                min_samples_split=2, min_samples_leaf=1, random_state=42
            ),
            'Histogram Gradient Boosting': HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.1, max_depth=15,
                random_state=42, class_weight='balanced'
            ),
            'SVM Optimized': SVC(
                kernel='rbf', C=100, gamma='scale', 
                class_weight='balanced', probability=True, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=10, max_iter=1000, class_weight='balanced', random_state=42
            )
        }
        
        # Train individual models with cross-validation
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation for model validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
            logger.info(f"{name} CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train on full balanced dataset
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions on test set
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_columns, model.feature_importances_))
                self.feature_importance[name] = importance
            
            # Store model and metrics
            self.trained_models[name] = model
            self.model_performance[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, target_names=le.classes_),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}")
        
        # Create ensemble model
        logger.info("Creating ultra high-accuracy ensemble...")
        ensemble_models = [
            ('rf', self.trained_models['Random Forest Optimized']),
            ('et', self.trained_models['Extra Trees Optimized']),
            ('gb', self.trained_models['Gradient Boosting Optimized']),
            ('hgb', self.trained_models['Histogram Gradient Boosting']),
            ('svm', self.trained_models['SVM Optimized']),
            ('lr', self.trained_models['Logistic Regression'])
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        # Cross-validation for ensemble
        ensemble_cv = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        
        self.trained_models['Ultra High-Accuracy Ensemble'] = ensemble
        self.model_performance['Ultra High-Accuracy Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'cv_mean': ensemble_cv.mean(),
            'cv_std': ensemble_cv.std(),
            'classification_report': classification_report(y_test, y_pred_ensemble, target_names=le.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist(),
            'predictions': y_pred_ensemble.tolist(),
            'probabilities': y_pred_proba_ensemble.tolist()
        }
        
        logger.info(f"Ultra High-Accuracy Ensemble - Test Accuracy: {ensemble_accuracy:.4f}")
        logger.info(f"Ultra High-Accuracy Ensemble - CV Accuracy: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")
        
        return self.model_performance
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'Ultra High-Accuracy Ensemble') -> Dict:
        """Predict classification for new exoplanet data"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        # Prepare new data with the same features used in training
        feature_columns = list(self.scalers['main'].feature_names_in_)
        
        # Create input data with the required features
        input_data = pd.DataFrame()
        for col in feature_columns:
            if col in new_data.columns:
                input_data[col] = new_data[col]
            else:
                input_data[col] = 0  # Default value for missing features
        
        # Scale features
        X_new_scaled = self.scalers['main'].transform(input_data[feature_columns])
        
        # Make predictions
        model = self.trained_models[model_name]
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)
        
        # Decode predictions
        predictions_decoded = self.label_encoders['target'].inverse_transform(predictions)
        
        return {
            'predictions': predictions_decoded.tolist(),
            'probabilities': probabilities.tolist(),
            'confidence': np.max(probabilities, axis=1).tolist()
        }
    
    def save_models(self, save_path: str = "ultra_high_accuracy_models"):
        """Save trained models and preprocessors"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = save_dir / f"{name.lower().replace(' ', '_').replace('-', '_')}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save other components
        joblib.dump(self.scalers, save_dir / "scalers.joblib")
        joblib.dump(self.label_encoders, save_dir / "label_encoders.joblib")
        joblib.dump(self.feature_importance, save_dir / "feature_importance.joblib")
        joblib.dump(self.model_performance, save_dir / "model_performance.joblib")
        
        logger.info(f"Models saved to {save_dir}")

def main():
    """Main function to train ultra high-accuracy models"""
    classifier = UltraHighAccuracyClassifier()
    
    # Load and preprocess optimized data
    df = classifier.load_optimized_data()
    
    if df.empty:
        logger.error("No data to train on!")
        return
    
    # Train ultra high-accuracy models
    performance = classifier.train_ultra_high_accuracy_models(df)
    
    # Save models
    classifier.save_models()
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("ULTRA HIGH-ACCURACY MODEL PERFORMANCE SUMMARY")
    logger.info("="*70)
    for model_name, metrics in performance.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  CV Accuracy:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
        logger.info("")
    
    # Find best model
    best_model = max(performance.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"🏆 BEST MODEL: {best_model[0]}")
    logger.info(f"   Test Accuracy: {best_model[1]['accuracy']*100:.2f}%")
    logger.info(f"   CV Accuracy: {best_model[1]['cv_mean']*100:.2f}%")
    
    # Test prediction
    logger.info("\nTesting prediction...")
    test_data = pd.DataFrame({
        'orbital_period': [365.25],
        'planetary_radius': [1.0],
        'planetary_mass': [1.0],
        'stellar_temperature': [5778],
        'stellar_radius': [1.0],
        'stellar_mass': [1.0],
        'distance': [1.3],
        'stellar_irradiance': [1.0],
        'equilibrium_temperature': [288]
    })
    
    try:
        result = classifier.predict_new_data(test_data, best_model[0])
        logger.info(f"Test prediction: {result}")
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")

if __name__ == "__main__":
    main()

