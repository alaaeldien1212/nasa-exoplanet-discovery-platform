#!/usr/bin/env python3
"""
Final High-Accuracy NASA Exoplanet ML Classifier
Robust approach to achieve maximum accuracy with proper data handling
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

class FinalHighAccuracyClassifier:
    def __init__(self, data_path: str = "../nasa data planet"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        
    def load_all_nasa_data_robust(self) -> pd.DataFrame:
        """Load all NASA data with robust preprocessing"""
        logger.info("Loading ALL NASA datasets with robust preprocessing...")
        
        datasets = {}
        csv_files = {
            'cumulative': self.data_path / 'cumulative_2025.09.27_12.55.48.csv',
            'toi': self.data_path / 'TOI_2025.09.27_12.56.11.csv',
            'k2pandc': self.data_path / 'k2pandc_2025.09.27_12.56.23.csv'
        }
        
        total_records = 0
        for name, file_path in csv_files.items():
            if file_path.exists():
                logger.info(f"Loading {name} dataset...")
                df = pd.read_csv(file_path, comment='#')
                datasets[name] = df
                total_records += len(df)
                logger.info(f"Loaded {len(df)} records from {name}")
        
        logger.info(f"Total records loaded: {total_records}")
        
        # Combine all datasets
        combined_df = self._combine_datasets_robust(datasets)
        processed_df = self._robust_preprocessing(combined_df)
        
        logger.info(f"Final processed dataset: {len(processed_df)} records")
        return processed_df
    
    def _combine_datasets_robust(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Robustly combine all NASA datasets"""
        combined_data = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            if dataset_name == 'cumulative':
                processed_df = self._process_kepler_robust(df)
            elif dataset_name == 'toi':
                processed_df = self._process_tess_robust(df)
            elif dataset_name == 'k2pandc':
                processed_df = self._process_confirmed_robust(df)
            
            if not processed_df.empty:
                combined_data.append(processed_df)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(final_df)} records")
            return final_df
        else:
            return pd.DataFrame()
    
    def _process_kepler_robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust Kepler data processing"""
        processed = pd.DataFrame()
        
        # Target mapping
        if 'koi_disposition' in df.columns:
            processed['target'] = df['koi_disposition'].map({
                'CONFIRMED': 'CONFIRMED',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'CANDIDATE'
        
        # Core Kepler features
        kepler_features = {
            'orbital_period': 'koi_period',
            'transit_duration': 'koi_duration',
            'planetary_radius': 'koi_prad',
            'stellar_temperature': 'koi_steff',
            'stellar_radius': 'koi_srad',
            'stellar_mass': 'koi_smass',
            'stellar_irradiance': 'koi_insol',
            'equilibrium_temperature': 'koi_teq',
            'signal_to_noise_ratio': 'koi_model_snr',
            'kepler_magnitude': 'koi_kepmag'
        }
        
        for new_col, old_col in kepler_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'Kepler'
        return processed
    
    def _process_tess_robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust TESS data processing"""
        processed = pd.DataFrame()
        
        # Target mapping
        if 'tfopwg_disp' in df.columns:
            processed['target'] = df['tfopwg_disp'].map({
                'CP': 'CONFIRMED',
                'PC': 'CANDIDATE',
                'FP': 'FALSE_POSITIVE',
                'KP': 'CANDIDATE',
                'APC': 'CANDIDATE',
                'FA': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'CANDIDATE'
        
        # Core TESS features
        tess_features = {
            'orbital_period': 'pl_orbper',
            'planetary_radius': 'pl_rade',
            'stellar_temperature': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'stellar_irradiance': 'pl_insol',
            'equilibrium_temperature': 'pl_eqt',
            'distance': 'st_dist',
            'tess_magnitude': 'st_tmag'
        }
        
        for new_col, old_col in tess_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'TESS'
        return processed
    
    def _process_confirmed_robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust confirmed planets data processing"""
        processed = pd.DataFrame()
        
        # Target mapping
        if 'disposition' in df.columns:
            processed['target'] = df['disposition'].map({
                'CONFIRMED': 'CONFIRMED',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE_POSITIVE',
                'REFUTED': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'CONFIRMED'
        
        # Core confirmed planets features
        confirmed_features = {
            'orbital_period': 'pl_orbper',
            'planetary_radius': 'pl_rade',
            'planetary_mass': 'pl_bmasse',
            'planetary_density': 'pl_dens',
            'stellar_temperature': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'stellar_metallicity': 'st_met',
            'stellar_surface_gravity': 'st_logg',
            'distance': 'sy_dist',
            'stellar_irradiance': 'pl_insol',
            'equilibrium_temperature': 'pl_eqt',
            'orbital_eccentricity': 'pl_orbeccen',
            'discovery_year': 'disc_year'
        }
        
        for new_col, old_col in confirmed_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'Multiple'
        return processed
    
    def _robust_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust preprocessing with proper handling of extreme values"""
        logger.info("Robust preprocessing...")
        
        # Remove rows without target labels
        df = df.dropna(subset=['target'])
        df = df[df['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        
        # Create robust derived features
        df = self._create_robust_features(df)
        
        # Handle missing values and extreme values
        df = self._handle_missing_and_extreme_values(df)
        
        # Remove any remaining rows with NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        logger.info(f"Available features: {len(df.columns)-1}")
        logger.info(f"Target distribution: {df['target'].value_counts()}")
        
        return df
    
    def _create_robust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create robust derived features"""
        # Transit probability (with safety checks)
        if 'orbital_period' in df.columns and 'stellar_radius' in df.columns:
            df['transit_probability'] = np.where(
                (df['orbital_period'] > 0) & (df['stellar_radius'] > 0),
                df['stellar_radius'] / df['orbital_period'],
                0
            )
        
        # Habitable zone indicators
        if 'stellar_irradiance' in df.columns:
            df['in_habitable_zone'] = ((df['stellar_irradiance'] >= 0.36) & 
                                     (df['stellar_irradiance'] <= 1.67)).astype(int)
            df['habitable_zone_score'] = np.where(
                df['stellar_irradiance'] > 0,
                1 - np.abs(np.log(df['stellar_irradiance']) / np.log(1.67)),
                0
            )
        
        # Planet-star ratios (with safety checks)
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = np.where(
                (df['stellar_radius'] > 0),
                df['planetary_radius'] / df['stellar_radius'],
                0
            )
        
        if 'planetary_mass' in df.columns and 'stellar_mass' in df.columns:
            df['mass_ratio'] = np.where(
                (df['stellar_mass'] > 0),
                df['planetary_mass'] / df['stellar_mass'],
                0
            )
        
        # Temperature categories
        if 'stellar_temperature' in df.columns:
            df['stellar_type_numeric'] = np.where(
                df['stellar_temperature'] < 4000, 1,  # M-type
                np.where(df['stellar_temperature'] < 5000, 2,  # K-type
                np.where(df['stellar_temperature'] < 6000, 3,  # G-type
                np.where(df['stellar_temperature'] < 7500, 4, 5))))  # F-type, A-type
        
        # Orbital period categories
        if 'orbital_period' in df.columns:
            df['short_period'] = (df['orbital_period'] < 10).astype(int)
            df['long_period'] = (df['orbital_period'] > 365).astype(int)
            df['medium_period'] = ((df['orbital_period'] >= 10) & (df['orbital_period'] <= 365)).astype(int)
        
        # Size categories
        if 'planetary_radius' in df.columns:
            df['is_earth_like'] = ((df['planetary_radius'] >= 0.8) & (df['planetary_radius'] <= 1.25)).astype(int)
            df['is_super_earth'] = ((df['planetary_radius'] >= 1.25) & (df['planetary_radius'] <= 2)).astype(int)
            df['is_neptune_like'] = ((df['planetary_radius'] >= 2) & (df['planetary_radius'] <= 4)).astype(int)
            df['is_jupiter_like'] = (df['planetary_radius'] >= 4).astype(int)
        
        # Distance-based features
        if 'distance' in df.columns:
            df['nearby'] = (df['distance'] < 100).astype(int)
            df['very_nearby'] = (df['distance'] < 10).astype(int)
        
        # Density features (with safety checks)
        if 'planetary_mass' in df.columns and 'planetary_radius' in df.columns:
            df['estimated_density'] = np.where(
                (df['planetary_radius'] > 0),
                df['planetary_mass'] / (df['planetary_radius'] ** 3),
                0
            )
            df['density_category'] = np.where(
                df['estimated_density'] < 2, 1,  # Gas giant
                np.where(df['estimated_density'] < 5, 2,  # Neptune-like
                np.where(df['estimated_density'] < 8, 3, 4)))  # Rocky, Super-dense
        
        # Mission-specific features
        if 'mission' in df.columns:
            df['is_kepler'] = (df['mission'] == 'Kepler').astype(int)
            df['is_tess'] = (df['mission'] == 'TESS').astype(int)
            df['is_confirmed'] = (df['mission'] == 'Multiple').astype(int)
        
        # Signal strength indicators
        if 'signal_to_noise_ratio' in df.columns:
            df['strong_signal'] = (df['signal_to_noise_ratio'] > 10).astype(int)
        
        # Discovery year features
        if 'discovery_year' in df.columns:
            df['recent_discovery'] = (df['discovery_year'] > 2015).astype(int)
            df['early_discovery'] = (df['discovery_year'] < 2010).astype(int)
        
        # Eccentricity features
        if 'orbital_eccentricity' in df.columns:
            df['circular_orbit'] = (df['orbital_eccentricity'] < 0.1).astype(int)
            df['eccentric_orbit'] = (df['orbital_eccentricity'] > 0.5).astype(int)
        
        return df
    
    def _handle_missing_and_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and extreme values robustly"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        for col in numeric_columns:
            if col in df.columns:
                # Replace infinite values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill missing values with median
                if df[col].isna().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # Fallback for all-NaN columns
                    df[col] = df[col].fillna(median_val)
                
                # Cap extreme values at 99th percentile
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def train_final_high_accuracy_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train final high-accuracy models"""
        logger.info("Training final high-accuracy models...")
        
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
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Advanced sampling for class imbalance
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Balanced training data shape: {X_train_balanced.shape}")
        
        # Define final high-accuracy models
        models = {
            'Final Random Forest': RandomForestClassifier(
                n_estimators=2000, max_depth=30, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Final Extra Trees': ExtraTreesClassifier(
                n_estimators=2000, max_depth=30, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Final Gradient Boosting': GradientBoostingClassifier(
                n_estimators=1000, learning_rate=0.02, max_depth=15,
                min_samples_split=2, min_samples_leaf=1, random_state=42
            ),
            'Final Histogram GB': HistGradientBoostingClassifier(
                max_iter=2000, learning_rate=0.03, max_depth=25,
                random_state=42, class_weight='balanced'
            ),
            'Final SVM': SVC(
                kernel='rbf', C=1000, gamma='scale', 
                class_weight='balanced', probability=True, random_state=42
            )
        }
        
        # Train individual models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
            
            # Train on full balanced dataset
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
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
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        
        # Create final ensemble
        logger.info("Creating final ensemble...")
        ensemble_models = [
            ('rf', self.trained_models['Final Random Forest']),
            ('et', self.trained_models['Final Extra Trees']),
            ('gb', self.trained_models['Final Gradient Boosting']),
            ('hgb', self.trained_models['Final Histogram GB']),
            ('svm', self.trained_models['Final SVM'])
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        # Cross-validation for ensemble
        ensemble_cv = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        
        self.trained_models['Final Ensemble'] = ensemble
        self.model_performance['Final Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'cv_mean': ensemble_cv.mean(),
            'cv_std': ensemble_cv.std(),
            'classification_report': classification_report(y_test, y_pred_ensemble, target_names=le.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist(),
            'predictions': y_pred_ensemble.tolist(),
            'probabilities': y_pred_proba_ensemble.tolist()
        }
        
        logger.info(f"Final Ensemble - Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        logger.info(f"Final Ensemble - CV Accuracy: {ensemble_cv.mean():.4f} ({ensemble_cv.mean()*100:.2f}%)")
        
        return self.model_performance
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'Final Ensemble') -> Dict:
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
        
        # Handle NaN values - fill with median for numeric columns
        for col in input_data.columns:
            if input_data[col].dtype in ['float64', 'int64']:
                if input_data[col].isna().any():
                    # Fill NaN with median of non-NaN values, or 0 if all NaN
                    median_val = input_data[col].median()
                    if pd.isna(median_val):
                        input_data[col] = input_data[col].fillna(0)
                    else:
                        input_data[col] = input_data[col].fillna(median_val)
        
        # Final check - replace any remaining NaN with 0
        input_data = input_data.fillna(0)
        
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
    
    def save_models(self, save_path: str = "final_high_accuracy_models"):
        """Save trained models and preprocessors"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = save_dir / f"{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save other components
        joblib.dump(self.scalers, save_dir / "scalers.joblib")
        joblib.dump(self.label_encoders, save_dir / "label_encoders.joblib")
        joblib.dump(self.feature_importance, save_dir / "feature_importance.joblib")
        joblib.dump(self.model_performance, save_dir / "model_performance.joblib")
        
        logger.info(f"Models saved to {save_dir}")

def main():
    """Main function to train final high-accuracy models"""
    classifier = FinalHighAccuracyClassifier()
    
    # Load all NASA data
    df = classifier.load_all_nasa_data_robust()
    
    if df.empty:
        logger.error("No data to train on!")
        return
    
    # Train final high-accuracy models
    performance = classifier.train_final_high_accuracy_models(df)
    
    # Save models
    classifier.save_models()
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("FINAL HIGH-ACCURACY MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    for model_name, metrics in performance.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  CV Accuracy:   {metrics['cv_mean']:.4f} ({metrics['cv_mean']*100:.2f}%)")
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
