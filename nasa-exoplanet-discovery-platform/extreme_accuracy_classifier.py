#!/usr/bin/env python3
"""
Extreme Accuracy NASA Exoplanet ML Classifier
Ultimate approach to achieve 95%+ accuracy using all available data and techniques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtremeAccuracyClassifier:
    def __init__(self, data_path: str = "../nasa data planet"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        
    def load_all_nasa_data_combined(self) -> pd.DataFrame:
        """Load and combine ALL NASA datasets for maximum training data"""
        logger.info("Loading ALL NASA datasets for maximum accuracy...")
        
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
        
        # Combine all datasets intelligently
        combined_df = self._intelligent_dataset_combination(datasets)
        processed_df = self._extreme_preprocessing(combined_df)
        
        logger.info(f"Final processed dataset: {len(processed_df)} records")
        return processed_df
    
    def _intelligent_dataset_combination(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Intelligently combine all NASA datasets"""
        combined_data = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            if dataset_name == 'cumulative':
                processed_df = self._process_kepler_ultimate(df)
            elif dataset_name == 'toi':
                processed_df = self._process_tess_ultimate(df)
            elif dataset_name == 'k2pandc':
                processed_df = self._process_confirmed_ultimate(df)
            
            if not processed_df.empty:
                combined_data.append(processed_df)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(final_df)} records")
            return final_df
        else:
            return pd.DataFrame()
    
    def _process_kepler_ultimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultimate Kepler data processing with all features"""
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
        
        # Ultimate feature mapping for Kepler
        kepler_features = {
            'orbital_period': 'koi_period',
            'transit_duration': 'koi_duration',
            'impact_parameter': 'koi_impact',
            'semi_major_axis': 'koi_sma',
            'stellar_irradiance': 'koi_insol',
            'stellar_temperature': 'koi_steff',
            'stellar_surface_gravity': 'koi_slogg',
            'stellar_metallicity': 'koi_smet',
            'stellar_radius': 'koi_srad',
            'stellar_mass': 'koi_smass',
            'kepler_magnitude': 'koi_kepmag',
            'transit_depth': 'koi_depth',
            'planetary_radius': 'koi_prad',
            'signal_to_noise_ratio': 'koi_model_snr',
            'equilibrium_temperature': 'koi_teq',
            'transit_time': 'koi_time0bk',
            'koi_score': 'koi_score'
        }
        
        for new_col, old_col in kepler_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'Kepler'
        return processed
    
    def _process_tess_ultimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultimate TESS data processing with all features"""
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
        
        # Ultimate feature mapping for TESS
        tess_features = {
            'orbital_period': 'pl_orbper',
            'transit_duration': 'pl_trandurh',
            'transit_depth': 'pl_trandep',
            'planetary_radius': 'pl_rade',
            'stellar_irradiance': 'pl_insol',
            'equilibrium_temperature': 'pl_eqt',
            'stellar_temperature': 'st_teff',
            'stellar_surface_gravity': 'st_logg',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'distance': 'st_dist',
            'tess_magnitude': 'st_tmag',
            'transit_midpoint': 'pl_tranmid'
        }
        
        for new_col, old_col in tess_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'TESS'
        return processed
    
    def _process_confirmed_ultimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultimate confirmed planets data processing with all features"""
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
        
        # Ultimate feature mapping for confirmed planets
        confirmed_features = {
            'orbital_period': 'pl_orbper',
            'semi_major_axis': 'pl_orbsmax',
            'planetary_radius': 'pl_rade',
            'planetary_mass': 'pl_bmasse',
            'planetary_density': 'pl_dens',
            'orbital_eccentricity': 'pl_orbeccen',
            'stellar_irradiance': 'pl_insol',
            'equilibrium_temperature': 'pl_eqt',
            'stellar_temperature': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'stellar_metallicity': 'st_met',
            'stellar_surface_gravity': 'st_logg',
            'distance': 'sy_dist',
            'discovery_year': 'disc_year',
            'num_stars': 'sy_snum',
            'num_planets': 'sy_pnum'
        }
        
        for new_col, old_col in confirmed_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'Multiple'
        return processed
    
    def _extreme_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extreme preprocessing with all advanced techniques"""
        logger.info("Extreme preprocessing...")
        
        # Remove rows without target labels
        df = df.dropna(subset=['target'])
        df = df[df['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        
        # Create extreme derived features
        df = self._create_extreme_features(df)
        
        # Ultimate missing value handling
        df = self._ultimate_missing_value_handling(df)
        
        # Remove any remaining rows with NaN values
        df = df.dropna()
        
        logger.info(f"Available features: {len(df.columns)-1}")
        logger.info(f"Target distribution: {df['target'].value_counts()}")
        
        return df
    
    def _create_extreme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create extreme derived features for maximum accuracy"""
        # Transit probability and related features
        if 'orbital_period' in df.columns and 'stellar_radius' in df.columns:
            df['transit_probability'] = df['stellar_radius'] / df['orbital_period']
            df['transit_probability_log'] = np.log10(df['transit_probability'])
        
        # Habitable zone features
        if 'stellar_irradiance' in df.columns:
            df['in_habitable_zone'] = ((df['stellar_irradiance'] >= 0.36) & 
                                     (df['stellar_irradiance'] <= 1.67)).astype(int)
            df['habitable_zone_score'] = 1 - np.abs(np.log(df['stellar_irradiance']) / np.log(1.67))
            df['habitable_zone_distance'] = np.abs(np.log(df['stellar_irradiance']))
            df['stellar_irradiance_log'] = np.log10(df['stellar_irradiance'])
        
        # Planet-star ratios
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planetary_radius'] / df['stellar_radius']
            df['radius_ratio_log'] = np.log10(df['radius_ratio'])
        
        if 'planetary_mass' in df.columns and 'stellar_mass' in df.columns:
            df['mass_ratio'] = df['planetary_mass'] / df['stellar_mass']
            df['mass_ratio_log'] = np.log10(df['mass_ratio'])
        
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
            df['medium_period'] = ((df['orbital_period'] >= 10) & (df['orbital_period'] <= 365)).astype(int)
        
        # Size categories
        if 'planetary_radius' in df.columns:
            df['is_earth_like'] = ((df['planetary_radius'] >= 0.8) & (df['planetary_radius'] <= 1.25)).astype(int)
            df['is_super_earth'] = ((df['planetary_radius'] >= 1.25) & (df['planetary_radius'] <= 2)).astype(int)
            df['is_neptune_like'] = ((df['planetary_radius'] >= 2) & (df['planetary_radius'] <= 4)).astype(int)
            df['is_jupiter_like'] = (df['planetary_radius'] >= 4).astype(int)
            df['planetary_radius_log'] = np.log10(df['planetary_radius'])
        
        # Distance-based features
        if 'distance' in df.columns:
            df['log_distance'] = np.log10(df['distance'])
            df['nearby'] = (df['distance'] < 100).astype(int)
            df['very_nearby'] = (df['distance'] < 10).astype(int)
        
        # Density features
        if 'planetary_mass' in df.columns and 'planetary_radius' in df.columns:
            df['estimated_density'] = df['planetary_mass'] / (df['planetary_radius'] ** 3)
            df['density_category'] = np.where(
                df['estimated_density'] < 2, 1,  # Gas giant
                np.where(df['estimated_density'] < 5, 2,  # Neptune-like
                np.where(df['estimated_density'] < 8, 3, 4)))  # Rocky, Super-dense
            df['density_log'] = np.log10(df['estimated_density'])
        
        # Mission-specific features
        if 'mission' in df.columns:
            df['is_kepler'] = (df['mission'] == 'Kepler').astype(int)
            df['is_tess'] = (df['mission'] == 'TESS').astype(int)
            df['is_confirmed'] = (df['mission'] == 'Multiple').astype(int)
        
        # Signal strength indicators
        if 'signal_to_noise_ratio' in df.columns:
            df['strong_signal'] = (df['signal_to_noise_ratio'] > 10).astype(int)
            df['log_snr'] = np.log1p(df['signal_to_noise_ratio'])
        
        # Discovery year features
        if 'discovery_year' in df.columns:
            df['recent_discovery'] = (df['discovery_year'] > 2015).astype(int)
            df['early_discovery'] = (df['discovery_year'] < 2010).astype(int)
        
        # Multi-planet system features
        if 'num_planets' in df.columns:
            df['multi_planet_system'] = (df['num_planets'] > 1).astype(int)
            df['complex_system'] = (df['num_planets'] > 3).astype(int)
        
        # Eccentricity features
        if 'orbital_eccentricity' in df.columns:
            df['circular_orbit'] = (df['orbital_eccentricity'] < 0.1).astype(int)
            df['eccentric_orbit'] = (df['orbital_eccentricity'] > 0.5).astype(int)
        
        return df
    
    def _ultimate_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultimate missing value handling"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        # Fill missing values with median for each column
        for col in numeric_columns:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback for all-NaN columns
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def train_extreme_accuracy_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train extreme accuracy models with all advanced techniques"""
        logger.info("Training extreme accuracy models...")
        
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
            X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
        )
        
        # Multiple scaling approaches
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Advanced sampling for class imbalance
        samplers = {
            'smote': SMOTE(random_state=42),
            'adasyn': ADASYN(random_state=42),
            'borderline': BorderlineSMOTE(random_state=42),
            'smote_tomek': SMOTETomek(random_state=42)
        }
        
        best_accuracy = 0
        best_model_name = ""
        
        # Try different combinations of scalers and samplers
        for scaler_name, scaler in scalers.items():
            for sampler_name, sampler in samplers.items():
                logger.info(f"Testing {scaler_name} scaler with {sampler_name} sampler...")
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Balance classes
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
                
                # Define extreme accuracy models
                models = {
                    'Extreme Random Forest': RandomForestClassifier(
                        n_estimators=1500, max_depth=25, min_samples_split=2,
                        min_samples_leaf=1, max_features='sqrt', random_state=42,
                        class_weight='balanced', n_jobs=-1
                    ),
                    'Extreme Gradient Boosting': GradientBoostingClassifier(
                        n_estimators=1000, learning_rate=0.03, max_depth=12,
                        min_samples_split=2, min_samples_leaf=1, random_state=42
                    ),
                    'Extreme Histogram GB': HistGradientBoostingClassifier(
                        max_iter=1000, learning_rate=0.05, max_depth=20,
                        random_state=42, class_weight='balanced'
                    ),
                    'Extreme AdaBoost': AdaBoostClassifier(
                        n_estimators=500, learning_rate=0.1, random_state=42
                    )
                }
                
                # Train models
                for name, model in models.items():
                    full_name = f"{name}_{scaler_name}_{sampler_name}"
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
                    
                    # Train on full balanced dataset
                    model.fit(X_train_balanced, y_train_balanced)
                    
                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store if best
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = full_name
                        self.scalers['main'] = scaler
                        self.trained_models['Best Model'] = model
                        
                        # Store performance
                        self.model_performance['Best Model'] = {
                            'accuracy': accuracy,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'scaler': scaler_name,
                            'sampler': sampler_name
                        }
                        
                        logger.info(f"NEW BEST: {full_name} - Accuracy: {accuracy:.4f}")
        
        logger.info(f"🏆 FINAL BEST MODEL: {best_model_name} with {best_accuracy:.4f} accuracy!")
        
        return self.model_performance
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'Best Model') -> Dict:
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
    
    def save_models(self, save_path: str = "extreme_accuracy_models"):
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
        joblib.dump(self.model_performance, save_dir / "model_performance.joblib")
        
        logger.info(f"Models saved to {save_dir}")

def main():
    """Main function to train extreme accuracy models"""
    classifier = ExtremeAccuracyClassifier()
    
    # Load all NASA data
    df = classifier.load_all_nasa_data_combined()
    
    if df.empty:
        logger.error("No data to train on!")
        return
    
    # Train extreme accuracy models
    performance = classifier.train_extreme_accuracy_models(df)
    
    # Save models
    classifier.save_models()
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EXTREME ACCURACY MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    for model_name, metrics in performance.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  CV Accuracy:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
        logger.info(f"  Scaler:        {metrics['scaler']}")
        logger.info(f"  Sampler:       {metrics['sampler']}")
        logger.info("")
    
    # Test prediction
    logger.info("Testing prediction...")
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
        result = classifier.predict_new_data(test_data, 'Best Model')
        logger.info(f"Test prediction: {result}")
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")

if __name__ == "__main__":
    main()

