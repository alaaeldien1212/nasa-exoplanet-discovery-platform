#!/usr/bin/env python3
"""
High-Accuracy NASA Exoplanet ML Classifier
Advanced ML techniques to achieve 95%+ accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighAccuracyExoplanetClassifier:
    def __init__(self, data_path: str = "../nasa data planet"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        self.feature_selector = None
        
    def load_all_nasa_data(self) -> pd.DataFrame:
        """Load and combine all available NASA datasets for maximum training data"""
        logger.info("Loading ALL NASA exoplanet datasets...")
        
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
        
        # Combine all datasets with comprehensive feature mapping
        combined_df = self._combine_all_datasets(datasets)
        processed_df = self._advanced_preprocessing(combined_df)
        
        logger.info(f"Final processed dataset: {len(processed_df)} records")
        return processed_df
    
    def _combine_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all NASA datasets with comprehensive feature mapping"""
        combined_data = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            if dataset_name == 'cumulative':
                processed_df = self._process_kepler_comprehensive(df)
            elif dataset_name == 'toi':
                processed_df = self._process_tess_comprehensive(df)
            elif dataset_name == 'k2pandc':
                processed_df = self._process_confirmed_comprehensive(df)
            
            if not processed_df.empty:
                combined_data.append(processed_df)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(final_df)} records")
            return final_df
        else:
            return pd.DataFrame()
    
    def _process_kepler_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Kepler data with all available features"""
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
        
        # Comprehensive feature mapping for Kepler
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
            'transit_time': 'koi_time0bk'
        }
        
        for new_col, old_col in kepler_features.items():
            if old_col in df.columns:
                processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        processed['mission'] = 'Kepler'
        return processed
    
    def _process_tess_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process TESS data with all available features"""
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
        
        # Comprehensive feature mapping for TESS
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
    
    def _process_confirmed_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process confirmed planets data with all available features"""
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
        
        # Comprehensive feature mapping for confirmed planets
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
            'stellar_type': 'st_spectype',
            'discovery_year': 'disc_year',
            'num_stars': 'sy_snum',
            'num_planets': 'sy_pnum'
        }
        
        for new_col, old_col in confirmed_features.items():
            if old_col in df.columns:
                if old_col == 'st_spectype':
                    # Handle spectral type as categorical
                    processed[new_col] = df[old_col].astype(str)
                else:
                    processed[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # Add discovery method as categorical
        if 'discoverymethod' in df.columns:
            processed['discovery_method'] = df['discoverymethod'].astype(str)
        
        processed['mission'] = 'Multiple'
        return processed
    
    def _advanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced preprocessing with sophisticated techniques"""
        logger.info("Advanced preprocessing...")
        
        # Remove rows without target labels
        df = df.dropna(subset=['target'])
        df = df[df['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.drop(['target'])
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encode categorical variables
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Advanced missing value imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        # Fill any remaining NaN values with median before KNN imputation
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Use KNN imputation for better missing value handling
        if len(numeric_columns) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        # Final check - remove any rows with NaN values
        df = df.dropna()
        
        # Create advanced derived features
        df = self._create_advanced_features(df)
        
        # Remove outliers more intelligently
        df = self._intelligent_outlier_removal(df)
        
        logger.info(f"Preprocessing complete. Features: {len(df.columns)}")
        return df
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced derived features for better classification"""
        # Transit probability
        if 'orbital_period' in df.columns and 'stellar_radius' in df.columns:
            df['transit_probability'] = df['stellar_radius'] / df['orbital_period']
        
        # Habitable zone indicators
        if 'stellar_irradiance' in df.columns:
            df['in_habitable_zone'] = ((df['stellar_irradiance'] >= 0.36) & 
                                     (df['stellar_irradiance'] <= 1.67)).astype(int)
            df['habitable_zone_distance'] = np.abs(np.log(df['stellar_irradiance']))
        
        # Planet-star ratios
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planetary_radius'] / df['stellar_radius']
        
        if 'planetary_mass' in df.columns and 'stellar_mass' in df.columns:
            df['mass_ratio'] = df['planetary_mass'] / df['stellar_mass']
        
        # Temperature categories
        if 'stellar_temperature' in df.columns:
            df['stellar_temp_category'] = pd.cut(
                df['stellar_temperature'], 
                bins=[0, 3000, 5000, 6000, 8000, 10000], 
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        
        # Orbital period categories
        if 'orbital_period' in df.columns:
            df['period_category'] = pd.cut(
                df['orbital_period'],
                bins=[0, 1, 10, 100, 1000, float('inf')],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        
        # Size categories
        if 'planetary_radius' in df.columns:
            df['size_category'] = pd.cut(
                df['planetary_radius'],
                bins=[0, 1.25, 2, 4, 8, float('inf')],
                labels=[1, 2, 3, 4, 5]  # Earth, Super-Earth, Neptune, Jupiter, Super-Jupiter
            ).astype(float)
        
        # Mission-specific features
        if 'mission' in df.columns:
            df['is_kepler'] = (df['mission'] == 'Kepler').astype(int)
            df['is_tess'] = (df['mission'] == 'TESS').astype(int)
            df['is_confirmed'] = (df['mission'] == 'Multiple').astype(int)
        
        # Signal strength indicators
        if 'signal_to_noise_ratio' in df.columns:
            df['strong_signal'] = (df['signal_to_noise_ratio'] > 10).astype(int)
            df['log_snr'] = np.log1p(df['signal_to_noise_ratio'])
        
        # Distance-based features
        if 'distance' in df.columns:
            df['log_distance'] = np.log1p(df['distance'])
            df['nearby'] = (df['distance'] < 100).astype(int)
        
        return df
    
    def _intelligent_outlier_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers more intelligently using isolation forest"""
        from sklearn.ensemble import IsolationForest
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        
        if len(numeric_columns) > 0:
            # Use isolation forest for outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[numeric_columns])
            
            # Keep non-outliers
            df = df[outlier_labels == 1]
        
        return df
    
    def train_ensemble_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train ensemble of advanced ML models for maximum accuracy"""
        logger.info("Training ensemble of advanced ML models...")
        
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
        
        # Advanced feature selection
        selector = SelectKBest(f_classif, k=min(20, len(feature_columns)))
        X_selected = selector.fit_transform(X, y_encoded)
        selected_features = feature_columns[selector.get_support()]
        self.feature_selector = selector
        
        logger.info(f"Selected {len(selected_features)} best features")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Advanced scaling
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Advanced sampling for class imbalance
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
        
        # Define ensemble of models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=500, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, class_weight='balanced'
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=500, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=8,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale', 
                class_weight='balanced', probability=True, random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', alpha=0.01, max_iter=500, random_state=42
            )
        }
        
        # Train individual models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Hyperparameter tuning
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [300, 500, 700],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [3, 5, 7]
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train_balanced, y_train_balanced)
                model = grid_search.best_estimator_
                logger.info(f"Best params for {name}: {grid_search.best_params_}")
            
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(selected_features, model.feature_importances_))
                self.feature_importance[name] = importance
            
            # Store model and metrics
            self.trained_models[name] = model
            self.model_performance[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, target_names=le.classes_),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}")
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        ensemble_models = [
            ('rf', self.trained_models['Random Forest']),
            ('et', self.trained_models['Extra Trees']),
            ('gb', self.trained_models['Gradient Boosting']),
            ('svm', self.trained_models['SVM']),
            ('nn', self.trained_models['Neural Network'])
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        self.trained_models['Ensemble'] = ensemble
        self.model_performance['Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'classification_report': classification_report(y_test, y_pred_ensemble, target_names=le.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist(),
            'predictions': y_pred_ensemble.tolist(),
            'probabilities': y_pred_proba_ensemble.tolist()
        }
        
        logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}")
        
        return self.model_performance
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'Ensemble') -> Dict:
        """Predict classification for new exoplanet data"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        # Prepare new data with the same features used in training
        feature_columns = self.feature_selector.feature_names_in_
        
        # Create input data with the required features
        input_data = pd.DataFrame()
        for col in feature_columns:
            if col in new_data.columns:
                input_data[col] = new_data[col]
            else:
                input_data[col] = 0  # Default value for missing features
        
        # Apply feature selection
        X_new_selected = self.feature_selector.transform(input_data)
        
        # Scale features
        X_new_scaled = self.scalers['main'].transform(X_new_selected)
        
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
    
    def save_models(self, save_path: str = "models"):
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
        joblib.dump(self.feature_selector, save_dir / "feature_selector.joblib")
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_path: str = "models"):
        """Load trained models and preprocessors"""
        load_dir = Path(load_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory {load_dir} not found")
        
        # Load models
        model_files = list(load_dir.glob("*_model.joblib"))
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()
            self.trained_models[model_name] = joblib.load(model_file)
        
        # Load other components
        self.scalers = joblib.load(load_dir / "scalers.joblib")
        self.label_encoders = joblib.load(load_dir / "label_encoders.joblib")
        self.feature_importance = joblib.load(load_dir / "feature_importance.joblib")
        self.model_performance = joblib.load(load_dir / "model_performance.joblib")
        self.feature_selector = joblib.load(load_dir / "feature_selector.joblib")
        
        logger.info(f"Models loaded from {load_dir}")

def main():
    """Main function to train high-accuracy models"""
    classifier = HighAccuracyExoplanetClassifier()
    
    # Load and preprocess all NASA data
    df = classifier.load_all_nasa_data()
    
    if df.empty:
        logger.error("No data to train on!")
        return
    
    # Train ensemble models
    performance = classifier.train_ensemble_models(df)
    
    # Save models
    classifier.save_models("high_accuracy_models")
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("HIGH-ACCURACY MODEL PERFORMANCE SUMMARY")
    logger.info("="*60)
    for model_name, metrics in performance.items():
        logger.info(f"{model_name}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Find best model
    best_model = max(performance.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1]['accuracy']*100:.2f}% accuracy!")
    
    # Test prediction
    logger.info("\nTesting prediction...")
    test_data = pd.DataFrame({
        'orbital_period': [365.25],
        'planetary_radius': [1.0],
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
