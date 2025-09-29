#!/usr/bin/env python3
"""
NASA Exoplanet ML Classifier
Advanced machine learning models for exoplanet classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetClassifier:
    def __init__(self, data_path: str = "../nasa data planet"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess NASA exoplanet data for ML training"""
        logger.info("Loading NASA exoplanet datasets...")
        
        datasets = {}
        csv_files = {
            'cumulative': self.data_path / 'cumulative_2025.09.27_12.55.48.csv',
            'toi': self.data_path / 'TOI_2025.09.27_12.56.11.csv',
            'k2pandc': self.data_path / 'k2pandc_2025.09.27_12.56.23.csv'
        }
        
        for name, file_path in csv_files.items():
            if file_path.exists():
                logger.info(f"Loading {name} dataset...")
                df = pd.read_csv(file_path, comment='#')
                datasets[name] = df
                logger.info(f"Loaded {len(df)} records from {name}")
        
        # Combine and preprocess datasets
        combined_df = self._combine_datasets(datasets)
        processed_df = self._preprocess_features(combined_df)
        
        logger.info(f"Final processed dataset: {len(processed_df)} records")
        return processed_df
    
    def _combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine different NASA datasets into a unified format"""
        combined_data = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            if dataset_name == 'cumulative':
                processed_df = self._process_kepler_data(df)
            elif dataset_name == 'toi':
                processed_df = self._process_tess_data(df)
            elif dataset_name == 'k2pandc':
                processed_df = self._process_confirmed_data(df)
            
            if not processed_df.empty:
                combined_data.append(processed_df)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(final_df)} records")
            return final_df
        else:
            return pd.DataFrame()
    
    def _process_kepler_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Kepler mission data"""
        processed = pd.DataFrame()
        
        # Create target variable based on KOI disposition
        if 'koi_disposition' in df.columns:
            processed['target'] = df['koi_disposition'].map({
                'CONFIRMED': 'CONFIRMED',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'UNKNOWN'
        
        # Extract features
        feature_mapping = {
            'koi_period': 'orbital_period',
            'koi_duration': 'transit_duration',
            'koi_impact': 'impact_parameter',
            'koi_sma': 'semi_major_axis',
            'koi_insol': 'stellar_irradiance',
            'koi_steff': 'stellar_temperature',
            'koi_slogg': 'stellar_surface_gravity',
            'koi_smet': 'stellar_metallicity',
            'koi_srad': 'stellar_radius',
            'koi_smass': 'stellar_mass',
            'koi_kepmag': 'kepler_magnitude',
            'koi_depth': 'transit_depth',
            'koi_prad': 'planetary_radius',
            'koi_model_snr': 'signal_to_noise_ratio'
        }
        
        for new_col, old_col in feature_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        processed['mission'] = 'Kepler'
        return processed
    
    def _process_tess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process TESS mission data"""
        processed = pd.DataFrame()
        
        # Create target variable based on TFOPWG disposition
        if 'tfopwg_disp' in df.columns:
            processed['target'] = df['tfopwg_disp'].map({
                'CP': 'CONFIRMED',
                'PC': 'CANDIDATE',
                'FP': 'FALSE_POSITIVE',
                'KP': 'CANDIDATE'
            })
        else:
            processed['target'] = 'CANDIDATE'  # TOI objects are typically candidates
        
        # Extract TESS-specific features
        feature_mapping = {
            'pl_orbper': 'orbital_period',
            'pl_rade': 'planetary_radius',
            'pl_bmasse': 'planetary_mass',
            'pl_dens': 'planetary_density',
            'st_teff': 'stellar_temperature',
            'st_rad': 'stellar_radius',
            'st_mass': 'stellar_mass',
            'sy_dist': 'distance',
            'pl_eqt': 'equilibrium_temperature',
            'pl_insol': 'stellar_irradiance'
        }
        
        for new_col, old_col in feature_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        processed['mission'] = 'TESS'
        return processed
    
    def _process_confirmed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process confirmed planets data"""
        processed = pd.DataFrame()
        
        # Create target variable
        if 'disposition' in df.columns:
            processed['target'] = df['disposition'].map({
                'Confirmed': 'CONFIRMED',
                'Candidate': 'CANDIDATE',
                'False Positive': 'FALSE_POSITIVE'
            })
        else:
            processed['target'] = 'CONFIRMED'  # Assume confirmed if in this dataset
        
        # Extract comprehensive features
        feature_mapping = {
            'pl_orbper': 'orbital_period',
            'pl_rade': 'planetary_radius',
            'pl_bmasse': 'planetary_mass',
            'pl_dens': 'planetary_density',
            'pl_orbsmax': 'semi_major_axis',
            'pl_eqt': 'equilibrium_temperature',
            'pl_insol': 'stellar_irradiance',
            'st_teff': 'stellar_temperature',
            'st_rad': 'stellar_radius',
            'st_mass': 'stellar_mass',
            'st_met': 'stellar_metallicity',
            'st_logg': 'stellar_surface_gravity',
            'sy_dist': 'distance',
            'sy_snum': 'num_stars',
            'sy_pnum': 'num_planets'
        }
        
        for new_col, old_col in feature_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        # Add discovery method as categorical feature
        if 'discoverymethod' in df.columns:
            processed['discovery_method'] = df['discoverymethod']
        
        processed['mission'] = 'Multiple'
        return processed
    
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for machine learning"""
        logger.info("Preprocessing features...")
        
        # Remove rows without target labels
        df = df.dropna(subset=['target'])
        df = df[df['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns.drop(['target'])
        
        # Fill missing numeric values with median
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        
        # Create additional features
        df = self._create_derived_features(df)
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Final processed dataset: {len(df)} records")
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing ones"""
        # Transit probability
        if 'orbital_period' in df.columns and 'stellar_radius' in df.columns:
            df['transit_probability'] = df['stellar_radius'] / df['orbital_period']
        
        # Habitable zone indicator
        if 'stellar_irradiance' in df.columns:
            df['in_habitable_zone'] = ((df['stellar_irradiance'] >= 0.36) & 
                                     (df['stellar_irradiance'] <= 1.67)).astype(int)
        
        # Planet-star size ratio
        if 'planetary_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planetary_radius'] / df['stellar_radius']
        
        # Temperature category
        if 'stellar_temperature' in df.columns:
            df['stellar_temp_category'] = pd.cut(
                df['stellar_temperature'], 
                bins=[0, 3000, 5000, 6000, 8000, 10000], 
                labels=['M', 'K', 'G', 'F', 'A']
            )
        
        # Orbital period category
        if 'orbital_period' in df.columns:
            df['period_category'] = pd.cut(
                df['orbital_period'],
                bins=[0, 1, 10, 100, 1000, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train multiple ML models for exoplanet classification"""
        logger.info("Training machine learning models...")
        
        # Prepare features and target
        feature_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        X = df[feature_columns]
        y = df['target']
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['target'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
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
                'classification_report': classification_report(y_test, y_pred, target_names=le.classes_),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}")
        
        return self.model_performance
    
    def hyperparameter_tuning(self, df: pd.DataFrame, model_name: str = 'XGBoost') -> Dict:
        """Perform hyperparameter tuning for the best model"""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Prepare data
        feature_columns = df.select_dtypes(include=[np.number]).columns.drop(['target'], errors='ignore')
        X = df[feature_columns]
        y = df['target']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grid based on model
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        # Best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'test_accuracy': test_accuracy,
            'model': best_model
        }
    
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = 'XGBoost') -> Dict:
        """Predict classification for new exoplanet data"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        # Prepare new data
        feature_columns = list(self.scalers['main'].feature_names_in_)
        X_new = new_data[feature_columns]
        
        # Handle missing values
        for col in feature_columns:
            if col not in X_new.columns:
                X_new[col] = 0  # Default value for missing features
        
        # Scale features
        X_new_scaled = self.scalers['main'].transform(X_new[feature_columns])
        
        # Make predictions
        model = self.trained_models[model_name]
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled) if hasattr(model, 'predict_proba') else None
        
        # Decode predictions
        if 'target' in self.label_encoders:
            predictions_decoded = self.label_encoders['target'].inverse_transform(predictions)
        else:
            predictions_decoded = predictions
        
        return {
            'predictions': predictions_decoded.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'confidence': np.max(probabilities, axis=1).tolist() if probabilities is not None else None
        }
    
    def save_models(self, save_path: str = "models"):
        """Save trained models and preprocessors"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = save_dir / f"{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save scalers and encoders
        joblib.dump(self.scalers, save_dir / "scalers.joblib")
        joblib.dump(self.label_encoders, save_dir / "label_encoders.joblib")
        joblib.dump(self.feature_importance, save_dir / "feature_importance.joblib")
        joblib.dump(self.model_performance, save_dir / "model_performance.joblib")
        
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
        
        logger.info(f"Models loaded from {load_dir}")

def main():
    """Main function to train and evaluate models"""
    classifier = ExoplanetClassifier()
    
    # Load and preprocess data
    df = classifier.load_and_preprocess_data()
    
    if df.empty:
        logger.error("No data to train on!")
        return
    
    # Train models
    performance = classifier.train_models(df)
    
    # Hyperparameter tuning for best model
    tuning_results = classifier.hyperparameter_tuning(df, 'XGBoost')
    
    # Save models
    classifier.save_models()
    
    # Print results
    logger.info("\nModel Performance Summary:")
    for model_name, metrics in performance.items():
        logger.info(f"{model_name}: {metrics['accuracy']:.4f}")
    
    logger.info(f"\nBest tuned model accuracy: {tuning_results['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()

