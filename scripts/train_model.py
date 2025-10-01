import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path

def train_exoplanet_model():
    """Train machine learning models for exoplanet detection"""
    
    print("Loading processed data...")
    
    # Load processed data
    data_path = Path("data/processed/exoplanet_training_data.csv")
    if not data_path.exists():
        print("Processed data not found. Please run prepare_data.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Prepare features and target
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_kepmag', 'period_log',
        'radius_log', 'temperature_log', 'mission_Kepler', 'mission_K2', 'mission_TESS'
    ]
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df['is_exoplanet']
    
    print(f"Using {len(available_features)} features")
    print(f"Feature names: {available_features}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'test_accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy {best_score:.4f}")
    
    # Save models and scaler
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Save best model
    joblib.dump(best_model, model_dir / "best_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    
    # Save feature names
    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(available_features, f, indent=2)
    
    # Save results
    with open(model_dir / "training_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'cv_scores': result['cv_scores'].tolist(),
                'test_accuracy': result['test_accuracy'],
                'classification_report': result['classification_report']
            }
        json.dump(serializable_results, f, indent=2)
    
    # Create model metadata
    metadata = {
        'best_model': best_name,
        'best_accuracy': best_score,
        'features_used': available_features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'exoplanet_ratio': y.mean()
    }
    
    with open(model_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModels saved to {model_dir}")
    print("Training complete!")
    
    return best_model, scaler, available_features

def create_simple_model():
    """Create a simple model for demonstration purposes"""
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = {
        'koi_period': np.random.lognormal(2, 1, n_samples),
        'koi_duration': np.random.lognormal(1, 0.5, n_samples),
        'koi_depth': np.random.lognormal(6, 1, n_samples),
        'koi_prad': np.random.lognormal(0.5, 0.8, n_samples),
        'koi_teq': np.random.normal(1000, 500, n_samples),
        'koi_steff': np.random.normal(5500, 1000, n_samples),
        'koi_slogg': np.random.normal(4.5, 0.5, n_samples),
        'koi_srad': np.random.lognormal(0, 0.3, n_samples),
        'mission_Kepler': np.random.choice([0, 1], n_samples),
        'mission_K2': np.random.choice([0, 1], n_samples),
        'mission_TESS': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on simple rules
    df['is_exoplanet'] = (
        (df['koi_period'] > 1) & 
        (df['koi_period'] < 1000) &
        (df['koi_prad'] > 0.5) &
        (df['koi_prad'] < 20) &
        (df['koi_depth'] > 100) &
        (df['koi_depth'] < 10000)
    ).astype(int)
    
    # Add some noise
    noise = np.random.random(n_samples) < 0.1
    df.loc[noise, 'is_exoplanet'] = 1 - df.loc[noise, 'is_exoplanet']
    
    print(f"Created synthetic dataset with {len(df)} samples")
    print(f"Exoplanet ratio: {df['is_exoplanet'].mean():.3f}")
    
    # Train simple model
    feature_columns = [col for col in df.columns if col != 'is_exoplanet']
    X = df[feature_columns]
    y = df['is_exoplanet']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, model_dir / "simple_model.pkl")
    
    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    metadata = {
        'model_type': 'RandomForest',
        'features_used': feature_columns,
        'training_samples': len(df),
        'exoplanet_ratio': y.mean(),
        'is_synthetic': True
    }
    
    with open(model_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Simple model created and saved!")
    return model, feature_columns

if __name__ == "__main__":
    try:
        train_exoplanet_model()
    except Exception as e:
        print(f"Error training with real data: {e}")
        print("Creating simple model instead...")
        create_simple_model()
