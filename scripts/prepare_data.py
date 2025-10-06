import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def prepare_exoplanet_data():
    """Prepare and clean NASA exoplanet datasets for training"""
    
    # Define file paths
    data_dir = Path("nasa data planet")
    cumulative_file = data_dir / "cumulative_2025.09.27_12.55.48.csv"
    k2_file = data_dir / "k2pandc_2025.09.27_12.56.23.csv"
    toi_file = data_dir / "TOI_2025.09.27_12.56.11.csv"
    
    print("Loading NASA exoplanet datasets...")
    
    # Load datasets
    cumulative_df = pd.read_csv(cumulative_file, comment='#')
    k2_df = pd.read_csv(k2_file, comment='#')
    toi_df = pd.read_csv(toi_file, comment='#')
    
    print(f"Cumulative dataset: {len(cumulative_df)} rows")
    print(f"K2 dataset: {len(k2_df)} rows")
    print(f"TOI dataset: {len(toi_df)} rows")
    
    # Process Cumulative dataset (Kepler)
    cumulative_processed = process_cumulative_data(cumulative_df)
    
    # Process K2 dataset
    k2_processed = process_k2_data(k2_df)
    
    # Process TOI dataset (TESS)
    toi_processed = process_toi_data(toi_df)
    
    # Combine all datasets
    combined_data = pd.concat([cumulative_processed, k2_processed, toi_processed], ignore_index=True)
    
    print(f"Combined dataset: {len(combined_data)} rows")
    
    # Clean and prepare features
    final_data = clean_and_prepare_features(combined_data)
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_data.to_csv(output_dir / "exoplanet_training_data.csv", index=False)
    
    # Create training examples for Llama
    create_llama_training_data(final_data, output_dir)
    
    print("Data preparation complete!")
    return final_data

def process_cumulative_data(df):
    """Process cumulative Kepler dataset"""
    processed = df.copy()
    
    # Map disposition to binary classification
    disposition_map = {
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    }
    
    processed['is_exoplanet'] = processed['koi_disposition'].map(disposition_map)
    processed['mission'] = 'Kepler'
    
    # Select relevant features
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_kepmag', 'is_exoplanet', 'mission'
    ]
    
    return processed[feature_columns].dropna()

def process_k2_data(df):
    """Process K2 dataset"""
    processed = df.copy()
    
    # Map disposition
    disposition_map = {
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    }
    
    processed['is_exoplanet'] = processed['disposition'].map(disposition_map)
    processed['mission'] = 'K2'
    
    # Map K2 columns to match Kepler structure
    processed['koi_period'] = processed['pl_orbper']
    
    # Calculate transit duration using orbital period (K2 doesn't have pl_trandurh)
    # Transit duration ≈ (period/π) * (stellar_radius/stellar_distance)^(1/3)
    # For simplicity, use a formula based on orbital period
    processed['koi_duration'] = processed['pl_orbper'] ** (1/3) * 0.5  # Hours
    
    # Calculate transit depth using planetary radius and stellar radius (K2 doesn't have pl_trandep)
    # Transit depth = (planetary_radius/stellar_radius)^2 * 1e6 (in ppm)
    processed['koi_depth'] = ((processed['pl_rade'] / processed['st_rad']) ** 2) * 1e6  # ppm
    
    processed['koi_prad'] = processed['pl_rade']
    processed['koi_teq'] = processed.get('pl_eqt', np.nan)
    processed['koi_insol'] = processed.get('pl_insol', np.nan)
    
    # Calculate signal-to-noise ratio for K2 (estimate based on transit depth and stellar properties)
    # SNR ≈ sqrt(transit_depth) * stellar_magnitude_factor
    stellar_mag = processed.get('sy_vmag', 12)  # Default magnitude
    processed['koi_model_snr'] = np.sqrt(processed['koi_depth']) * (15 - stellar_mag) / 3  # Estimated SNR
    
    processed['koi_steff'] = processed['st_teff']
    processed['koi_slogg'] = processed['st_logg']
    processed['koi_srad'] = processed['st_rad']
    processed['koi_kepmag'] = processed.get('sy_vmag', np.nan)
    
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_kepmag', 'is_exoplanet', 'mission'
    ]
    
    return processed[feature_columns].dropna()

def process_toi_data(df):
    """Process TOI dataset"""
    processed = df.copy()
    
    # Map TFOPWG disposition
    disposition_map = {
        'CP': 1,  # Confirmed Planet
        'PC': 1,  # Planet Candidate
        'FP': 0,  # False Positive
        'KP': 0   # Known Planet (already confirmed elsewhere)
    }
    
    processed['is_exoplanet'] = processed['tfopwg_disp'].map(disposition_map)
    processed['mission'] = 'TESS'
    
    # Map TOI columns
    processed['koi_period'] = processed['pl_orbper']
    
    # Use TOI transit duration if available, otherwise calculate
    if 'pl_trandurh' in processed.columns:
        processed['koi_duration'] = processed['pl_trandurh']
    else:
        processed['koi_duration'] = processed['pl_orbper'] ** (1/3) * 0.5  # Hours
    
    # Use TOI transit depth if available, otherwise calculate
    if 'pl_trandep' in processed.columns:
        processed['koi_depth'] = processed['pl_trandep']
    else:
        processed['koi_depth'] = ((processed['pl_rade'] / processed['st_rad']) ** 2) * 1e6  # ppm
    
    processed['koi_prad'] = processed['pl_rade']
    processed['koi_teq'] = processed.get('pl_eqt', np.nan)
    processed['koi_insol'] = processed.get('pl_insol', np.nan)
    
    # Calculate signal-to-noise ratio for TESS (estimate)
    stellar_mag = processed.get('st_tmag', 12)  # TESS magnitude
    processed['koi_model_snr'] = np.sqrt(processed['koi_depth']) * (15 - stellar_mag) / 3  # Estimated SNR
    
    processed['koi_steff'] = processed['st_teff']
    processed['koi_slogg'] = processed['st_logg']
    processed['koi_srad'] = processed['st_rad']
    processed['koi_kepmag'] = processed['st_tmag']
    
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_kepmag', 'is_exoplanet', 'mission'
    ]
    
    return processed[feature_columns].dropna()

def clean_and_prepare_features(df):
    """Clean and prepare features for training"""
    cleaned = df.copy()
    
    # Remove rows with missing critical features
    critical_features = ['koi_period', 'koi_prad', 'koi_steff']
    cleaned = cleaned.dropna(subset=critical_features)
    
    # Fill missing values with median for numerical features
    numerical_features = [
        'koi_duration', 'koi_depth', 'koi_teq', 'koi_insol',
        'koi_model_snr', 'koi_slogg', 'koi_srad', 'koi_kepmag'
    ]
    
    for feature in numerical_features:
        if feature in cleaned.columns:
            cleaned[feature] = cleaned[feature].fillna(cleaned[feature].median())
    
    # Remove outliers (values beyond 3 standard deviations)
    for feature in numerical_features:
        if feature in cleaned.columns:
            mean = cleaned[feature].mean()
            std = cleaned[feature].std()
            cleaned = cleaned[abs(cleaned[feature] - mean) <= 3 * std]
    
    # Add derived features
    cleaned['period_log'] = np.log10(cleaned['koi_period'])
    cleaned['radius_log'] = np.log10(cleaned['koi_prad'])
    cleaned['temperature_log'] = np.log10(cleaned['koi_teq'])
    
    # Encode mission as categorical
    mission_encoded = pd.get_dummies(cleaned['mission'], prefix='mission')
    cleaned = pd.concat([cleaned, mission_encoded], axis=1)
    
    print(f"Final cleaned dataset: {len(cleaned)} rows")
    print(f"Exoplanet ratio: {cleaned['is_exoplanet'].mean():.3f}")
    
    return cleaned

def create_llama_training_data(df, output_dir):
    """Create training data formatted for Llama fine-tuning"""
    
    training_examples = []
    
    for idx, row in df.iterrows():
        # Create a descriptive text about the exoplanet candidate
        features_text = f"""
Orbital Period: {row['koi_period']:.2f} days
Transit Duration: {row['koi_duration']:.2f} hours
Transit Depth: {row['koi_depth']:.2f} ppm
Planetary Radius: {row['koi_prad']:.2f} Earth radii
Equilibrium Temperature: {row['koi_teq']:.0f} K
Stellar Effective Temperature: {row['koi_steff']:.0f} K
Stellar Surface Gravity: {row['koi_slogg']:.2f}
Stellar Radius: {row['koi_srad']:.2f} Solar radii
Mission: {row['mission']}
"""
        
        # Determine classification
        classification = "exoplanet" if row['is_exoplanet'] == 1 else "false positive"
        
        # Create training example
        example = {
            "instruction": "Analyze the following exoplanet candidate data and determine if it's a confirmed exoplanet or false positive:",
            "input": features_text.strip(),
            "output": f"This is a {classification}. The data shows characteristics consistent with {'a genuine exoplanet' if row['is_exoplanet'] == 1 else 'a false positive signal'}."
        }
        
        training_examples.append(example)
    
    # Save training data
    with open(output_dir / "llama_training_data.json", 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    # Also create a JSONL format for easier processing
    with open(output_dir / "llama_training_data.jsonl", 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(training_examples)} training examples for Llama")

if __name__ == "__main__":
    prepare_exoplanet_data()
