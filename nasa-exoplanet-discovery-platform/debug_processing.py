#!/usr/bin/env python3
"""
Debug the data processing step by step
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_processing():
    data_path = Path("../nasa data planet")
    confirmed_file = data_path / 'k2pandc_2025.09.27_12.56.23.csv'
    
    print("Loading data...")
    df = pd.read_csv(confirmed_file, comment='#')
    print(f"Original shape: {df.shape}")
    
    # Check target column
    if 'disposition' in df.columns:
        print(f"Disposition values: {df['disposition'].value_counts()}")
        
        # Map target
        target_mapped = df['disposition'].map({
            'Confirmed': 'CONFIRMED',
            'Candidate': 'CANDIDATE',
            'False Positive': 'FALSE_POSITIVE',
            'Refuted': 'FALSE_POSITIVE'
        })
        print(f"Target mapping result: {target_mapped.value_counts()}")
        
        # Check valid targets
        valid_targets = target_mapped[target_mapped.isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
        print(f"Valid targets: {valid_targets.value_counts()}")
    
    # Check feature columns
    feature_mapping = {
        'pl_orbper': 'orbital_period',
        'pl_rade': 'planetary_radius',
        'st_teff': 'stellar_temperature',
        'st_rad': 'stellar_radius',
        'st_mass': 'stellar_mass',
        'sy_dist': 'distance',
        'pl_insol': 'stellar_irradiance',
        'pl_eqt': 'equilibrium_temperature'
    }
    
    print("\nFeature availability:")
    for new_col, old_col in feature_mapping.items():
        if old_col in df.columns:
            non_null_count = df[old_col].notna().sum()
            print(f"  {old_col} -> {new_col}: {non_null_count} non-null values")
        else:
            print(f"  {old_col} -> {new_col}: NOT FOUND")
    
    # Try to create a simple dataset
    print("\nCreating simple dataset...")
    processed = pd.DataFrame()
    
    # Add target
    if 'disposition' in df.columns:
        processed['target'] = df['disposition'].map({
            'Confirmed': 'CONFIRMED',
            'Candidate': 'CANDIDATE',
            'False Positive': 'FALSE_POSITIVE',
            'Refuted': 'FALSE_POSITIVE'
        })
    
    # Add one simple feature
    if 'pl_orbper' in df.columns:
        processed['orbital_period'] = pd.to_numeric(df['pl_orbper'], errors='coerce')
    
    if 'pl_rade' in df.columns:
        processed['planetary_radius'] = pd.to_numeric(df['pl_rade'], errors='coerce')
    
    print(f"Processed shape: {processed.shape}")
    print(f"Processed columns: {list(processed.columns)}")
    
    # Remove rows without target
    processed = processed.dropna(subset=['target'])
    print(f"After removing null targets: {processed.shape}")
    
    # Filter valid targets
    processed = processed[processed['target'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])]
    print(f"After filtering valid targets: {processed.shape}")
    
    if processed.shape[0] > 0:
        print(f"Final target distribution: {processed['target'].value_counts()}")
        print(f"Sample data:")
        print(processed.head())

if __name__ == "__main__":
    debug_processing()

