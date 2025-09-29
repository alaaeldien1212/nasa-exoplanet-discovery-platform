#!/usr/bin/env python3
"""
Debug script to check NASA data structure
"""

import pandas as pd
from pathlib import Path

def debug_data():
    data_path = Path("../nasa data planet")
    
    csv_files = {
        'cumulative': data_path / 'cumulative_2025.09.27_12.55.48.csv',
        'toi': data_path / 'TOI_2025.09.27_12.56.11.csv',
        'k2pandc': data_path / 'k2pandc_2025.09.27_12.56.23.csv'
    }
    
    for name, file_path in csv_files.items():
        if file_path.exists():
            print(f"\n=== {name.upper()} DATASET ===")
            df = pd.read_csv(file_path, comment='#')
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Numeric columns: {list(df.select_dtypes(include=['number']).columns)}")
            
            # Show first few rows
            print("\nFirst 3 rows:")
            print(df.head(3))
            
            # Check for target columns
            target_cols = [col for col in df.columns if 'disp' in col.lower() or 'target' in col.lower()]
            if target_cols:
                print(f"Potential target columns: {target_cols}")
                for col in target_cols:
                    print(f"  {col}: {df[col].value_counts()}")

if __name__ == "__main__":
    debug_data()

