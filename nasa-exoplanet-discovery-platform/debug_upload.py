#!/usr/bin/env python3
"""
Debug the upload issue
"""

import pandas as pd
import io
from working_classifier import WorkingExoplanetClassifier

def test_upload():
    """Test the upload process step by step"""
    print("Testing upload process...")
    
    # Load classifier
    classifier = WorkingExoplanetClassifier()
    classifier.load_model()
    
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('/Users/staynza/Desktop/nasa/predictions_20250928_214113.csv', comment='#')
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {list(df.columns)}")
    
    # Check required columns
    required_columns = ['orbital_period', 'planetary_radius', 'stellar_temperature', 'stellar_radius']
    missing_columns = [col for col in required_columns if col not in df.columns]
    print(f"Missing columns: {missing_columns}")
    
    if missing_columns:
        print("ERROR: Missing required columns!")
        return
    
    # Try to predict
    print("Making predictions...")
    try:
        results = classifier.predict(df)
        print("Predictions successful!")
        print(f"Predictions shape: {results['predictions'].shape}")
        print(f"Confidence shape: {results['confidence'].shape}")
        print(f"First 5 predictions: {results['predictions'][:5]}")
        print(f"First 5 confidence: {results['confidence'][:5]}")
        
        # Test JSON serialization
        print("Testing JSON serialization...")
        import json
        test_data = {
            "predictions": results['predictions'].tolist(),
            "confidence": results['confidence'].tolist(),
            "average_confidence": float(results['confidence'].mean())
        }
        json_str = json.dumps(test_data)
        print("JSON serialization successful!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upload()
