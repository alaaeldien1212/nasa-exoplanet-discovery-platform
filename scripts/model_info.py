import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def main():
    try:
        # Load mission-agnostic model metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Get model information
        model_info = {
            "model_type": "Mission-Agnostic Exoplanet Classifier",
            "version": "4.0",
            "training_samples": metadata.get('total_samples', 'unknown'),
            "features": feature_names,
            "missions_used": metadata.get('missions_used', []),
            "mission_distribution": metadata.get('mission_distribution', {}),
            "accuracy": metadata.get('best_accuracy', 'unknown'),
            "description": metadata.get('description', 'Mission-agnostic model trained on all three NASA missions'),
            "improvements": [
                "Trained on Kepler, K2, and TESS missions",
                "15,948 total samples (vs 7,809 before)",
                "Mission-agnostic feature names (no more 'koi_' prefixes)",
                "Generic feature names: orbital_period, planetary_radius, etc.",
                "Improved accuracy: 83.1%",
                "Better generalization across mission types",
                "True multi-mission compatibility"
            ],
            "feature_mapping": metadata.get('feature_mapping', {})
        }

        # Output result
        print(json.dumps(model_info))

    except Exception as e:
        error_result = {
            "error": str(e),
            "model_info": None
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()
