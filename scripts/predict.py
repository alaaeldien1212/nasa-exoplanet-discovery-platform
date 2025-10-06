import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.mission_agnostic_predictor import mission_agnostic_predictor

def main():
    try:
        mission_agnostic_predictor._api_mode = True
        input_data = json.loads(sys.stdin.read())
        
        # Map input features to mission-agnostic names
        mapped_input_data = {}
        feature_mapping = mission_agnostic_predictor.feature_mapping
        
        for key, value in input_data.items():
            if key in feature_mapping:
                mapped_input_data[feature_mapping[key]] = value
            else:
                mapped_input_data[key] = value
        
        result = mission_agnostic_predictor.predict(mapped_input_data)
        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "error": str(e),
            "prediction": None,
            "confidence": None
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()
