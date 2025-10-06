import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from lib.enhanced_predictor import enhanced_predictor

def main():
    try:
        # Set API mode to suppress print statements
        enhanced_predictor._api_mode = True
        
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Make prediction
        result = enhanced_predictor.predict(input_data)
        
        # Output result
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
