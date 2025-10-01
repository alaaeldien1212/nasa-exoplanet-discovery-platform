import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from lib.predictor import predictor

def main():
    try:
        # Set API mode to suppress print statements
        predictor._api_mode = True
        
        # Get model information
        model_info = predictor.get_model_info()
        
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
