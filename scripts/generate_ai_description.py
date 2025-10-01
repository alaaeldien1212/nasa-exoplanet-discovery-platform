import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.llama_analyzer import llama_analyzer

def generate_ai_description():
    """Generate AI description for a single prediction"""
    
    try:
        # Read prediction data from stdin
        input_data = sys.stdin.read()
        prediction_data = json.loads(input_data)
        
        # Generate AI description
        ai_description = llama_analyzer.generate_analysis_description(prediction_data)
        
        # Return result
        result = {
            'success': True,
            'ai_description': ai_description,
            'row': prediction_data.get('row', 0)
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'row': 0
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    generate_ai_description()
