#!/usr/bin/env python3
"""
Model Evaluation Script for NASA Planet Expert
Tests the trained model with various exoplanet-related questions
"""

import subprocess
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_name: str = "nasa-planet-expert"):
        self.model_name = model_name
        self.test_questions = [
            "Tell me about the planet Kepler-452b.",
            "What is the discovery method for TRAPPIST-1b?",
            "Is Proxima Centauri b potentially habitable?",
            "What are the physical properties of HD 209458 b?",
            "When was the first exoplanet discovered?",
            "What is the habitable zone and how is it calculated?",
            "Explain the transit method for exoplanet detection.",
            "What makes a planet potentially habitable?",
            "Tell me about the TRAPPIST-1 planetary system.",
            "What is the difference between a super-Earth and a mini-Neptune?"
        ]
    
    def test_model(self, question: str, timeout: int = 30) -> dict:
        """Test the model with a single question"""
        try:
            start_time = time.time()
            result = subprocess.run([
                'ollama', 'run', self.model_name, question
            ], capture_output=True, text=True, timeout=timeout)
            
            response_time = time.time() - start_time
            
            return {
                'question': question,
                'response': result.stdout.strip() if result.returncode == 0 else None,
                'error': result.stderr if result.returncode != 0 else None,
                'response_time': response_time,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'question': question,
                'response': None,
                'error': 'Timeout',
                'response_time': timeout,
                'success': False
            }
    
    def run_evaluation(self) -> list:
        """Run comprehensive evaluation"""
        logger.info(f"Starting evaluation of model: {self.model_name}")
        
        results = []
        for i, question in enumerate(self.test_questions, 1):
            logger.info(f"Testing question {i}/{len(self.test_questions)}: {question[:50]}...")
            result = self.test_model(question)
            results.append(result)
            
            if result['success']:
                logger.info(f"✓ Success (response time: {result['response_time']:.2f}s)")
            else:
                logger.error(f"✗ Failed: {result['error']}")
            
            time.sleep(1)  # Brief pause between questions
        
        return results
    
    def save_evaluation_results(self, results: list):
        """Save evaluation results to file"""
        output_file = Path("evaluation_results.json")
        
        evaluation_data = {
            'model_name': self.model_name,
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_questions': len(results),
            'successful_responses': sum(1 for r in results if r['success']),
            'average_response_time': sum(r['response_time'] for r in results) / len(results),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
        
        # Print summary
        success_rate = (evaluation_data['successful_responses'] / evaluation_data['total_questions']) * 100
        logger.info(f"Evaluation Summary:")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Average Response Time: {evaluation_data['average_response_time']:.2f}s")
        logger.info(f"  Total Questions: {evaluation_data['total_questions']}")

def main():
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()
    evaluator.save_evaluation_results(results)

if __name__ == "__main__":
    main()
