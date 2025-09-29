#!/usr/bin/env python3
"""
Ollama Training Script for NASA Planet Data
Trains a fine-tuned model using NASA exoplanet data
"""

import json
import subprocess
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaNASAPlanetTrainer:
    def __init__(self, training_data_path: str = "training_data/nasa_planet_training_data.json"):
        self.training_data_path = Path(training_data_path)
        self.model_name = "nasa-planet-expert"
        self.base_model = "llama3.2:3b"  # Using a smaller model for fine-tuning
        self.output_dir = Path("models")
        
    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Ollama is installed: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Ollama check failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Ollama version check timed out")
            return False
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            return False
    
    def check_base_model(self) -> bool:
        """Check if the base model is available"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                models = result.stdout
                if self.base_model in models:
                    logger.info(f"Base model {self.base_model} is available")
                    return True
                else:
                    logger.info(f"Base model {self.base_model} not found. Pulling...")
                    return self.pull_base_model()
            else:
                logger.error(f"Failed to list models: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Model check timed out")
            return False
    
    def pull_base_model(self) -> bool:
        """Pull the base model"""
        try:
            logger.info(f"Pulling base model {self.base_model}...")
            result = subprocess.run(['ollama', 'pull', self.base_model], 
                                  capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info(f"Successfully pulled {self.base_model}")
                return True
            else:
                logger.error(f"Failed to pull base model: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Model pull timed out")
            return False
    
    def convert_training_data_to_ollama_format(self) -> Optional[Path]:
        """Convert training data to Ollama Modelfile format"""
        if not self.training_data_path.exists():
            logger.error(f"Training data file not found: {self.training_data_path}")
            return None
        
        logger.info("Converting training data to Ollama format...")
        
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Create Modelfile
        modelfile_path = Path("Modelfile")
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(f"FROM {self.base_model}\n\n")
            f.write("# NASA Planet Expert Model\n")
            f.write("# Fine-tuned for exoplanet discovery and analysis\n\n")
            f.write("PARAMETER temperature 0.7\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write("PARAMETER top_k 40\n\n")
            
            f.write("# System prompt for NASA planet expertise\n")
            f.write("SYSTEM \"\"\"\n")
            f.write("You are a NASA exoplanet expert with comprehensive knowledge of discovered planets, ")
            f.write("their physical properties, discovery methods, and potential for habitability. ")
            f.write("You can analyze planetary data, explain discovery techniques, and assess ")
            f.write("the potential for life on exoplanets. Always provide accurate, scientific ")
            f.write("information based on NASA's exoplanet archive data.\n\"\"\"\n\n")
            
            # Add training examples
            f.write("# Training examples\n")
            for i, example in enumerate(training_data[:100]):  # Limit to first 100 examples
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                
                if instruction and output:
                    f.write(f"TEMPLATE \"\"\"\n")
                    f.write(f"### Instruction:\n{instruction}\n\n")
                    f.write(f"### Response:\n{output}\n")
                    f.write(f"\"\"\"\n\n")
        
        logger.info(f"Created Modelfile with {min(len(training_data), 100)} examples")
        return modelfile_path
    
    def create_ollama_training_script(self) -> Path:
        """Create a shell script for Ollama training"""
        script_path = Path("train_ollama_model.sh")
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# NASA Planet Expert Model Training Script\n")
            f.write("# This script trains a custom Ollama model using NASA exoplanet data\n\n")
            
            f.write("echo 'Starting NASA Planet Expert Model Training...'\n\n")
            
            f.write("# Check if Ollama is running\n")
            f.write("if ! pgrep -x 'ollama' > /dev/null; then\n")
            f.write("    echo 'Starting Ollama service...'\n")
            f.write("    ollama serve &\n")
            f.write("    sleep 5\n")
            f.write("fi\n\n")
            
            f.write("# Create the model\n")
            f.write(f"echo 'Creating model {self.model_name}...'\n")
            f.write(f"ollama create {self.model_name} -f Modelfile\n\n")
            
            f.write("# Test the model\n")
            f.write(f"echo 'Testing model {self.model_name}...'\n")
            f.write(f"ollama run {self.model_name} 'Tell me about the planet Kepler-452b.'\n\n")
            
            f.write("echo 'Training complete!'\n")
            f.write(f"echo 'You can now use the model with: ollama run {self.model_name}'\n")
        
        # Make script executable
        script_path.chmod(0o755)
        logger.info(f"Created training script: {script_path}")
        return script_path
    
    def create_python_training_script(self) -> Path:
        """Create a Python script for advanced training with custom parameters"""
        script_path = Path("advanced_training.py")
        
        with open(script_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Advanced Ollama Training Script for NASA Planet Data
Provides more control over training parameters and evaluation
"""

import subprocess
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedOllamaTrainer:
    def __init__(self):
        self.model_name = "nasa-planet-expert"
        self.training_data_path = Path("training_data/nasa_planet_training_data.json")
        
    def create_custom_modelfile(self, num_examples: int = 50, temperature: float = 0.7):
        """Create a custom Modelfile with specific parameters"""
        modelfile_path = Path("CustomModelfile")
        
        with open(self.training_data_path, 'r') as f:
            training_data = json.load(f)
        
        with open(modelfile_path, 'w') as f:
            f.write("FROM llama3.2:3b\\n\\n")
            f.write("# NASA Planet Expert - Custom Training\\n")
            f.write(f"PARAMETER temperature {temperature}\\n")
            f.write("PARAMETER top_p 0.9\\n")
            f.write("PARAMETER top_k 40\\n")
            f.write("PARAMETER repeat_penalty 1.1\\n\\n")
            
            f.write("SYSTEM \\"\\"\\"\\n")
            f.write("You are a NASA exoplanet expert specializing in planetary discovery, ")
            f.write("characterization, and habitability assessment. You have access to ")
            f.write("comprehensive data from NASA's Exoplanet Archive and can provide ")
            f.write("detailed scientific analysis of exoplanets.\\n\\"\\"\\"\\n\\n")
            
            # Add training examples
            for example in training_data[:num_examples]:
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                
                if instruction and output:
                    f.write(f"TEMPLATE \\"\\"\\"\\n")
                    f.write(f"### Instruction:\\n{instruction}\\n\\n")
                    f.write(f"### Response:\\n{output}\\n")
                    f.write(f"\\"\\"\\"\\n\\n")
        
        logger.info(f"Created custom Modelfile with {num_examples} examples")
        return modelfile_path
    
    def train_model(self, modelfile_path: Path):
        """Train the model using the custom Modelfile"""
        try:
            logger.info(f"Training model {self.model_name}...")
            result = subprocess.run([
                'ollama', 'create', self.model_name, '-f', str(modelfile_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Model training completed successfully!")
                return True
            else:
                logger.error(f"Training failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return False
    
    def test_model(self, test_questions: list):
        """Test the trained model with sample questions"""
        logger.info("Testing the trained model...")
        
        for question in test_questions:
            logger.info(f"Testing: {question}")
            try:
                result = subprocess.run([
                    'ollama', 'run', self.model_name, question
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info(f"Response: {result.stdout[:200]}...")
                else:
                    logger.error(f"Test failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Test timed out")
    
    def evaluate_model_performance(self):
        """Evaluate model performance with test dataset"""
        test_questions = [
            "Tell me about the planet Kepler-452b.",
            "What is the discovery method for TRAPPIST-1b?",
            "Is Proxima Centauri b potentially habitable?",
            "What are the physical properties of HD 209458 b?",
            "When was the first exoplanet discovered?"
        ]
        
        logger.info("Starting model evaluation...")
        self.test_model(test_questions)
    
    def cleanup_old_models(self):
        """Remove old versions of the model"""
        try:
            subprocess.run(['ollama', 'rm', self.model_name], 
                         capture_output=True, text=True)
            logger.info(f"Cleaned up old version of {self.model_name}")
        except:
            logger.info("No old model to clean up")

def main():
    trainer = AdvancedOllamaTrainer()
    
    # Clean up any existing model
    trainer.cleanup_old_models()
    
    # Create custom Modelfile with 100 examples
    modelfile_path = trainer.create_custom_modelfile(num_examples=100, temperature=0.6)
    
    # Train the model
    if trainer.train_model(modelfile_path):
        # Evaluate the model
        trainer.evaluate_model_performance()
        logger.info("Training and evaluation complete!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
''')
        
        script_path.chmod(0o755)
        logger.info(f"Created advanced training script: {script_path}")
        return script_path
    
    def create_evaluation_script(self) -> Path:
        """Create a script to evaluate the trained model"""
        script_path = Path("evaluate_model.py")
        
        with open(script_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
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
''')
        
        script_path.chmod(0o755)
        logger.info(f"Created evaluation script: {script_path}")
        return script_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting NASA Planet Expert Model Training Pipeline")
        
        # Check prerequisites
        if not self.check_ollama_installation():
            logger.error("Ollama is not properly installed or running")
            return False
        
        if not self.check_base_model():
            logger.error("Failed to prepare base model")
            return False
        
        # Convert training data
        modelfile_path = self.convert_training_data_to_ollama_format()
        if not modelfile_path:
            logger.error("Failed to convert training data")
            return False
        
        # Create training scripts
        shell_script = self.create_ollama_training_script()
        python_script = self.create_python_training_script()
        eval_script = self.create_evaluation_script()
        
        logger.info("Training pipeline setup complete!")
        logger.info(f"Created files:")
        logger.info(f"  - {modelfile_path}")
        logger.info(f"  - {shell_script}")
        logger.info(f"  - {python_script}")
        logger.info(f"  - {eval_script}")
        
        logger.info("\\nTo start training, run one of the following:")
        logger.info(f"  ./{shell_script}")
        logger.info(f"  python3 {python_script}")
        
        logger.info("\\nTo evaluate the trained model:")
        logger.info(f"  python3 {eval_script}")
        
        return True

def main():
    """Main function"""
    trainer = OllamaNASAPlanetTrainer()
    
    if trainer.run_training_pipeline():
        logger.info("Training pipeline setup completed successfully!")
    else:
        logger.error("Training pipeline setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

