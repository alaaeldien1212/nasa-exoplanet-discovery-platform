#!/usr/bin/env python3
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
            f.write("FROM llama3.2:3b\n\n")
            f.write("# NASA Planet Expert - Custom Training\n")
            f.write(f"PARAMETER temperature {temperature}\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write("PARAMETER top_k 40\n")
            f.write("PARAMETER repeat_penalty 1.1\n\n")
            
            f.write("SYSTEM \"\"\"\n")
            f.write("You are a NASA exoplanet expert specializing in planetary discovery, ")
            f.write("characterization, and habitability assessment. You have access to ")
            f.write("comprehensive data from NASA's Exoplanet Archive and can provide ")
            f.write("detailed scientific analysis of exoplanets.\n\"\"\"\n\n")
            
            # Add training examples
            for example in training_data[:num_examples]:
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                
                if instruction and output:
                    f.write(f"TEMPLATE \"\"\"\n")
                    f.write(f"### Instruction:\n{instruction}\n\n")
                    f.write(f"### Response:\n{output}\n")
                    f.write(f"\"\"\"\n\n")
        
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
