#!/usr/bin/env python3
"""
NASA Exoplanet Discovery Platform - Startup Script
Initializes the platform and trains models if needed
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm', 'plotly', 'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user'
            ] + missing_packages)
            logger.info("Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            logger.error("Failed to install dependencies. Please install manually:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_nasa_data():
    """Check if NASA data is available"""
    logger.info("Checking NASA data...")
    
    data_path = Path("../nasa data planet")
    required_files = [
        "cumulative_2025.09.27_12.55.48.csv",
        "TOI_2025.09.27_12.56.11.csv", 
        "k2pandc_2025.09.27_12.56.23.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing NASA data files: {missing_files}")
        logger.info("The platform will work with reduced functionality")
        return False
    
    logger.info("NASA data files found!")
    return True

def check_ollama():
    """Check if Ollama is available for LLM explanations"""
    logger.info("Checking Ollama installation...")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Ollama found: {result.stdout.strip()}")
            
            # Check if NASA Planet Expert model is available
            try:
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True, timeout=30)
                if 'nasa-planet-expert' in result.stdout:
                    logger.info("NASA Planet Expert model is available!")
                    return True
                else:
                    logger.warning("NASA Planet Expert model not found. LLM explanations will not be available.")
                    return False
            except:
                logger.warning("Could not check Ollama models")
                return False
        else:
            logger.warning("Ollama not working properly")
            return False
    except FileNotFoundError:
        logger.warning("Ollama not found. LLM explanations will not be available.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Ollama check timed out")
        return False

def train_models():
    """Train ML models if they don't exist"""
    logger.info("Training ML models...")
    
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.glob("*.joblib")):
        logger.info("Trained models already exist")
        return True
    
    try:
        from ml_classifier import ExoplanetClassifier
        
        classifier = ExoplanetClassifier()
        df = classifier.load_and_preprocess_data()
        
        if df.empty:
            logger.error("No data available for training")
            return False
        
        logger.info(f"Training models with {len(df)} records...")
        classifier.train_models(df)
        classifier.save_models()
        
        logger.info("Models trained successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        return False

def start_platform():
    """Start the FastAPI platform"""
    logger.info("Starting NASA Exoplanet Discovery Platform...")
    
    try:
        # Start the FastAPI server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 'app:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
    except KeyboardInterrupt:
        logger.info("Platform stopped by user")
    except Exception as e:
        logger.error(f"Failed to start platform: {e}")

def main():
    """Main startup function"""
    logger.info("=" * 60)
    logger.info("NASA Exoplanet Discovery Platform - Startup")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Check NASA data
    nasa_data_available = check_nasa_data()
    
    # Check Ollama
    ollama_available = check_ollama()
    
    # Train models if needed
    if nasa_data_available:
        if not train_models():
            logger.error("Model training failed")
            sys.exit(1)
    else:
        logger.warning("Skipping model training due to missing NASA data")
    
    # Print startup summary
    logger.info("=" * 60)
    logger.info("STARTUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Dependencies: Installed")
    logger.info(f"{'✅' if nasa_data_available else '❌'} NASA Data: {'Available' if nasa_data_available else 'Missing'}")
    logger.info(f"{'✅' if ollama_available else '❌'} Ollama LLM: {'Available' if ollama_available else 'Not Available'}")
    logger.info(f"✅ Models: {'Trained' if nasa_data_available else 'Not Trained'}")
    logger.info("=" * 60)
    
    if nasa_data_available:
        logger.info("🚀 Starting web platform...")
        logger.info("📱 Open your browser to: http://localhost:8000")
        logger.info("🛑 Press Ctrl+C to stop the platform")
        logger.info("=" * 60)
        
        start_platform()
    else:
        logger.error("Cannot start platform without NASA data")
        logger.info("Please ensure NASA data files are in '../nasa data planet/' directory")
        sys.exit(1)

if __name__ == "__main__":
    main()

