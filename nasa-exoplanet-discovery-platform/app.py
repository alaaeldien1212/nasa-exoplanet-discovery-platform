#!/usr/bin/env python3
"""
NASA Exoplanet Discovery Platform - Web Application
FastAPI-based web interface for exoplanet classification and analysis
"""

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import io
import logging
from typing import List, Dict, Optional
from pathlib import Path
import uvicorn
from datetime import datetime
import asyncio
import subprocess
import os
import joblib

# Import our ML classifier
from final_high_accuracy_classifier import FinalHighAccuracyClassifier
from fixed_high_accuracy_classifier import FixedHighAccuracyClassifier
from enhanced_chatbot import EnhancedExoplanetChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NASA Exoplanet Discovery Platform",
    description="AI-powered exoplanet classification and discovery platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
static_dir = Path("static")
templates_dir = Path("templates")
models_dir = Path("models")
uploads_dir = Path("uploads")

for directory in [static_dir, templates_dir, models_dir, uploads_dir]:
    directory.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global classifier instance
classifier = None
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ML classifier and chatbot on startup"""
    global classifier, chatbot
    logger.info("Starting NASA Exoplanet Discovery Platform...")
    
    classifier = FixedHighAccuracyClassifier()
    
    # Initialize enhanced chatbot with Ollama integration
    chatbot = EnhancedExoplanetChatbot(classifier)
    logger.info("Enhanced AI Chatbot with Ollama LLM initialized")
    
    # Try to load existing models
    try:
        # Check if fixed high accuracy models exist
        fixed_models_dir = Path("fixed_high_accuracy_models")
        if fixed_models_dir.exists() and (fixed_models_dir / "fixed_random_forest_model.joblib").exists():
            # Load all available models
            model_files = {
                'Fixed Random Forest': 'fixed_random_forest_model.joblib',
                'Fixed Gradient Boosting': 'fixed_gradient_boosting_model.joblib',
                'Fixed Extra Trees': 'fixed_extra_trees_model.joblib',
                'Fixed SVM': 'fixed_svm_model.joblib',
                'Fixed Histogram GB': 'fixed_histogram_gb_model.joblib',
                'Fixed Ensemble': 'fixed_ensemble_model.joblib'
            }
            
            loaded_models = []
            for model_name, model_file in model_files.items():
                model_path = fixed_models_dir / model_file
                if model_path.exists():
                    try:
                        classifier.trained_models[model_name] = joblib.load(model_path)
                        loaded_models.append(model_name)
                        logger.info(f"Loaded {model_name} model")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
            
            # Load shared components
            classifier.scalers = {'main': joblib.load(fixed_models_dir / "scaler.joblib")}
            classifier.label_encoders = joblib.load(fixed_models_dir / "label_encoders.joblib")
            classifier.feature_importance = joblib.load(fixed_models_dir / "feature_importance.joblib")
            classifier.model_performance = joblib.load(fixed_models_dir / "model_performance.joblib")
            
            logger.info(f"Loaded {len(loaded_models)} FIXED HIGH-ACCURACY models: {', '.join(loaded_models)}")
        else:
            logger.info("No existing final high-accuracy models found. Training new models...")
            # Train models with available data
            try:
                df = classifier.load_all_nasa_data_robust()
                if not df.empty:
                    classifier.train_final_high_accuracy_models(df)
                    classifier.save_models()
                    logger.info("Successfully trained and saved new FINAL HIGH-ACCURACY models")
                else:
                    logger.warning("No data available for training")
            except Exception as e:
                logger.error(f"Error training models: {e}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "NASA Exoplanet Discovery Platform"
    })

@app.get("/api/models/status")
async def get_model_status():
    """Get status of trained models"""
    if classifier is None:
        return {"status": "initializing", "models": []}
    
    if not classifier.trained_models:
        return {"status": "no_models", "models": []}
    
    models_info = []
    for name, performance in classifier.model_performance.items():
        models_info.append({
            "name": name,
            "accuracy": performance.get("accuracy", 0),
            "available": name in classifier.trained_models
        })
    
    return {
        "status": "ready",
        "models": models_info,
        "total_models": len(classifier.trained_models)
    }

@app.post("/api/predict")
async def predict_exoplanet(
    orbital_period: float = Form(...),
    transit_duration: float = Form(...),
    planetary_radius: float = Form(...),
    stellar_temperature: float = Form(...),
    stellar_radius: float = Form(...),
    stellar_mass: float = Form(...),
    distance: float = Form(...),
    model_name: str = Form("Fixed Random Forest")
):
    """Predict exoplanet classification from manual input"""
    if classifier is None or not classifier.trained_models:
        raise HTTPException(status_code=503, detail="Models not ready")
    
    # Create input data
    input_data = pd.DataFrame({
        'orbital_period': [orbital_period],
        'transit_duration': [transit_duration],
        'planetary_radius': [planetary_radius],
        'stellar_temperature': [stellar_temperature],
        'stellar_radius': [stellar_radius],
        'stellar_mass': [stellar_mass],
        'distance': [distance]
    })
    
    try:
        # Make prediction
        result = classifier.predict_new_data(input_data, model_name)
        
        # Get LLM explanation if available
        explanation = await get_llm_explanation(
            orbital_period, planetary_radius, stellar_temperature, 
            result['predictions'][0], result['confidence'][0]
        )
        
        return {
            "prediction": result['predictions'][0],
            "confidence": result['confidence'][0],
            "probabilities": dict(zip(classifier.label_encoders['target'].classes_, result['probabilities'][0])),
            "explanation": explanation,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_and_predict(file: UploadFile = File(...), model_name: str = Form("Fixed Random Forest")):
    """Upload CSV file and predict classifications"""
    if classifier is None or not classifier.trained_models:
        raise HTTPException(status_code=503, detail="Models not ready")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read uploaded file
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # Handle NASA CSV files with metadata headers
        lines = content_str.split('\n')
        header_line = None
        data_start_line = None
        
        # Find the header row (first non-comment line)
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                if header_line is None:
                    header_line = i
                    data_start_line = i
                    break
        
        if header_line is None:
            raise HTTPException(status_code=400, detail="No valid header found in CSV file")
        
        # Read CSV starting from the header row
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_line:])), comment='#')
        
        # Map NASA column names to our expected column names
        column_mapping = {
            'pl_orbper': 'orbital_period',
            'koi_period': 'orbital_period',
            'pl_rade': 'planetary_radius', 
            'koi_prad': 'planetary_radius',
            'st_teff': 'stellar_temperature',
            'koi_steff': 'stellar_temperature',
            'st_rad': 'stellar_radius',
            'koi_srad': 'stellar_radius',
            'st_mass': 'stellar_mass',
            'koi_smass': 'stellar_mass',
            'sy_dist': 'distance',
            'st_dist': 'distance',
            'pl_insol': 'stellar_irradiance',
            'koi_insol': 'stellar_irradiance',
            'pl_eqt': 'equilibrium_temperature',
            'koi_teq': 'equilibrium_temperature',
            'koi_duration': 'transit_duration'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Validate required columns (only what the model needs)
        required_columns = ['orbital_period', 'planetary_radius', 'stellar_temperature', 'stellar_radius']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Available columns: {available_columns[:10]}..."
            )
        
        # Make predictions
        results = classifier.predict_new_data(df, model_name)
        
        # Add predictions to dataframe
        df['predicted_classification'] = results['predictions']
        df['confidence'] = results['confidence']
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"predictions_{timestamp}.csv"
        output_path = uploads_dir / output_filename
        df.to_csv(output_path, index=False)
        
        return {
            "message": "Predictions completed successfully",
            "total_records": len(df),
            "predictions": results['predictions'],
            "confidence": results['confidence'],
            "average_confidence": np.mean(results['confidence']),
            "output_file": output_filename,
            "download_url": f"/download/{output_filename}"
        }
        
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download prediction results"""
    file_path = uploads_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )

@app.get("/api/retrain")
async def retrain_models():
    """Retrain models with current data"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        # Load and preprocess data
        df = classifier.load_and_preprocess_data()
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available for training")
        
        # Train models
        performance = classifier.train_models(df)
        
        # Save models
        classifier.save_models()
        
        return {
            "message": "Models retrained successfully",
            "performance": {name: {"accuracy": metrics["accuracy"]} for name, metrics in performance.items()},
            "training_data_size": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hyperparameter-tuning")
async def hyperparameter_tuning(
    model_name: str = Form("Fixed Random Forest"),
    max_depth: int = Form(7),
    n_estimators: int = Form(200),
    learning_rate: float = Form(0.1)
):
    """Perform hyperparameter tuning"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        # Load data for tuning
        df = classifier.load_and_preprocess_data()
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available for tuning")
        
        # Perform tuning
        tuning_results = classifier.hyperparameter_tuning(df, model_name)
        
        return {
            "message": "Hyperparameter tuning completed",
            "best_params": tuning_results["best_params"],
            "best_score": tuning_results["best_score"],
            "test_accuracy": tuning_results["test_accuracy"],
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance for trained models"""
    if classifier is None or not classifier.feature_importance:
        raise HTTPException(status_code=503, detail="No feature importance data available")
    
    return {
        "feature_importance": classifier.feature_importance,
        "models": list(classifier.feature_importance.keys())
    }

@app.get("/api/model-performance")
async def get_model_performance():
    """Get detailed model performance metrics"""
    if classifier is None or not classifier.model_performance:
        raise HTTPException(status_code=503, detail="No performance data available")
    
    return {
        "performance": classifier.model_performance,
        "models": list(classifier.model_performance.keys())
    }

async def get_llm_explanation(
    orbital_period: float,
    planetary_radius: float,
    stellar_temperature: float,
    prediction: str,
    confidence: float
) -> str:
    """Get explanation from the trained NASA Planet Expert LLM"""
    try:
        # Create a prompt for the LLM
        prompt = f"""
        Based on these planetary characteristics:
        - Orbital Period: {orbital_period} days
        - Planetary Radius: {planetary_radius} Earth radii
        - Stellar Temperature: {stellar_temperature} K
        
        Our AI model classified this as: {prediction} (confidence: {confidence:.2f})
        
        Please provide a scientific explanation for this classification, including:
        1. What these parameters tell us about the planet
        2. Why this classification makes sense
        3. Potential implications for habitability
        4. How this compares to known exoplanets
        """
        
        # Call Ollama API
        result = subprocess.run([
            'ollama', 'run', 'nasa-planet-expert', prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "LLM explanation not available at this time."
            
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        return "LLM explanation not available at this time."

@app.get("/api/llm/explain")
async def explain_planet(
    planet_name: str,
    orbital_period: Optional[float] = None,
    planetary_radius: Optional[float] = None,
    stellar_temperature: Optional[float] = None
):
    """Get detailed explanation about a specific planet from the LLM"""
    try:
        if planet_name:
            prompt = f"Tell me everything about the exoplanet {planet_name}"
        else:
            prompt = f"""
            Explain this exoplanet:
            - Orbital Period: {orbital_period} days
            - Planetary Radius: {planetary_radius} Earth radii
            - Stellar Temperature: {stellar_temperature} K
            """
        
        result = subprocess.run([
            'ollama', 'run', 'nasa-planet-expert', prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return {
                "explanation": result.stdout.strip(),
                "planet": planet_name,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
            
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Analytics dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Analytics Dashboard"
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Data upload page"""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Upload Data"
    })

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Manual prediction page"""
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "title": "Predict Exoplanet"
    })

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    """AI Chatbot page"""
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "title": "AI Assistant"
    })

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chatbot API endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not ready")
    
    try:
        body = await request.json()
        message = body.get("message", "")
        chat_history = body.get("chat_history", [])
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Process the query
        result = chatbot.process_query(message, chat_history)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
