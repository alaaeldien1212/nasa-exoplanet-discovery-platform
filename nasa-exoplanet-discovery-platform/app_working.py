#!/usr/bin/env python3
"""
NASA EXOPLANET DISCOVERY PLATFORM - WORKING VERSION
Simple, reliable web application with working ML models
"""

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime
import asyncio
import subprocess
import os
import joblib

# Import our working classifier
from working_classifier import WorkingExoplanetClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NASA Exoplanet Discovery Platform")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
classifier = None
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the ML classifier on startup"""
    global classifier
    logger.info("Starting NASA Exoplanet Discovery Platform...")
    
    classifier = WorkingExoplanetClassifier()
    
    # Try to load existing model
    if classifier.load_model():
        logger.info("Loaded existing working model")
    else:
        logger.info("No existing model found. Training new model...")
        try:
            df = classifier.load_nasa_data()
            if not df.empty:
                classifier.train_model(df)
                classifier.save_model()
                logger.info("Successfully trained and saved new working model")
            else:
                logger.warning("No data available for training")
        except Exception as e:
            logger.error(f"Error training model: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "NASA Exoplanet Discovery Platform"
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard"
    })

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Manual prediction page"""
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "title": "Predict Exoplanet"
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload CSV page"""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Upload CSV"
    })

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    """AI Chatbot page"""
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "title": "AI Chatbot"
    })

@app.get("/api/models/status")
async def get_model_status():
    """Get status of trained models"""
    if classifier is None:
        return {"status": "initializing", "models": []}
    
    if classifier.model is None:
        return {"status": "no_models", "models": []}
    
    return {
        "status": "ready",
        "models": [{
            "name": "Working Random Forest",
            "accuracy": 0.7125,
            "available": True
        }],
        "total_models": 1
    }

@app.post("/api/predict")
async def predict_exoplanet(
    orbital_period: float = Form(...),
    transit_duration: float = Form(...),
    planetary_radius: float = Form(...),
    stellar_temperature: float = Form(...),
    stellar_radius: float = Form(...),
    stellar_mass: float = Form(...),
    distance: float = Form(...)
):
    """Predict exoplanet classification from manual input"""
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Create input data (only use features the model needs)
    input_data = pd.DataFrame({
        'orbital_period': [orbital_period],
        'planetary_radius': [planetary_radius],
        'stellar_temperature': [stellar_temperature],
        'stellar_radius': [stellar_radius]
    })
    
    try:
        # Make prediction
        result = classifier.predict(input_data)
        
        # Generate simple explanation
        prediction = result['predictions'][0]
        confidence = result['confidence'][0]
        
        explanation = f"This exoplanet is classified as {prediction} with {confidence:.1%} confidence. "
        if prediction == "CONFIRMED":
            explanation += "This indicates strong evidence for a real exoplanet."
        elif prediction == "CANDIDATE":
            explanation += "This requires further observation to confirm."
        else:
            explanation += "This appears to be a false positive signal."
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(classifier.class_names, result['probabilities'][0].tolist())),
            "explanation": explanation,
            "model_used": "Working Random Forest",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_and_predict(file: UploadFile = File(...)):
    """Upload CSV file and predict classifications"""
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
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
        
        # Find the header row (first non-comment line)
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                if header_line is None:
                    header_line = i
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
            'koi_duration': 'transit_duration',
            'pl_disposition': 'disposition',
            'koi_disposition': 'disposition'
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
        results = classifier.predict(df)
        
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
            "predictions": results['predictions'].tolist(),
            "confidence": results['confidence'].tolist(),
            "average_confidence": float(np.mean(results['confidence'])),
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
    """Retrain the model with current data"""
    global classifier
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        logger.info("Starting model retraining...")
        
        # Load fresh data
        df = classifier.load_nasa_data()
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available for training")
        
        # Train new model
        accuracy = classifier.train_model(df)
        
        # Save model
        classifier.save_model()
        
        return {
            "message": "Model retrained successfully",
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_ai(message: str = Form(...)):
    """Simple AI chat endpoint"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    # Simple responses based on keywords
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in ['accuracy', 'performance', 'model']):
        return {
            "response": "Our working Random Forest model achieves 71.25% accuracy on exoplanet classification. It performs well on CONFIRMED planets (79% recall) and FALSE POSITIVE detection (77% recall).",
            "timestamp": datetime.now().isoformat()
        }
    elif any(keyword in message_lower for keyword in ['exoplanet', 'planet', 'discovery']):
        return {
            "response": "Exoplanets are planets that orbit stars outside our solar system. Our platform uses machine learning to classify potential exoplanet signals as CONFIRMED, CANDIDATE, or FALSE POSITIVE based on transit data.",
            "timestamp": datetime.now().isoformat()
        }
    elif any(keyword in message_lower for keyword in ['help', 'how', 'what']):
        return {
            "response": "You can upload CSV files with exoplanet data for classification, or use the manual prediction form. The system analyzes orbital period, planetary radius, stellar temperature, and stellar radius to make predictions.",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "response": "I'm here to help with exoplanet discovery and classification. Ask me about model performance, exoplanets, or how to use the platform!",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
