import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import io
from lib.predictor import predictor
from lib.database import db
from lib.llama_analyzer import llama_analyzer

def parse_nasa_csv(csv_data):
    """Parse NASA CSV data with robust error handling"""
    
    try:
        # Decode the data
        text_data = csv_data.decode('utf-8')
        lines = text_data.split('\n')
        
        # Find the header line (first non-comment line with data)
        header_line = None
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                # Check if this looks like a header (contains common NASA column names)
                if any(col in line.lower() for col in ['kepid', 'koi_', 'pl_', 'st_', 'period', 'radius']):
                    header_line = line
                    data_start = i
                    break
        
        if not header_line:
            raise Exception("Could not find valid header in CSV")
        
        # Extract header columns
        header_cols = [col.strip() for col in header_line.split(',')]
        
        # Extract data lines
        data_lines = []
        for i in range(data_start + 1, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                # Check if line has the right number of columns
                cols = line.split(',')
                if len(cols) == len(header_cols):
                    data_lines.append(line)
                elif len(cols) > len(header_cols):
                    # Try to handle extra columns by truncating
                    data_lines.append(','.join(cols[:len(header_cols)]))
                # Skip lines with too few columns
        
        if not data_lines:
            raise Exception("No valid data rows found")
        
        # Create CSV content
        csv_content = header_line + '\n' + '\n'.join(data_lines)
        
        # Parse with pandas
        df = pd.read_csv(io.StringIO(csv_content))
        
        print(f"Successfully parsed NASA CSV: {len(df)} rows, {len(df.columns)} columns", file=sys.stderr)
        return df
        
    except Exception as e:
        print(f"Error parsing NASA CSV: {e}", file=sys.stderr)
        # Fallback to simple parsing
        try:
            df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')), comment='#', on_bad_lines='skip')
            print(f"Fallback parsing successful: {len(df)} rows", file=sys.stderr)
            return df
        except Exception as e2:
            print(f"Fallback parsing failed: {e2}", file=sys.stderr)
            raise Exception(f"Could not parse CSV file: {e}")

def analyze_uploaded_data():
    """Analyze uploaded CSV file using Llama AI and our trained model"""
    
    try:
        # Set API mode to suppress print statements
        predictor._api_mode = True
        
        # Read CSV data from stdin
        csv_data = sys.stdin.buffer.read()
        
        # Parse CSV with NASA-specific handling
        df = parse_nasa_csv(csv_data)
        
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns", file=sys.stderr)
        
        # Analyze data structure with Llama-like logic
        analysis_result = analyze_data_structure(df)
        
        # Process data for prediction
        processed_data = process_uploaded_data(df)
        
        # Make predictions
        predictions = []
        processed_rows = 0
        
        for idx, row in processed_data.iterrows():
            try:
                # Convert row to dict for prediction
                row_data = row.to_dict()
                
                # Extract coordinates for database lookup
                ra = float(row_data.get('ra', 0)) if pd.notna(row_data.get('ra')) else None
                dec = float(row_data.get('dec', 0)) if pd.notna(row_data.get('dec')) else None
                koi_id = str(row_data.get('kepid', '')) if pd.notna(row_data.get('kepid')) else None
                
                # Check database for existing planet
                existing_planet = None
                if ra and dec:
                    existing_planet = db.find_planet_by_coordinates(ra, dec)
                elif koi_id:
                    existing_planet = db.find_planet_by_koi_id(koi_id)
                
                # Make prediction
                result = predictor.predict(row_data)
                
                if 'error' not in result:
                    # Prepare prediction data
                    prediction_data = {
                        'row': int(idx + 1),
                        'prediction': int(result['prediction']),
                        'classification': str(result['classification']),
                        'confidence': float(result['confidence']),
                        'features': {
                            'koi_period': float(row_data.get('koi_period', 0)) if pd.notna(row_data.get('koi_period')) else 0,
                            'koi_prad': float(row_data.get('koi_prad', 0)) if pd.notna(row_data.get('koi_prad')) else 0,
                            'koi_steff': float(row_data.get('koi_steff', 0)) if pd.notna(row_data.get('koi_steff')) else 0,
                            'koi_duration': float(row_data.get('koi_duration', 0)) if pd.notna(row_data.get('koi_duration')) else 0,
                            'koi_depth': float(row_data.get('koi_depth', 0)) if pd.notna(row_data.get('koi_depth')) else 0,
                            'koi_teq': float(row_data.get('koi_teq', 0)) if pd.notna(row_data.get('koi_teq')) else 0,
                            'ra': ra,
                            'dec': dec,
                            'koi_id': koi_id
                        }
                    }
                    
                    # Add database information if found
                    if existing_planet:
                        prediction_data['database_match'] = {
                            'planet_name': existing_planet.get('planet_name'),
                            'status': existing_planet.get('status'),
                            'mission': existing_planet.get('mission'),
                            'db_confidence': existing_planet.get('confidence'),
                            'is_known_planet': True
                        }
                    else:
                        prediction_data['database_match'] = {
                            'is_known_planet': False
                        }
                        
                        # Add to database if prediction is confident
                        if result['confidence'] > 0.8:
                            db_data = {
                                'koi_id': koi_id,
                                'ra': ra,
                                'dec': dec,
                                'period': float(row_data.get('koi_period', 0)) if pd.notna(row_data.get('koi_period')) else None,
                                'radius': float(row_data.get('koi_prad', 0)) if pd.notna(row_data.get('koi_prad')) else None,
                                'stellar_temp': float(row_data.get('koi_steff', 0)) if pd.notna(row_data.get('koi_steff')) else None,
                                'duration': float(row_data.get('koi_duration', 0)) if pd.notna(row_data.get('koi_duration')) else None,
                                'depth': float(row_data.get('koi_depth', 0)) if pd.notna(row_data.get('koi_depth')) else None,
                                'equilibrium_temp': float(row_data.get('koi_teq', 0)) if pd.notna(row_data.get('koi_teq')) else None,
                                'confidence': float(result['confidence']),
                                'status': 'Confirmed' if result['prediction'] == 1 else 'Candidate'
                            }
                            db.add_prediction_result(db_data)
                    
                    # Generate fast fallback description for speed
                    prediction_data['ai_description'] = llama_analyzer._get_fallback_description(prediction_data)
                    
                    predictions.append(prediction_data)
                    processed_rows += 1
            except Exception as e:
                print(f"Error processing row {idx}: {e}", file=sys.stderr)
                continue
        
        # Calculate summary statistics
        exoplanets = int(sum(1 for p in predictions if p['prediction'] == 1))
        false_positives = int(len(predictions) - exoplanets)
        avg_confidence = float(np.mean([p['confidence'] for p in predictions])) if predictions else 0.0
        
        # Limit predictions for frontend display (keep only first 100 for performance)
        display_predictions = predictions[:100] if len(predictions) > 100 else predictions
        
        # Create result
        result = {
            'filename': 'uploaded_file.csv',
            'totalRows': int(len(df)),
            'processedRows': int(processed_rows),
            'predictions': display_predictions,
            'summary': {
                'exoplanets': exoplanets,
                'falsePositives': false_positives,
                'avgConfidence': avg_confidence
            },
            'analysis': analysis_result,
            'totalPredictions': len(predictions)  # Include total count
        }
        
        # Output result
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "filename": "uploaded_file.csv",
            "totalRows": 0,
            "processedRows": 0,
            "predictions": [],
            "summary": {
                "exoplanets": 0,
                "falsePositives": 0,
                "avgConfidence": 0
            }
        }
        print(json.dumps(error_result))

def analyze_data_structure(df):
    """Analyze data structure using AI-like logic"""
    
    # Convert pandas types to Python types for JSON serialization
    data_types = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    missing_values = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
    
    analysis = {
        'columns': list(df.columns),
        'dataTypes': data_types,
        'missingValues': missing_values,
        'sampleData': df.head(3).fillna('').to_dict('records'),
        'dataQuality': 'Good' if df.isnull().sum().sum() < len(df) * 0.5 else 'Poor',
        'recommendedFeatures': []
    }
    
    # AI-like feature detection
    feature_mapping = {
        'period': ['period', 'orbital_period', 'koi_period', 'pl_orbper'],
        'radius': ['radius', 'planetary_radius', 'koi_prad', 'pl_rade'],
        'temperature': ['temperature', 'stellar_temp', 'koi_steff', 'st_teff'],
        'duration': ['duration', 'transit_duration', 'koi_duration', 'pl_trandurh'],
        'depth': ['depth', 'transit_depth', 'koi_depth', 'pl_trandep'],
        'equilibrium_temp': ['equilibrium_temp', 'koi_teq', 'pl_eqt']
    }
    
    for feature_type, possible_names in feature_mapping.items():
        for col in df.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                analysis['recommendedFeatures'].append({
                    'type': feature_type,
                    'column': col,
                    'sample_values': [float(x) if pd.notna(x) else 0 for x in df[col].dropna().head(3)]
                })
                break
    
    return analysis

def process_uploaded_data(df):
    """Process uploaded data to match our model's expected format"""
    
    processed = df.copy()
    
    # Map common column names to our expected format
    column_mapping = {
        'period': 'koi_period',
        'orbital_period': 'koi_period',
        'pl_orbper': 'koi_period',
        'radius': 'koi_prad',
        'planetary_radius': 'koi_prad',
        'pl_rade': 'koi_prad',
        'temperature': 'koi_steff',
        'stellar_temp': 'koi_steff',
        'st_teff': 'koi_steff',
        'duration': 'koi_duration',
        'transit_duration': 'koi_duration',
        'pl_trandurh': 'koi_duration',
        'depth': 'koi_depth',
        'transit_depth': 'koi_depth',
        'pl_trandep': 'koi_depth',
        'equilibrium_temp': 'koi_teq',
        'pl_eqt': 'koi_teq',
        'insolation': 'koi_insol',
        'pl_insol': 'koi_insol',
        'snr': 'koi_model_snr',
        'signal_to_noise': 'koi_model_snr',
        'surface_gravity': 'koi_slogg',
        'st_logg': 'koi_slogg',
        'stellar_radius': 'koi_srad',
        'st_rad': 'koi_srad',
        'magnitude': 'koi_kepmag',
        'stellar_magnitude': 'koi_kepmag',
        'st_tmag': 'koi_kepmag'
    }
    
    # Rename columns
    for old_name, new_name in column_mapping.items():
        for col in df.columns:
            if old_name.lower() in col.lower():
                processed = processed.rename(columns={col: new_name})
                break
    
    # Add missing columns with default values
    required_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_kepmag'
    ]
    
    for col in required_columns:
        if col not in processed.columns:
            processed[col] = np.nan
    
    # Fill missing values with reasonable defaults
    defaults = {
        'koi_duration': 2.5,
        'koi_depth': 1000,
        'koi_teq': 300,
        'koi_insol': 1.0,
        'koi_model_snr': 10.0,
        'koi_slogg': 4.5,
        'koi_srad': 1.0,
        'koi_kepmag': 12.0
    }
    
    for col, default_val in defaults.items():
        if col in processed.columns:
            processed[col] = processed[col].fillna(default_val)
    
    # Add mission encoding
    processed['mission_Kepler'] = 1
    processed['mission_K2'] = 0
    processed['mission_TESS'] = 0
    
    # Add derived features
    processed['period_log'] = np.log10(processed['koi_period'].fillna(1))
    processed['radius_log'] = np.log10(processed['koi_prad'].fillna(1))
    processed['temperature_log'] = np.log10(processed['koi_teq'].fillna(100))
    
    # Remove rows with missing critical data
    critical_columns = ['koi_period', 'koi_prad', 'koi_steff']
    processed = processed.dropna(subset=critical_columns)
    
    return processed

if __name__ == "__main__":
    analyze_uploaded_data()
