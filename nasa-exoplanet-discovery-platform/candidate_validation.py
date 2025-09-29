#!/usr/bin/env python3
"""
NASA Exoplanet Candidate Validation System
==========================================

This script validates exoplanet candidates by:
1. Testing our AI model against known ground truth data
2. Analyzing prediction confidence and accuracy
3. Identifying potentially new discoveries
4. Providing scientific validation reports
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetValidator:
    def __init__(self, platform_url="http://localhost:8000"):
        self.platform_url = platform_url
        self.validation_results = []
        
    def load_tess_data(self, csv_path):
        """Load TESS data with ground truth labels"""
        logger.info(f"Loading TESS data from {csv_path}")
        
        # Find the header row
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        header_line = None
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                header_line = i
                break
        
        if header_line is None:
            raise ValueError("No valid header found in CSV file")
        
        # Read CSV starting from header
        df = pd.read_csv(csv_path, skiprows=header_line)
        
        # Map NASA column names to our expected format
        column_mapping = {
            'toi': 'toi_id',
            'tfopwg_disp': 'ground_truth',
            'pl_orbper': 'orbital_period',
            'pl_trandurh': 'transit_duration',
            'pl_rade': 'planetary_radius',
            'st_teff': 'stellar_temperature',
            'st_rad': 'stellar_radius',
            'st_dist': 'distance',
            'pl_insol': 'stellar_irradiance',
            'pl_eqt': 'equilibrium_temperature'
        }
        
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Loaded {len(df)} TESS records")
        return df
    
    def predict_candidate(self, row, model_name="Final Gradient Boosting"):
        """Predict classification for a single candidate"""
        try:
            # Prepare prediction data
            prediction_data = {
                'orbital_period': float(row.get('orbital_period', 0)),
                'transit_duration': float(row.get('transit_duration', 0)),
                'planetary_radius': float(row.get('planetary_radius', 0)),
                'stellar_temperature': float(row.get('stellar_temperature', 0)),
                'stellar_radius': float(row.get('stellar_radius', 0)),
                'stellar_mass': 0,  # Default value
                'distance': float(row.get('distance', 0)),
                'model_name': model_name
            }
            
            # Make API call
            response = requests.post(
                f"{self.platform_url}/api/predict",
                data=prediction_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def validate_candidates(self, df, sample_size=20):
        """Validate a sample of candidates against ground truth"""
        logger.info(f"Validating {sample_size} candidates...")
        
        # Sample candidates with different dispositions
        pc_candidates = df[df['ground_truth'] == 'PC'].head(sample_size // 2)
        fp_candidates = df[df['ground_truth'] == 'FP'].head(sample_size // 2)
        
        sample_df = pd.concat([pc_candidates, fp_candidates]).reset_index(drop=True)
        
        results = []
        
        for idx, row in sample_df.iterrows():
            logger.info(f"Validating candidate {idx+1}/{len(sample_df)}: TOI {row.get('toi_id', 'Unknown')}")
            
            prediction_result = self.predict_candidate(row)
            
            if prediction_result:
                # Map our predictions to ground truth format
                our_prediction = prediction_result['prediction']
                ground_truth = row['ground_truth']
                
                # Convert predictions for comparison
                if 'CANDIDATE' in our_prediction or 'CONFIRMED' in our_prediction:
                    our_prediction_mapped = 'PC'
                else:
                    our_prediction_mapped = 'FP'
                
                is_correct = (our_prediction_mapped == ground_truth)
                
                result = {
                    'toi_id': row.get('toi_id', 'Unknown'),
                    'ground_truth': ground_truth,
                    'our_prediction': our_prediction,
                    'our_prediction_mapped': our_prediction_mapped,
                    'confidence': prediction_result['confidence'],
                    'is_correct': is_correct,
                    'orbital_period': row.get('orbital_period', 0),
                    'planetary_radius': row.get('planetary_radius', 0),
                    'stellar_temperature': row.get('stellar_temperature', 0)
                }
                
                results.append(result)
                self.validation_results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Analyze validation results"""
        if not results:
            return {}
        
        df_results = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = df_results['is_correct'].mean()
        
        # Calculate accuracy by class
        pc_accuracy = df_results[df_results['ground_truth'] == 'PC']['is_correct'].mean()
        fp_accuracy = df_results[df_results['ground_truth'] == 'FP']['is_correct'].mean()
        
        # Calculate average confidence
        avg_confidence = df_results['confidence'].mean()
        
        # Find high-confidence candidates
        high_confidence_candidates = df_results[
            (df_results['confidence'] > 0.9) & 
            (df_results['our_prediction_mapped'] == 'PC')
        ]
        
        analysis = {
            'total_candidates': len(results),
            'accuracy': accuracy,
            'pc_accuracy': pc_accuracy,
            'fp_accuracy': fp_accuracy,
            'avg_confidence': avg_confidence,
            'high_confidence_candidates': high_confidence_candidates.to_dict('records'),
            'false_positives': df_results[
                (df_results['ground_truth'] == 'FP') & 
                (df_results['our_prediction_mapped'] == 'PC')
            ].to_dict('records'),
            'false_negatives': df_results[
                (df_results['ground_truth'] == 'PC') & 
                (df_results['our_prediction_mapped'] == 'FP')
            ].to_dict('records')
        }
        
        return analysis
    
    def generate_report(self, analysis):
        """Generate a comprehensive validation report"""
        print("\n" + "="*80)
        print("🔬 NASA EXOPLANET CANDIDATE VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Total Candidates Tested: {analysis['total_candidates']}")
        print(f"   • Overall Accuracy: {analysis['accuracy']:.2%}")
        print(f"   • Planet Candidate Accuracy: {analysis['pc_accuracy']:.2%}")
        print(f"   • False Positive Accuracy: {analysis['fp_accuracy']:.2%}")
        print(f"   • Average Confidence: {analysis['avg_confidence']:.2%}")
        
        print(f"\n🎯 HIGH-CONFIDENCE CANDIDATES (>90% confidence):")
        if analysis['high_confidence_candidates']:
            for candidate in analysis['high_confidence_candidates']:
                print(f"   • TOI {candidate['toi_id']}: {candidate['confidence']:.2%} confidence")
                print(f"     - Orbital Period: {candidate['orbital_period']:.2f} days")
                print(f"     - Planetary Radius: {candidate['planetary_radius']:.2f} Earth radii")
                print(f"     - Stellar Temperature: {candidate['stellar_temperature']:.0f} K")
        else:
            print("   • No high-confidence candidates found")
        
        print(f"\n❌ FALSE POSITIVES (Ground Truth: FP, Our Prediction: PC):")
        if analysis['false_positives']:
            for fp in analysis['false_positives']:
                print(f"   • TOI {fp['toi_id']}: {fp['confidence']:.2%} confidence")
        else:
            print("   • No false positives detected")
        
        print(f"\n❌ FALSE NEGATIVES (Ground Truth: PC, Our Prediction: FP):")
        if analysis['false_negatives']:
            for fn in analysis['false_negatives']:
                print(f"   • TOI {fn['toi_id']}: {fn['confidence']:.2%} confidence")
                print(f"     - This is a CONFIRMED PLANET CANDIDATE that we missed!")
        else:
            print("   • No false negatives detected")
        
        print(f"\n🌟 SCIENTIFIC CONCLUSIONS:")
        if analysis['accuracy'] > 0.8:
            print("   ✅ Model shows high accuracy and is reliable for candidate validation")
        else:
            print("   ⚠️  Model accuracy needs improvement")
        
        if analysis['high_confidence_candidates']:
            print(f"   🎯 Found {len(analysis['high_confidence_candidates'])} high-confidence candidates for follow-up")
        
        print("\n" + "="*80)

def main():
    """Main validation function"""
    validator = ExoplanetValidator()
    
    # Load TESS data
    tess_data_path = Path("../nasa data planet/TOI_2025.09.27_12.56.11.csv")
    
    if not tess_data_path.exists():
        logger.error(f"TESS data file not found: {tess_data_path}")
        return
    
    try:
        df = validator.load_tess_data(tess_data_path)
        
        # Validate candidates
        results = validator.validate_candidates(df, sample_size=20)
        
        # Analyze results
        analysis = validator.analyze_results(results)
        
        # Generate report
        validator.generate_report(analysis)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")

if __name__ == "__main__":
    main()

