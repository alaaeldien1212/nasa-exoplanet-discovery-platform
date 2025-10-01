import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ExoplanetDatabase:
    def __init__(self):
        self.db_path = Path("database") / "exoplanets.db"
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure database exists, create if not"""
        if not self.db_path.exists():
            print("Database not found, creating...")
            from scripts.setup_database import create_database, import_nasa_datasets
            create_database()
            import_nasa_datasets()
    
    def find_planet_by_coordinates(self, ra: float, dec: float, tolerance: float = 0.01) -> Optional[Dict]:
        """Find planet by coordinates with tolerance"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search for planets within coordinate tolerance
        cursor.execute('''
            SELECT koi_id, planet_name, ra, dec, period, radius, stellar_temp,
                   duration, depth, equilibrium_temp, stellar_radius, stellar_mass,
                   stellar_luminosity, distance, mission, status, confidence
            FROM exoplanets 
            WHERE ABS(ra - ?) <= ? AND ABS(dec - ?) <= ?
            ORDER BY (ABS(ra - ?) + ABS(dec - ?)) ASC
            LIMIT 1
        ''', (ra, tolerance, dec, tolerance, ra, dec))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'koi_id': result[0],
                'planet_name': result[1],
                'ra': result[2],
                'dec': result[3],
                'period': result[4],
                'radius': result[5],
                'stellar_temp': result[6],
                'duration': result[7],
                'depth': result[8],
                'equilibrium_temp': result[9],
                'stellar_radius': result[10],
                'stellar_mass': result[11],
                'stellar_luminosity': result[12],
                'distance': result[13],
                'mission': result[14],
                'status': result[15],
                'confidence': result[16]
            }
        
        return None
    
    def find_planet_by_koi_id(self, koi_id: str) -> Optional[Dict]:
        """Find planet by KOI ID"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT koi_id, planet_name, ra, dec, period, radius, stellar_temp,
                   duration, depth, equilibrium_temp, stellar_radius, stellar_mass,
                   stellar_luminosity, distance, mission, status, confidence
            FROM exoplanets 
            WHERE koi_id = ?
        ''', (koi_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'koi_id': result[0],
                'planet_name': result[1],
                'ra': result[2],
                'dec': result[3],
                'period': result[4],
                'radius': result[5],
                'stellar_temp': result[6],
                'duration': result[7],
                'depth': result[8],
                'equilibrium_temp': result[9],
                'stellar_radius': result[10],
                'stellar_mass': result[11],
                'stellar_luminosity': result[12],
                'distance': result[13],
                'mission': result[14],
                'status': result[15],
                'confidence': result[16]
            }
        
        return None
    
    def add_prediction_result(self, prediction_data: Dict) -> bool:
        """Add a new prediction result to the database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if planet already exists
            existing = self.find_planet_by_coordinates(
                prediction_data.get('ra', 0), 
                prediction_data.get('dec', 0)
            )
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE exoplanets 
                    SET confidence = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE koi_id = ?
                ''', (prediction_data.get('confidence', 0), existing['koi_id']))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO exoplanets 
                    (koi_id, planet_name, ra, dec, period, radius, stellar_temp,
                     duration, depth, equilibrium_temp, stellar_radius, stellar_mass,
                     stellar_luminosity, distance, mission, status, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_data.get('koi_id'),
                    prediction_data.get('planet_name'),
                    prediction_data.get('ra'),
                    prediction_data.get('dec'),
                    prediction_data.get('period'),
                    prediction_data.get('radius'),
                    prediction_data.get('stellar_temp'),
                    prediction_data.get('duration'),
                    prediction_data.get('depth'),
                    prediction_data.get('equilibrium_temp'),
                    prediction_data.get('stellar_radius'),
                    prediction_data.get('stellar_mass'),
                    prediction_data.get('stellar_luminosity'),
                    prediction_data.get('distance'),
                    prediction_data.get('mission', 'Unknown'),
                    prediction_data.get('status', 'Candidate'),
                    prediction_data.get('confidence', 0)
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding prediction: {e}")
            return False
        finally:
            conn.close()
    
    def get_similar_planets(self, ra: float, dec: float, limit: int = 5) -> List[Dict]:
        """Get similar planets by coordinates"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT koi_id, planet_name, ra, dec, period, radius, stellar_temp,
                   mission, status, confidence,
                   (ABS(ra - ?) + ABS(dec - ?)) as distance
            FROM exoplanets 
            ORDER BY distance ASC
            LIMIT ?
        ''', (ra, dec, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        planets = []
        for result in results:
            planets.append({
                'koi_id': result[0],
                'planet_name': result[1],
                'ra': result[2],
                'dec': result[3],
                'period': result[4],
                'radius': result[5],
                'stellar_temp': result[6],
                'mission': result[7],
                'status': result[8],
                'confidence': result[9],
                'coordinate_distance': result[10]
            })
        
        return planets
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM exoplanets")
        total_count = cursor.fetchone()[0]
        
        # By status
        cursor.execute("SELECT status, COUNT(*) FROM exoplanets GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # By mission
        cursor.execute("SELECT mission, COUNT(*) FROM exoplanets GROUP BY mission")
        mission_counts = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM exoplanets WHERE confidence > 0")
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_planets': total_count,
            'status_distribution': status_counts,
            'mission_distribution': mission_counts,
            'average_confidence': avg_confidence
        }

# Global database instance
db = ExoplanetDatabase()
