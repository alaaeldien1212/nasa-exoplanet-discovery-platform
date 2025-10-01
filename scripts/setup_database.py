import sqlite3
import pandas as pd
import json
import os
from pathlib import Path

def create_database():
    """Create SQLite database with exoplanet tables"""
    
    # Create database directory
    db_dir = Path("database")
    db_dir.mkdir(exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_dir / "exoplanets.db")
    cursor = conn.cursor()
    
    # Create exoplanets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exoplanets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            koi_id TEXT UNIQUE,
            planet_name TEXT,
            ra REAL,
            dec REAL,
            period REAL,
            radius REAL,
            stellar_temp REAL,
            duration REAL,
            depth REAL,
            equilibrium_temp REAL,
            stellar_radius REAL,
            stellar_mass REAL,
            stellar_luminosity REAL,
            distance REAL,
            discovery_year INTEGER,
            mission TEXT,
            status TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for fast coordinate matching
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coordinates ON exoplanets(ra, dec)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_koi_id ON exoplanets(koi_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_planet_name ON exoplanets(planet_name)')
    
    conn.commit()
    conn.close()
    
    print("Database created successfully!")

def import_nasa_datasets():
    """Import NASA datasets into the database"""
    
    data_dir = Path("nasa data planet")
    db_path = Path("database") / "exoplanets.db"
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found!")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Process each CSV file
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            # Read CSV with NASA-specific handling
            df = pd.read_csv(csv_file, comment='#', on_bad_lines='skip')
            
            # Extract mission from filename
            mission = "Unknown"
            if "cumulative" in csv_file.name.lower():
                mission = "Kepler"
            elif "k2" in csv_file.name.lower():
                mission = "K2"
            elif "toi" in csv_file.name.lower():
                mission = "TESS"
            
            # Process each row
            imported_count = 0
            for idx, row in df.iterrows():
                try:
                    # Extract coordinates and basic info
                    koi_id = str(row.get('kepid', '')) if pd.notna(row.get('kepid')) else None
                    ra = float(row.get('ra', 0)) if pd.notna(row.get('ra')) else None
                    dec = float(row.get('dec', 0)) if pd.notna(row.get('dec')) else None
                    
                    # Skip if no coordinates
                    if not ra or not dec:
                        continue
                    
                    # Extract planet parameters
                    period = float(row.get('koi_period', 0)) if pd.notna(row.get('koi_period')) else None
                    radius = float(row.get('koi_prad', 0)) if pd.notna(row.get('koi_prad')) else None
                    stellar_temp = float(row.get('koi_steff', 0)) if pd.notna(row.get('koi_steff')) else None
                    duration = float(row.get('koi_duration', 0)) if pd.notna(row.get('koi_duration')) else None
                    depth = float(row.get('koi_depth', 0)) if pd.notna(row.get('koi_depth')) else None
                    equilibrium_temp = float(row.get('koi_teq', 0)) if pd.notna(row.get('koi_teq')) else None
                    
                    # Stellar parameters
                    stellar_radius = float(row.get('koi_srad', 0)) if pd.notna(row.get('koi_srad')) else None
                    stellar_mass = float(row.get('koi_smass', 0)) if pd.notna(row.get('koi_smass')) else None
                    stellar_luminosity = float(row.get('koi_slogg', 0)) if pd.notna(row.get('koi_slogg')) else None
                    distance = float(row.get('koi_dist', 0)) if pd.notna(row.get('koi_dist')) else None
                    
                    # Determine status
                    status = "Candidate"
                    if row.get('koi_disposition') == 'CONFIRMED':
                        status = "Confirmed"
                    elif row.get('koi_disposition') == 'FALSE POSITIVE':
                        status = "False Positive"
                    
                    # Generate planet name if confirmed
                    planet_name = None
                    if status == "Confirmed" and koi_id:
                        planet_name = f"Kepler-{koi_id}b" if mission == "Kepler" else f"K2-{koi_id}b" if mission == "K2" else f"TOI-{koi_id}b"
                    
                    # Insert into database
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO exoplanets 
                        (koi_id, planet_name, ra, dec, period, radius, stellar_temp, 
                         duration, depth, equilibrium_temp, stellar_radius, stellar_mass, 
                         stellar_luminosity, distance, mission, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (koi_id, planet_name, ra, dec, period, radius, stellar_temp,
                          duration, depth, equilibrium_temp, stellar_radius, stellar_mass,
                          stellar_luminosity, distance, mission, status))
                    
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
            
            print(f"Imported {imported_count} records from {csv_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print("Dataset import completed!")

def get_database_stats():
    """Get statistics about the database"""
    
    db_path = Path("database") / "exoplanets.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM exoplanets")
    total_count = cursor.fetchone()[0]
    
    # Get counts by status
    cursor.execute("SELECT status, COUNT(*) FROM exoplanets GROUP BY status")
    status_counts = dict(cursor.fetchall())
    
    # Get counts by mission
    cursor.execute("SELECT mission, COUNT(*) FROM exoplanets GROUP BY mission")
    mission_counts = dict(cursor.fetchall())
    
    conn.close()
    
    print(f"\nDatabase Statistics:")
    print(f"Total exoplanets: {total_count}")
    print(f"By status: {status_counts}")
    print(f"By mission: {mission_counts}")

if __name__ == "__main__":
    print("Setting up exoplanet database...")
    create_database()
    import_nasa_datasets()
    get_database_stats()
