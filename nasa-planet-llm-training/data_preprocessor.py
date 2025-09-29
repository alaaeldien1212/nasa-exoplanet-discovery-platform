#!/usr/bin/env python3
"""
NASA Planet Data Preprocessor for LLM Training
Converts NASA exoplanet CSV data into training format for Ollama
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASAPlanetDataPreprocessor:
    def __init__(self, data_dir: str = "../nasa data planet"):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all NASA CSV files"""
        csv_files = {
            'cumulative': self.data_dir / 'cumulative_2025.09.27_12.55.48.csv',
            'toi': self.data_dir / 'TOI_2025.09.27_12.56.11.csv',
            'k2pandc': self.data_dir / 'k2pandc_2025.09.27_12.56.23.csv'
        }
        
        datasets = {}
        for name, file_path in csv_files.items():
            if file_path.exists():
                logger.info(f"Loading {name} dataset from {file_path}")
                # Skip comment lines and read data
                df = pd.read_csv(file_path, comment='#')
                datasets[name] = df
                logger.info(f"Loaded {len(df)} records from {name}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return datasets
    
    def clean_and_prepare_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Clean and prepare the datasets for training"""
        combined_data = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            # Create standardized columns for each dataset
            if dataset_name == 'cumulative':
                processed_df = self._process_cumulative_data(df)
            elif dataset_name == 'toi':
                processed_df = self._process_toi_data(df)
            elif dataset_name == 'k2pandc':
                processed_df = self._process_k2pandc_data(df)
            
            # Basic cleaning - remove rows without planet names
            if 'planet_name' in processed_df.columns:
                processed_df = processed_df.dropna(subset=['planet_name'])
            
            combined_data.append(processed_df)
        
        # Combine all datasets
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(final_df)} records")
            return final_df
        else:
            return pd.DataFrame()
    
    def _process_cumulative_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cumulative dataset (Kepler data)"""
        processed = pd.DataFrame()
        
        # Create planet names from Kepler data
        if 'kepler_name' in df.columns:
            processed['planet_name'] = df['kepler_name']
        elif 'kepoi_name' in df.columns:
            processed['planet_name'] = df['kepoi_name']
        else:
            processed['planet_name'] = 'Unknown Planet'
        
        # Map available columns to standardized format
        column_mapping = {
            'koi_period': 'orbital_period_days',
            'koi_duration': 'transit_duration_hours',
            'koi_impact': 'impact_parameter',
            'koi_sma': 'semi_major_axis_au',
            'koi_insol': 'stellar_irradiance',
            'koi_steff': 'star_temperature_k',
            'koi_slogg': 'star_surface_gravity',
            'koi_smet': 'star_metallicity',
            'koi_srad': 'star_radius_solar',
            'koi_smass': 'star_mass_solar',
            'koi_kepmag': 'kepler_magnitude'
        }
        
        for new_col, old_col in column_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        processed['discovery_method'] = 'Transit'
        processed['discovery_facility'] = 'Kepler'
        processed['dataset_source'] = 'cumulative'
        return processed
    
    def _process_toi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process TOI (TESS Object of Interest) dataset"""
        processed = pd.DataFrame()
        
        # Create planet names from TOI data
        if 'toi' in df.columns:
            processed['planet_name'] = 'TOI-' + df['toi'].astype(str)
        
        column_mapping = {
            'pl_orbper': 'orbital_period_days',
            'pl_rade': 'planet_radius_earth',
            'pl_bmasse': 'planet_mass_earth',
            'st_teff': 'star_temperature_k',
            'st_rad': 'star_radius_solar',
            'st_mass': 'star_mass_solar',
            'sy_dist': 'distance_pc'
        }
        
        for new_col, old_col in column_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        processed['discovery_method'] = 'Transit'
        processed['discovery_facility'] = 'TESS'
        processed['dataset_source'] = 'toi'
        
        return processed
    
    def _process_k2pandc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process K2 and confirmed planets dataset"""
        processed = pd.DataFrame()
        
        column_mapping = {
            'pl_name': 'planet_name',
            'hostname': 'host_star',
            'discoverymethod': 'discovery_method',
            'disc_year': 'discovery_year',
            'disc_facility': 'discovery_facility',
            'pl_orbper': 'orbital_period_days',
            'pl_rade': 'planet_radius_earth',
            'pl_bmasse': 'planet_mass_earth',
            'pl_dens': 'planet_density',
            'st_teff': 'star_temperature_k',
            'st_rad': 'star_radius_solar',
            'st_mass': 'star_mass_solar',
            'sy_dist': 'distance_pc',
            'pl_eqt': 'equilibrium_temperature_k'
        }
        
        for new_col, old_col in column_mapping.items():
            if old_col in df.columns:
                processed[new_col] = df[old_col]
        
        processed['dataset_source'] = 'k2pandc'
        return processed
    
    def generate_training_texts(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate training texts in conversation format"""
        training_texts = []
        
        for _, row in df.iterrows():
            # Skip rows with missing essential data
            if pd.isna(row.get('planet_name')):
                continue
            
            # Create comprehensive planet description
            planet_info = self._create_planet_description(row)
            
            # Generate multiple training examples for each planet
            examples = self._generate_training_examples(row, planet_info)
            training_texts.extend(examples)
        
        logger.info(f"Generated {len(training_texts)} training examples")
        return training_texts
    
    def _create_planet_description(self, row: pd.Series) -> str:
        """Create a comprehensive description of the planet"""
        description_parts = []
        
        # Basic information
        planet_name = row.get('planet_name', 'Unknown Planet')
        host_star = row.get('host_star', 'Unknown Star')
        
        description_parts.append(f"Planet Name: {planet_name}")
        if host_star and host_star != 'Unknown Star':
            description_parts.append(f"Host Star: {host_star}")
        
        # Physical properties
        if not pd.isna(row.get('planet_radius_earth')):
            description_parts.append(f"Planet Radius: {row['planet_radius_earth']:.2f} Earth radii")
        
        if not pd.isna(row.get('planet_mass_earth')):
            description_parts.append(f"Planet Mass: {row['planet_mass_earth']:.2f} Earth masses")
        
        if not pd.isna(row.get('planet_density')):
            description_parts.append(f"Planet Density: {row['planet_density']:.2f} g/cm³")
        
        # Orbital properties
        if not pd.isna(row.get('orbital_period_days')):
            period = row['orbital_period_days']
            if period < 1:
                description_parts.append(f"Orbital Period: {period*24:.1f} hours")
            elif period < 365:
                description_parts.append(f"Orbital Period: {period:.1f} days")
            else:
                description_parts.append(f"Orbital Period: {period/365:.2f} years")
        
        # Stellar properties
        if not pd.isna(row.get('star_temperature_k')):
            description_parts.append(f"Host Star Temperature: {row['star_temperature_k']:.0f} K")
        
        if not pd.isna(row.get('star_radius_solar')):
            description_parts.append(f"Host Star Radius: {row['star_radius_solar']:.2f} Solar radii")
        
        if not pd.isna(row.get('star_mass_solar')):
            description_parts.append(f"Host Star Mass: {row['star_mass_solar']:.2f} Solar masses")
        
        # Discovery information
        discovery_method = row.get('discovery_method', 'Unknown')
        if discovery_method and discovery_method != 'Unknown':
            description_parts.append(f"Discovery Method: {discovery_method}")
        
        discovery_year = row.get('discovery_year')
        if not pd.isna(discovery_year):
            description_parts.append(f"Discovery Year: {int(discovery_year)}")
        
        discovery_facility = row.get('discovery_facility', 'Unknown')
        if discovery_facility and discovery_facility != 'Unknown':
            description_parts.append(f"Discovery Facility: {discovery_facility}")
        
        # Distance
        if not pd.isna(row.get('distance_pc')):
            description_parts.append(f"Distance from Earth: {row['distance_pc']:.1f} parsecs")
        
        # Temperature
        if not pd.isna(row.get('equilibrium_temperature_k')):
            temp_k = row['equilibrium_temperature_k']
            temp_c = temp_k - 273.15
            description_parts.append(f"Equilibrium Temperature: {temp_k:.0f} K ({temp_c:.0f}°C)")
        
        return "\n".join(description_parts)
    
    def _generate_training_examples(self, row: pd.Series, planet_info: str) -> List[Dict[str, str]]:
        """Generate multiple training examples for each planet"""
        examples = []
        planet_name = row.get('planet_name', 'Unknown Planet')
        
        # Example 1: Basic planet information
        examples.append({
            "instruction": f"Tell me about the planet {planet_name}.",
            "input": "",
            "output": planet_info
        })
        
        # Example 2: Discovery information
        if not pd.isna(row.get('discovery_year')) and not pd.isna(row.get('discovery_method')):
            examples.append({
                "instruction": f"When and how was {planet_name} discovered?",
                "input": "",
                "output": f"{planet_name} was discovered in {int(row['discovery_year'])} using the {row['discovery_method']} method"
            })
        
        # Example 3: Physical properties
        if not pd.isna(row.get('planet_radius_earth')) or not pd.isna(row.get('planet_mass_earth')):
            size_info = []
            if not pd.isna(row.get('planet_radius_earth')):
                size_info.append(f"radius of {row['planet_radius_earth']:.2f} Earth radii")
            if not pd.isna(row.get('planet_mass_earth')):
                size_info.append(f"mass of {row['planet_mass_earth']:.2f} Earth masses")
            
            examples.append({
                "instruction": f"What are the physical properties of {planet_name}?",
                "input": "",
                "output": f"{planet_name} has a {', '.join(size_info)}."
            })
        
        # Example 4: Habitability assessment
        if not pd.isna(row.get('equilibrium_temperature_k')):
            temp = row['equilibrium_temperature_k']
            if temp < 273:
                habitability = "likely too cold for liquid water"
            elif temp > 373:
                habitability = "likely too hot for liquid water"
            else:
                habitability = "potentially in the habitable zone for liquid water"
            
            examples.append({
                "instruction": f"Is {planet_name} potentially habitable?",
                "input": "",
                "output": f"Based on its equilibrium temperature of {temp:.0f} K, {planet_name} is {habitability}."
            })
        
        return examples
    
    def save_training_data(self, training_texts: List[Dict[str, str]], output_dir: str = "training_data"):
        """Save training data in various formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON (for Ollama fine-tuning)
        json_path = output_path / "nasa_planet_training_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_texts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(training_texts)} training examples to {json_path}")
        
        # Save as JSONL (alternative format)
        jsonl_path = output_path / "nasa_planet_training_data.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in training_texts:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Saved training data in JSONL format to {jsonl_path}")
        
        # Save statistics
        stats = {
            "total_examples": len(training_texts),
            "unique_planets": len(set(ex['output'] for ex in training_texts)),
            "avg_output_length": np.mean([len(ex['output']) for ex in training_texts]),
            "instruction_types": {}
        }
        
        for example in training_texts:
            instruction = example['instruction']
            if "Tell me about" in instruction:
                stats["instruction_types"]["planet_description"] = stats["instruction_types"].get("planet_description", 0) + 1
            elif "discovered" in instruction:
                stats["instruction_types"]["discovery_info"] = stats["instruction_types"].get("discovery_info", 0) + 1
            elif "physical properties" in instruction:
                stats["instruction_types"]["physical_properties"] = stats["instruction_types"].get("physical_properties", 0) + 1
            elif "habitable" in instruction:
                stats["instruction_types"]["habitability"] = stats["instruction_types"].get("habitability", 0) + 1
        
        stats_path = output_path / "training_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved training statistics to {stats_path}")
        
        return output_path

def main():
    """Main function to run the preprocessing pipeline"""
    preprocessor = NASAPlanetDataPreprocessor()
    
    # Load and process data
    logger.info("Loading NASA planet datasets...")
    datasets = preprocessor.load_csv_files()
    
    if not datasets:
        logger.error("No datasets found. Please check the data directory.")
        return
    
    logger.info("Cleaning and preparing data...")
    combined_df = preprocessor.clean_and_prepare_data(datasets)
    
    if combined_df.empty:
        logger.error("No data to process after cleaning.")
        return
    
    logger.info("Generating training texts...")
    training_texts = preprocessor.generate_training_texts(combined_df)
    
    logger.info("Saving training data...")
    output_dir = preprocessor.save_training_data(training_texts)
    
    logger.info(f"Preprocessing complete! Training data saved to {output_dir}")
    logger.info(f"Total training examples: {len(training_texts)}")

if __name__ == "__main__":
    main()
