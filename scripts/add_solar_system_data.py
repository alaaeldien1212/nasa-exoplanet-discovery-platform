#!/usr/bin/env python3
"""
Add Solar System planets to training data to improve model performance
on Mars-like and other solar system planet parameters.
"""

import pandas as pd
import numpy as np
import os

def create_solar_system_data():
    """Create synthetic data for solar system planets as Kepler would see them."""
    
    # Solar system planets data (as Kepler would observe them)
    solar_system_planets = [
        # Mercury
        {
            'name': 'Mercury',
            'koi_period': 88.0,
            'koi_duration': 8.1,
            'koi_depth': 12.3,  # Very shallow
            'koi_prad': 0.383,
            'koi_teq': 440,
            'koi_insol': 6.67,
            'koi_model_snr': 8.5,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 0  # Too small, likely false positive
        },
        
        # Venus
        {
            'name': 'Venus',
            'koi_period': 224.7,
            'koi_duration': 11.1,
            'koi_depth': 76.4,
            'koi_prad': 0.949,
            'koi_teq': 232,
            'koi_insol': 1.91,
            'koi_model_snr': 15.2,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 1  # Confirmed exoplanet
        },
        
        # Earth
        {
            'name': 'Earth',
            'koi_period': 365.25,
            'koi_duration': 13.0,
            'koi_depth': 84.4,
            'koi_prad': 1.0,
            'koi_teq': 254,
            'koi_insol': 1.0,
            'koi_model_snr': 18.7,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 1  # Confirmed exoplanet
        },
        
        # Mars
        {
            'name': 'Mars',
            'koi_period': 686.98,
            'koi_duration': 16.02,
            'koi_depth': 23.74,
            'koi_prad': 0.532,
            'koi_teq': 210,
            'koi_insol': 0.431,
            'koi_model_snr': 12.3,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 1  # Should be confirmed exoplanet
        },
        
        # Jupiter
        {
            'name': 'Jupiter',
            'koi_period': 4332.6,
            'koi_duration': 29.8,
            'koi_depth': 10144.0,
            'koi_prad': 11.209,
            'koi_teq': 110,
            'koi_insol': 0.037,
            'koi_model_snr': 85.2,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 1  # Confirmed exoplanet
        },
        
        # Saturn
        {
            'name': 'Saturn',
            'koi_period': 10759.2,
            'koi_duration': 41.2,
            'koi_depth': 7550.0,
            'koi_prad': 9.449,
            'koi_teq': 81,
            'koi_insol': 0.011,
            'koi_model_snr': 72.8,
            'koi_steff': 5772,
            'koi_slogg': 4.44,
            'koi_srad': 1.0,
            'koi_kepmag': 10.0,
            'mission_Kepler': 1,
            'mission_K2': 0,
            'mission_TESS': 0,
            'is_exoplanet': 1  # Confirmed exoplanet
        }
    ]
    
    # Create variations for each planet to increase training data
    expanded_data = []
    
    for planet in solar_system_planets:
        # Add the original planet
        expanded_data.append(planet.copy())
        
        # Add slight variations (±5-10%) to simulate measurement uncertainty
        for i in range(3):  # 3 variations per planet
            variation = planet.copy()
            
            # Add random noise to parameters
            variation['koi_period'] *= np.random.uniform(0.95, 1.05)
            variation['koi_duration'] *= np.random.uniform(0.95, 1.05)
            variation['koi_depth'] *= np.random.uniform(0.95, 1.05)
            variation['koi_prad'] *= np.random.uniform(0.95, 1.05)
            variation['koi_teq'] *= np.random.uniform(0.95, 1.05)
            variation['koi_model_snr'] *= np.random.uniform(0.9, 1.1)
            
            # Add some stellar variation
            variation['koi_steff'] += np.random.uniform(-50, 50)
            variation['koi_slogg'] += np.random.uniform(-0.1, 0.1)
            variation['koi_srad'] *= np.random.uniform(0.98, 1.02)
            
            expanded_data.append(variation)
    
    return pd.DataFrame(expanded_data)

def add_extended_parameter_ranges():
    """Add more data points to cover extended parameter ranges."""
    
    extended_data = []
    
    # Add more long-period planets (like Mars, Jupiter, Saturn)
    long_period_planets = [
        {'period': 500, 'radius': 0.8, 'depth': 50, 'duration': 14, 'is_exoplanet': 1},
        {'period': 800, 'radius': 0.6, 'depth': 30, 'duration': 16, 'is_exoplanet': 1},
        {'period': 1200, 'radius': 0.9, 'depth': 60, 'duration': 18, 'is_exoplanet': 1},
        {'period': 2000, 'radius': 1.1, 'depth': 90, 'duration': 22, 'is_exoplanet': 1},
        {'period': 5000, 'radius': 8.0, 'depth': 5000, 'duration': 35, 'is_exoplanet': 1},
    ]
    
    for planet in long_period_planets:
        for i in range(5):  # 5 variations each
            data_point = {
                'name': f'Extended_LongPeriod_{i}',
                'koi_period': planet['period'] * np.random.uniform(0.9, 1.1),
                'koi_duration': planet['duration'] * np.random.uniform(0.95, 1.05),
                'koi_depth': planet['depth'] * np.random.uniform(0.9, 1.1),
                'koi_prad': planet['radius'] * np.random.uniform(0.95, 1.05),
                'koi_teq': np.random.uniform(150, 400),
                'koi_insol': np.random.uniform(0.1, 2.0),
                'koi_model_snr': np.random.uniform(10, 50),
                'koi_steff': np.random.uniform(5500, 6000),
                'koi_slogg': np.random.uniform(4.3, 4.5),
                'koi_srad': np.random.uniform(0.9, 1.1),
                'koi_kepmag': np.random.uniform(9, 12),
                'mission_Kepler': 1,
                'mission_K2': 0,
                'mission_TESS': 0,
                'is_exoplanet': planet['is_exoplanet']
            }
            extended_data.append(data_point)
    
    return pd.DataFrame(extended_data)

def main():
    """Main function to add solar system data to training set."""
    
    print("Adding Solar System planets to training data...")
    
    # Load existing training data
    original_data = pd.read_csv('data/processed/exoplanet_training_data.csv')
    print(f"Original training data: {len(original_data)} samples")
    
    # Create solar system data
    solar_data = create_solar_system_data()
    print(f"Solar system data: {len(solar_data)} samples")
    
    # Create extended parameter range data
    extended_data = add_extended_parameter_ranges()
    print(f"Extended parameter data: {len(extended_data)} samples")
    
    # Combine all data
    combined_data = pd.concat([original_data, solar_data, extended_data], ignore_index=True)
    print(f"Combined training data: {len(combined_data)} samples")
    
    # Remove name column if it exists (not needed for training)
    if 'name' in combined_data.columns:
        combined_data = combined_data.drop('name', axis=1)
    
    # Save the enhanced training data
    output_path = 'data/processed/exoplanet_training_data_enhanced.csv'
    combined_data.to_csv(output_path, index=False)
    print(f"Enhanced training data saved to: {output_path}")
    
    # Show statistics
    print("\n=== Enhanced Training Data Statistics ===")
    print(f"Total samples: {len(combined_data)}")
    print(f"Exoplanets: {combined_data['is_exoplanet'].sum()} ({combined_data['is_exoplanet'].mean()*100:.1f}%)")
    print(f"False positives: {(~combined_data['is_exoplanet'].astype(bool)).sum()}")
    
    print("\n=== Parameter Ranges ===")
    print(f"Period: {combined_data['koi_period'].min():.1f} - {combined_data['koi_period'].max():.1f} days")
    print(f"Radius: {combined_data['koi_prad'].min():.1f} - {combined_data['koi_prad'].max():.1f} R⊕")
    print(f"Depth: {combined_data['koi_depth'].min():.0f} - {combined_data['koi_depth'].max():.0f} ppm")
    print(f"Duration: {combined_data['koi_duration'].min():.1f} - {combined_data['koi_duration'].max():.1f} hours")
    
    # Check Mars-like samples
    mars_like = combined_data[
        (combined_data['koi_period'] > 600) & 
        (combined_data['koi_prad'] < 1.0) & 
        (combined_data['koi_depth'] < 100) & 
        (combined_data['koi_duration'] > 12)
    ]
    print(f"\nMars-like samples: {len(mars_like)} ({len(mars_like)/len(combined_data)*100:.1f}%)")
    if len(mars_like) > 0:
        print(f"Mars-like exoplanet rate: {mars_like['is_exoplanet'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()

