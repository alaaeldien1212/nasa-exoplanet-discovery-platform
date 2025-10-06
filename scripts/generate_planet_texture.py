#!/usr/bin/env python3
"""
AI-Powered Planet Texture Generator
Generates realistic planet textures based on exoplanet characteristics
"""

import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import colorsys
import math
import random

def get_dynamic_planet_color(temperature):
    """
    Generate dynamic planet color based on temperature with smooth transitions
    Matches the JavaScript color system for consistency
    """
    def interpolate_color(color1, color2, ratio):
        """Interpolate between two RGB tuples"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        return (r, g, b)
    
    # Temperature-based color ranges (same as JavaScript version)
    if temperature <= 50:
        # Deep space blue to dark blue (0K - 50K)
        ratio = temperature / 50
        return interpolate_color((0, 0, 51), (26, 35, 126), ratio)
    elif temperature <= 150:
        # Dark blue to medium blue (50K - 150K)
        ratio = (temperature - 50) / 100
        return interpolate_color((26, 35, 126), (25, 118, 210), ratio)
    elif temperature <= 250:
        # Medium blue to cyan-blue (150K - 250K)
        ratio = (temperature - 150) / 100
        return interpolate_color((25, 118, 210), (2, 136, 209), ratio)
    elif temperature <= 350:
        # Cyan-blue to teal (250K - 350K)
        ratio = (temperature - 250) / 100
        return interpolate_color((2, 136, 209), (0, 151, 167), ratio)
    elif temperature <= 450:
        # Teal to green (350K - 450K)
        ratio = (temperature - 350) / 100
        return interpolate_color((0, 151, 167), (56, 142, 60), ratio)
    elif temperature <= 550:
        # Green to yellow-green (450K - 550K)
        ratio = (temperature - 450) / 100
        return interpolate_color((56, 142, 60), (104, 159, 56), ratio)
    elif temperature <= 650:
        # Yellow-green to yellow (550K - 650K)
        ratio = (temperature - 550) / 100
        return interpolate_color((104, 159, 56), (251, 192, 45), ratio)
    elif temperature <= 750:
        # Yellow to orange (650K - 750K)
        ratio = (temperature - 650) / 100
        return interpolate_color((251, 192, 45), (255, 143, 0), ratio)
    elif temperature <= 850:
        # Orange to red-orange (750K - 850K)
        ratio = (temperature - 750) / 100
        return interpolate_color((255, 143, 0), (245, 124, 0), ratio)
    elif temperature <= 950:
        # Red-orange to red (850K - 950K)
        ratio = (temperature - 850) / 100
        return interpolate_color((245, 124, 0), (211, 47, 47), ratio)
    elif temperature <= 1100:
        # Red to dark red (950K - 1100K)
        ratio = (temperature - 950) / 150
        return interpolate_color((211, 47, 47), (183, 28, 28), ratio)
    elif temperature <= 1300:
        # Dark red to deep red (1100K - 1300K)
        ratio = (temperature - 1100) / 200
        return interpolate_color((183, 28, 28), (141, 27, 27), ratio)
    elif temperature <= 1500:
        # Deep red to purple-red (1300K - 1500K)
        ratio = (temperature - 1300) / 200
        return interpolate_color((141, 27, 27), (106, 27, 154), ratio)
    elif temperature <= 1700:
        # Purple-red to purple (1500K - 1700K)
        ratio = (temperature - 1500) / 200
        return interpolate_color((106, 27, 154), (123, 31, 162), ratio)
    elif temperature <= 1900:
        # Purple to pink (1700K - 1900K)
        ratio = (temperature - 1700) / 200
        return interpolate_color((123, 31, 162), (194, 24, 91), ratio)
    elif temperature <= 2100:
        # Pink to hot pink (1900K - 2100K)
        ratio = (temperature - 1900) / 200
        return interpolate_color((194, 24, 91), (233, 30, 99), ratio)
    else:
        # Hot pink to white (2100K+)
        ratio = min((temperature - 2100) / 500, 1)
        return interpolate_color((233, 30, 99), (255, 255, 255), ratio)

def generate_planet_texture(planet_data):
    """
    Generate a realistic planet texture based on exoplanet characteristics
    
    Args:
        planet_data (dict): Dictionary containing planet characteristics
        
    Returns:
        str: Base64 encoded PNG image
    """
    
    # Extract planet characteristics
    radius = planet_data.get('radius', 1.0)
    temperature = planet_data.get('temperature', 288)
    classification = planet_data.get('classification', 'exoplanet')
    confidence = planet_data.get('confidence', 0.5)
    
    # Create image
    size = 512
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Add randomization for more dynamic textures
    random.seed(hash(str(planet_data)) % 1000000)  # Deterministic but varied
    
    # Use dynamic color system based on temperature
    base_color = get_dynamic_planet_color(temperature)
    
    # Determine planet type based on radius and temperature
    if classification == 'false positive':
        planet_type = 'rocky'
        base_color = (100, 100, 100)  # Gray for false positives
    elif radius < 1.5:
        if temperature < 200:
            planet_type = 'ice'
        elif temperature < 350:
            planet_type = 'habitable'
        else:
            planet_type = 'rocky'
    elif radius < 4.0:
        planet_type = 'gas_giant_cold' if temperature < 200 else 'gas_giant_hot'
    else:
        planet_type = 'super_giant'
    
    # Create base planet surface
    center = size // 2
    planet_radius = int(size * 0.4)
    
    # Draw base planet
    draw.ellipse([center - planet_radius, center - planet_radius,
                  center + planet_radius, center + planet_radius],
                 fill=base_color)
    
    # Add surface features based on planet type
    if planet_type == 'habitable':
        add_habitable_features(draw, center, planet_radius, base_color)
    elif planet_type == 'ice':
        add_ice_features(draw, center, planet_radius, base_color)
    elif planet_type == 'rocky':
        add_rocky_features(draw, center, planet_radius, base_color)
    elif planet_type in ['gas_giant_cold', 'gas_giant_hot']:
        add_gas_giant_features(draw, center, planet_radius, base_color, temperature)
    elif planet_type == 'super_giant':
        add_super_giant_features(draw, center, planet_radius, base_color)
    
    # Add atmospheric effects
    if classification == 'exoplanet' and radius > 1.5:
        add_atmosphere(draw, center, planet_radius, base_color)
    
    # Add lighting effects
    add_lighting_effects(draw, center, planet_radius, base_color)
    
    # Convert to base64
    import io
    import base64
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def add_habitable_features(draw, center, radius, base_color):
    """Add features for habitable planets"""
    # Add continents
    for _ in range(random.randint(3, 7)):
        x = center + random.randint(-radius//2, radius//2)
        y = center + random.randint(-radius//2, radius//2)
        continent_radius = random.randint(20, 60)
        
        # Check if continent is within planet bounds
        if ((x - center)**2 + (y - center)**2)**0.5 + continent_radius < radius:
            continent_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
            draw.ellipse([x - continent_radius, y - continent_radius,
                         x + continent_radius, y + continent_radius],
                        fill=continent_color)
    
    # Add polar ice caps
    ice_color = (200, 220, 255)
    # North pole
    draw.ellipse([center - radius//3, center - radius, center + radius//3, center - radius//2],
                fill=ice_color)
    # South pole
    draw.ellipse([center - radius//3, center + radius//2, center + radius//3, center + radius],
                fill=ice_color)

def add_ice_features(draw, center, radius, base_color):
    """Add features for ice worlds"""
    # Add ice cracks
    for _ in range(random.randint(5, 10)):
        x1 = center + random.randint(-radius//2, radius//2)
        y1 = center + random.randint(-radius//2, radius//2)
        x2 = x1 + random.randint(-50, 50)
        y2 = y1 + random.randint(-50, 50)
        
        if ((x1 - center)**2 + (y1 - center)**2)**0.5 < radius and \
           ((x2 - center)**2 + (y2 - center)**2)**0.5 < radius:
            draw.line([x1, y1, x2, y2], fill=(150, 150, 200), width=2)
    
    # Add ice formations
    for _ in range(random.randint(3, 6)):
        x = center + random.randint(-radius//2, radius//2)
        y = center + random.randint(-radius//2, radius//2)
        formation_radius = random.randint(10, 30)
        
        if ((x - center)**2 + (y - center)**2)**0.5 + formation_radius < radius:
            draw.ellipse([x - formation_radius, y - formation_radius,
                         x + formation_radius, y + formation_radius],
                        fill=(180, 200, 255))

def add_rocky_features(draw, center, radius, base_color):
    """Add features for rocky planets"""
    # Add craters
    for _ in range(random.randint(8, 15)):
        x = center + random.randint(-radius//2, radius//2)
        y = center + random.randint(-radius//2, radius//2)
        crater_radius = random.randint(5, 25)
        
        if ((x - center)**2 + (y - center)**2)**0.5 + crater_radius < radius:
            # Crater rim
            draw.ellipse([x - crater_radius, y - crater_radius,
                         x + crater_radius, y + crater_radius],
                        fill=tuple(max(0, min(255, c - 40)) for c in base_color))
            # Crater center
            draw.ellipse([x - crater_radius//2, y - crater_radius//2,
                         x + crater_radius//2, y + crater_radius//2],
                        fill=tuple(max(0, min(255, c - 80)) for c in base_color))
    
    # Add mountain ranges
    for _ in range(random.randint(2, 4)):
        start_x = center + random.randint(-radius//2, radius//2)
        start_y = center + random.randint(-radius//2, radius//2)
        
        for i in range(10):
            x = start_x + i * 10
            y = start_y + random.randint(-20, 20)
            
            if ((x - center)**2 + (y - center)**2)**0.5 < radius:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                           fill=tuple(max(0, min(255, c + 30)) for c in base_color))

def add_gas_giant_features(draw, center, radius, base_color, temperature):
    """Add features for gas giants"""
    # Add atmospheric bands
    num_bands = random.randint(4, 8)
    band_height = (radius * 2) // num_bands
    
    for i in range(num_bands):
        y_start = center - radius + i * band_height
        y_end = y_start + band_height
        
        # Vary band color based on temperature
        if temperature > 500:
            band_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
        else:
            band_color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in base_color)
        
        # Create band shape
        for y in range(y_start, y_end):
            if y >= center - radius and y <= center + radius:
                # Calculate width at this y position
                width = int(((radius**2 - (y - center)**2)**0.5) * 2)
                x_start = center - width // 2
                x_end = center + width // 2
                
                draw.line([x_start, y, x_end, y], fill=band_color, width=1)
    
    # Add storm systems
    for _ in range(random.randint(1, 3)):
        x = center + random.randint(-radius//2, radius//2)
        y = center + random.randint(-radius//2, radius//2)
        storm_radius = random.randint(15, 40)
        
        if ((x - center)**2 + (y - center)**2)**0.5 + storm_radius < radius:
            storm_color = tuple(max(0, min(255, c - 50)) for c in base_color)
            draw.ellipse([x - storm_radius, y - storm_radius,
                         x + storm_radius, y + storm_radius],
                        fill=storm_color)

def add_super_giant_features(draw, center, radius, base_color):
    """Add features for super gas giants"""
    # Add more intense atmospheric bands
    num_bands = random.randint(6, 12)
    band_height = (radius * 2) // num_bands
    
    for i in range(num_bands):
        y_start = center - radius + i * band_height
        y_end = y_start + band_height
        
        band_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
        
        for y in range(y_start, y_end):
            if y >= center - radius and y <= center + radius:
                width = int(((radius**2 - (y - center)**2)**0.5) * 2)
                x_start = center - width // 2
                x_end = center + width // 2
                
                draw.line([x_start, y, x_end, y], fill=band_color, width=2)
    
    # Add multiple storm systems
    for _ in range(random.randint(3, 6)):
        x = center + random.randint(-radius//2, radius//2)
        y = center + random.randint(-radius//2, radius//2)
        storm_radius = random.randint(20, 50)
        
        if ((x - center)**2 + (y - center)**2)**0.5 + storm_radius < radius:
            storm_color = tuple(max(0, min(255, c - 80)) for c in base_color)
            draw.ellipse([x - storm_radius, y - storm_radius,
                         x + storm_radius, y + storm_radius],
                        fill=storm_color)

def add_atmosphere(draw, center, radius, base_color):
    """Add atmospheric glow effect"""
    # Create atmospheric layer
    atm_radius = int(radius * 1.1)
    atm_color = tuple(max(0, min(255, c // 4)) for c in base_color) + (50,)
    
    # Draw atmospheric glow
    for i in range(5):
        current_radius = radius + i * 2
        if current_radius <= atm_radius:
            alpha = 50 - i * 10
            atm_color_with_alpha = atm_color[:3] + (alpha,)
            
            # Create temporary image for blending
            temp_img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.ellipse([center - current_radius, center - current_radius,
                              center + current_radius, center + current_radius],
                             fill=atm_color_with_alpha)
            
            # Blend with main image
            draw._image.paste(temp_img, (0, 0), temp_img)

def add_lighting_effects(draw, center, radius, base_color):
    """Add realistic lighting effects"""
    # Add terminator line (day/night boundary)
    terminator_x = center - radius // 3
    
    # Night side
    night_color = tuple(max(0, c // 3) for c in base_color)
    draw.ellipse([center - radius, center - radius, terminator_x, center + radius],
                fill=night_color)
    
    # Add highlight
    highlight_x = center + radius // 4
    highlight_y = center - radius // 4
    highlight_radius = radius // 6
    
    highlight_color = tuple(min(255, c + 50) for c in base_color)
    draw.ellipse([highlight_x - highlight_radius, highlight_y - highlight_radius,
                 highlight_x + highlight_radius, highlight_y + highlight_radius],
                fill=highlight_color)

if __name__ == "__main__":
    # Read planet data from stdin
    try:
        planet_data = json.loads(sys.stdin.read())
        texture_base64 = generate_planet_texture(planet_data)
        
        # Output result as JSON
        result = {
            "success": True,
            "texture": texture_base64,
            "planet_type": "generated"
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)
