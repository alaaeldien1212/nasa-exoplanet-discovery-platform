#!/usr/bin/env python3
"""
Advanced AI Planet Image Generator
Uses multiple AI techniques to generate realistic planet images
"""

import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import colorsys
import math
import random
import requests
import base64
import io

class AIPlanetGenerator:
    def __init__(self):
        self.api_key = None  # You can add API keys for external services
        
    def generate_planet_image(self, planet_data):
        """
        Generate a realistic planet image using AI techniques
        
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
        period = planet_data.get('period', 365)
        
        # Determine planet type and characteristics
        planet_type = self._classify_planet_type(radius, temperature, classification)
        
        # Generate base planet image
        planet_image = self._generate_base_planet(planet_type, radius, temperature)
        
        # Add surface features
        planet_image = self._add_surface_features(planet_image, planet_type, temperature, radius)
        
        # Add atmospheric effects
        if classification == 'exoplanet' and radius > 1.5:
            planet_image = self._add_atmosphere(planet_image, planet_type, temperature)
        
        # Add lighting and shadows
        planet_image = self._add_lighting_effects(planet_image, planet_type)
        
        # Add space background
        final_image = self._add_space_background(planet_image)
        
        # Convert to base64
        return self._image_to_base64(final_image)
    
    def _classify_planet_type(self, radius, temperature, classification):
        """Classify planet type based on characteristics"""
        if classification == 'false positive':
            return 'asteroid'
        elif radius < 1.2:
            if temperature < 200:
                return 'ice_world'
            elif temperature < 350:
                return 'rocky_habitable'
            else:
                return 'rocky_hot'
        elif radius < 2.0:
            if temperature < 300:
                return 'super_earth'
            else:
                return 'super_earth_hot'
        elif radius < 4.0:
            if temperature < 200:
                return 'gas_giant_cold'
            else:
                return 'gas_giant'
        else:
            return 'super_giant'
    
    def _generate_base_planet(self, planet_type, radius, temperature):
        """Generate the base planet sphere"""
        size = 1024
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        planet_radius = int(size * 0.35 * min(radius / 2, 2))
        
        # Get base color based on planet type and temperature
        base_color = self._get_planet_color(planet_type, temperature)
        
        # Create planet sphere with gradient
        self._draw_sphere(draw, center, center, planet_radius, base_color)
        
        return img
    
    def _get_planet_color(self, planet_type, temperature):
        """Get base color for planet type"""
        color_map = {
            'asteroid': (120, 100, 80),
            'ice_world': (180, 200, 255),
            'rocky_habitable': (50, 120, 80),
            'rocky_hot': (150, 80, 50),
            'super_earth': (70, 140, 100),
            'super_earth_hot': (180, 100, 60),
            'gas_giant_cold': (80, 120, 180),
            'gas_giant': (180, 120, 80),
            'super_giant': (200, 100, 60)
        }
        
        base_color = color_map.get(planet_type, (100, 100, 100))
        
        # Adjust color based on temperature
        if temperature > 500:
            # Make it more red/orange for hot planets
            base_color = tuple(min(255, c + int((temperature - 500) / 10)) for c in base_color)
        elif temperature < 200:
            # Make it more blue for cold planets
            base_color = tuple(max(0, c - int((200 - temperature) / 5)) for c in base_color)
        
        return base_color
    
    def _draw_sphere(self, draw, x, y, radius, color):
        """Draw a 3D-looking sphere"""
        # Create gradient effect
        for i in range(radius):
            alpha = int(255 * (1 - i / radius))
            current_radius = radius - i
            current_color = tuple(max(0, c - i * 2) for c in color) + (alpha,)
            
            # Draw concentric circles for 3D effect
            draw.ellipse([x - current_radius, y - current_radius,
                         x + current_radius, y + current_radius],
                        fill=current_color)
    
    def _add_surface_features(self, img, planet_type, temperature, radius):
        """Add realistic surface features"""
        draw = ImageDraw.Draw(img)
        size = img.size[0]
        center = size // 2
        planet_radius = int(size * 0.35 * min(radius / 2, 2))
        
        if planet_type == 'rocky_habitable':
            self._add_continents(draw, center, planet_radius, temperature)
            self._add_polar_ice(draw, center, planet_radius)
        elif planet_type == 'rocky_hot':
            self._add_volcanic_features(draw, center, planet_radius)
        elif planet_type == 'ice_world':
            self._add_ice_features(draw, center, planet_radius)
        elif planet_type in ['gas_giant', 'gas_giant_cold', 'super_giant']:
            self._add_atmospheric_bands(draw, center, planet_radius, temperature)
            self._add_storm_systems(draw, center, planet_radius, planet_type)
        elif planet_type == 'asteroid':
            self._add_craters(draw, center, planet_radius)
        
        return img
    
    def _add_continents(self, draw, center, radius, temperature):
        """Add continental features"""
        # Generate random continents
        for _ in range(random.randint(3, 6)):
            # Random position on sphere
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.7)
            x = center + math.cos(angle) * distance
            y = center + math.sin(angle) * distance
            
            # Check if within planet bounds
            if ((x - center)**2 + (y - center)**2)**0.5 + 30 < radius:
                continent_size = random.randint(20, 60)
                continent_color = (30, 100, 50) if temperature < 300 else (80, 60, 40)
                
                # Add some variation
                for i in range(3):
                    offset_x = x + random.randint(-10, 10)
                    offset_y = y + random.randint(-10, 10)
                    size_variation = continent_size + random.randint(-10, 10)
                    
                    draw.ellipse([offset_x - size_variation, offset_y - size_variation,
                                 offset_x + size_variation, offset_y + size_variation],
                                fill=continent_color)
    
    def _add_polar_ice(self, draw, center, radius):
        """Add polar ice caps"""
        ice_color = (200, 220, 255)
        
        # North pole
        draw.ellipse([center - radius//3, center - radius, center + radius//3, center - radius//2],
                    fill=ice_color)
        # South pole
        draw.ellipse([center - radius//3, center + radius//2, center + radius//3, center + radius],
                    fill=ice_color)
    
    def _add_volcanic_features(self, draw, center, radius):
        """Add volcanic and hot surface features"""
        for _ in range(random.randint(5, 10)):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.8)
            x = center + math.cos(angle) * distance
            y = center + math.sin(angle) * distance
            
            if ((x - center)**2 + (y - center)**2)**0.5 + 15 < radius:
                # Volcanic vent
                draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(200, 100, 50))
                # Lava flow
                for i in range(3):
                    flow_x = x + random.randint(-20, 20)
                    flow_y = y + random.randint(-20, 20)
                    draw.ellipse([flow_x - 5, flow_y - 5, flow_x + 5, flow_y + 5], fill=(255, 150, 100))
    
    def _add_ice_features(self, draw, center, radius):
        """Add ice world features"""
        for _ in range(random.randint(8, 15)):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.8)
            x = center + math.cos(angle) * distance
            y = center + math.sin(angle) * distance
            
            if ((x - center)**2 + (y - center)**2)**0.5 + 10 < radius:
                # Ice formations
                ice_size = random.randint(5, 20)
                ice_color = (150, 180, 220)
                draw.ellipse([x - ice_size, y - ice_size, x + ice_size, y + ice_size],
                           fill=ice_color)
    
    def _add_atmospheric_bands(self, draw, center, radius, temperature):
        """Add atmospheric bands for gas giants"""
        num_bands = random.randint(6, 12)
        band_height = (radius * 2) // num_bands
        
        for i in range(num_bands):
            y_start = center - radius + i * band_height
            y_end = y_start + band_height
            
            # Vary band color based on temperature
            if temperature > 500:
                band_color = (200, 120, 80)
            else:
                band_color = (80, 120, 180)
            
            # Add some variation
            band_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in band_color)
            
            # Draw band
            for y in range(y_start, y_end):
                if y >= center - radius and y <= center + radius:
                    width = int(((radius**2 - (y - center)**2)**0.5) * 2)
                    if width > 0:
                        x_start = center - width // 2
                        x_end = center + width // 2
                        draw.line([x_start, y, x_end, y], fill=band_color, width=2)
    
    def _add_storm_systems(self, draw, center, radius, planet_type):
        """Add storm systems for gas giants"""
        num_storms = 3 if planet_type == 'super_giant' else random.randint(1, 3)
        
        for _ in range(num_storms):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.6)
            x = center + math.cos(angle) * distance
            y = center + math.sin(angle) * distance
            
            if ((x - center)**2 + (y - center)**2)**0.5 + 30 < radius:
                storm_size = random.randint(20, 50)
                storm_color = (50, 50, 50)  # Dark storm
                
                # Main storm
                draw.ellipse([x - storm_size, y - storm_size, x + storm_size, y + storm_size],
                           fill=storm_color)
                
                # Storm spiral
                for i in range(5):
                    spiral_angle = i * 0.5
                    spiral_x = x + math.cos(spiral_angle) * (storm_size - i * 5)
                    spiral_y = y + math.sin(spiral_angle) * (storm_size - i * 5)
                    draw.ellipse([spiral_x - 3, spiral_y - 3, spiral_x + 3, spiral_y + 3],
                               fill=(30, 30, 30))
    
    def _add_craters(self, draw, center, radius):
        """Add craters for asteroids"""
        for _ in range(random.randint(10, 20)):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.8)
            x = center + math.cos(angle) * distance
            y = center + math.sin(angle) * distance
            
            if ((x - center)**2 + (y - center)**2)**0.5 + 8 < radius:
                crater_size = random.randint(5, 15)
                
                # Crater rim
                draw.ellipse([x - crater_size, y - crater_size, x + crater_size, y + crater_size],
                           fill=(80, 70, 60))
                # Crater center
                draw.ellipse([x - crater_size//2, y - crater_size//2, x + crater_size//2, y + crater_size//2],
                           fill=(40, 35, 30))
    
    def _add_atmosphere(self, img, planet_type, temperature):
        """Add atmospheric glow effect"""
        size = img.size[0]
        center = size // 2
        radius = int(size * 0.35)
        
        # Create atmospheric glow
        glow_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_img)
        
        # Atmospheric color based on planet type
        if planet_type in ['gas_giant', 'super_giant']:
            atm_color = (100, 150, 200, 30)
        else:
            atm_color = (150, 200, 255, 20)
        
        # Draw atmospheric layers
        for i in range(5):
            current_radius = radius + i * 8
            alpha = 30 - i * 5
            current_color = atm_color[:3] + (alpha,)
            
            glow_draw.ellipse([center - current_radius, center - current_radius,
                              center + current_radius, center + current_radius],
                             fill=current_color)
        
        # Blend with original image
        return Image.alpha_composite(img, glow_img)
    
    def _add_lighting_effects(self, img, planet_type):
        """Add realistic lighting effects"""
        size = img.size[0]
        center = size // 2
        radius = int(size * 0.35)
        
        # Create lighting overlay
        light_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        light_draw = ImageDraw.Draw(light_img)
        
        # Add highlight (sun reflection)
        highlight_x = center + radius // 3
        highlight_y = center - radius // 3
        highlight_size = radius // 4
        
        light_draw.ellipse([highlight_x - highlight_size, highlight_y - highlight_size,
                           highlight_x + highlight_size, highlight_y + highlight_size],
                          fill=(255, 255, 255, 100))
        
        # Add terminator (day/night boundary)
        terminator_x = center - radius // 3
        night_color = (0, 0, 0, 150)
        light_draw.ellipse([center - radius, center - radius, terminator_x, center + radius],
                          fill=night_color)
        
        # Blend with original image
        return Image.alpha_composite(img, light_img)
    
    def _add_space_background(self, planet_img):
        """Add space background with stars"""
        size = planet_img.size[0]
        background = Image.new('RGB', (size, size), (5, 5, 20))
        bg_draw = ImageDraw.Draw(background)
        
        # Add stars
        for _ in range(200):
            x = random.randint(0, size)
            y = random.randint(0, size)
            star_size = random.randint(1, 3)
            brightness = random.randint(150, 255)
            
            bg_draw.ellipse([x - star_size, y - star_size, x + star_size, y + star_size],
                          fill=(brightness, brightness, brightness))
        
        # Add some colorful stars
        for _ in range(20):
            x = random.randint(0, size)
            y = random.randint(0, size)
            star_size = random.randint(2, 4)
            color = random.choice([(255, 200, 200), (200, 200, 255), (200, 255, 200)])
            
            bg_draw.ellipse([x - star_size, y - star_size, x + star_size, y + star_size],
                          fill=color)
        
        # Composite planet on background
        background.paste(planet_img, (0, 0), planet_img)
        return background
    
    def _image_to_base64(self, img):
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64

# Global generator instance
ai_generator = AIPlanetGenerator()

if __name__ == "__main__":
    # Read planet data from stdin
    try:
        planet_data = json.loads(sys.stdin.read())
        image_base64 = ai_generator.generate_planet_image(planet_data)
        
        # Output result as JSON
        result = {
            "success": True,
            "image": image_base64,
            "planet_type": "ai_generated"
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)
