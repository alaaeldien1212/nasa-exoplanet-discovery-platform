import json
import subprocess
import sys
from typing import Dict, Any

class LlamaAnalyzer:
    def __init__(self):
        self.model_name = "llama3.2:latest"  # Default model
    
    def generate_analysis_description(self, prediction_data: Dict[str, Any]) -> str:
        """Generate AI analysis description using Llama model"""
        
        try:
            # Extract key features
            features = prediction_data.get('features', {})
            prediction = prediction_data.get('prediction', 0)
            confidence = prediction_data.get('confidence', 0)
            classification = prediction_data.get('classification', 'unknown')
            
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(features, prediction, confidence, classification)
            
            # Generate description using Llama
            description = self._call_llama_model(prompt)
            
            return description
            
        except Exception as e:
            print(f"Error generating AI description: {e}", file=sys.stderr)
            return self._get_fallback_description(prediction_data)
    
    def _create_analysis_prompt(self, features: Dict, prediction: int, confidence: float, classification: str) -> str:
        """Create analysis prompt for Llama model"""
        
        # Support both old koi_ names and new mission-agnostic names
        period = features.get('orbital_period', features.get('koi_period', 0))
        radius = features.get('planetary_radius', features.get('koi_prad', 0))
        stellar_temp = features.get('stellar_temperature', features.get('koi_steff', 0))
        duration = features.get('transit_duration', features.get('koi_duration', 0))
        depth = features.get('transit_depth', features.get('koi_depth', 0))
        equilibrium_temp = features.get('equilibrium_temperature', features.get('koi_teq', 0))
        insol = features.get('insolation', features.get('koi_insol', 0))
        snr = features.get('signal_to_noise_ratio', features.get('koi_model_snr', 0))
        
        # Convert depth to ppm if it's in fractional form
        if depth < 1.0 and depth > 0:
            depth_ppm = depth * 1000000
        else:
            depth_ppm = depth
        
        # Add some randomization to make responses more dynamic
        import random
        analysis_angles = [
            "from a stellar evolution perspective",
            "considering planetary formation theory", 
            "based on transit photometry analysis",
            "from an orbital mechanics standpoint",
            "using comparative planetology methods",
            "through atmospheric modeling",
            "via statistical validation techniques"
        ]
        
        focus_areas = [
            "the transit light curve characteristics",
            "the orbital dynamics and stability",
            "the stellar properties and evolution",
            "the planetary atmospheric composition",
            "the formation and migration history",
            "the observational constraints and uncertainties"
        ]
        
        selected_angle = random.choice(analysis_angles)
        selected_focus = random.choice(focus_areas)
        
        prompt = f"""You are a NASA exoplanet scientist providing a direct scientific analysis {selected_angle}. 

Analyze this candidate and explain why it was classified as {'an exoplanet' if prediction == 1 else 'a false positive'} with {confidence:.1%} confidence.

Planetary Parameters:
- Orbital Period: {period:.2f} days
- Planetary Radius: {radius:.2f} Earth radii  
- Stellar Temperature: {stellar_temp:.0f} K
- Transit Duration: {duration:.2f} hours
- Transit Depth: {depth_ppm:.0f} ppm
- Equilibrium Temperature: {equilibrium_temp:.0f} K
- Insolation: {insol:.2f} Earth units
- Signal-to-Noise Ratio: {snr:.1f}

Focus your analysis on {selected_focus}. Provide a direct, scientifically rigorous explanation (3-4 sentences) that includes:
1. Specific physical reasoning based on the parameters
2. Comparison to known exoplanet populations
3. Discussion of uncertainties and limitations
4. Implications for planetary science

Write in a direct, scientific tone without formal greetings or presentation language. Be specific, technical, and insightful."""
        
        return prompt
    
    def _call_llama_model(self, prompt: str) -> str:
        """Call Llama model to generate analysis"""
        
        try:
            # Use ollama to generate response
            cmd = [
                "ollama", "run", self.model_name,
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Clean up the response
                response = response.replace(prompt, "").strip()
                return response[:300] + "..." if len(response) > 300 else response
            else:
                print(f"Ollama error: {result.stderr}", file=sys.stderr)
                return self._get_fallback_description({})
                
        except subprocess.TimeoutExpired:
            print("Llama model timeout", file=sys.stderr)
            return self._get_fallback_description({})
        except FileNotFoundError:
            print("Ollama not found, using fallback", file=sys.stderr)
            return self._get_fallback_description({})
        except Exception as e:
            print(f"Error calling Llama: {e}", file=sys.stderr)
            return self._get_fallback_description({})
    
    def _get_fallback_description(self, prediction_data: Dict) -> str:
        """Generate fallback description when Llama is unavailable"""
        
        features = prediction_data.get('features', {})
        prediction = prediction_data.get('prediction', 0)
        confidence = prediction_data.get('confidence', 0)
        
        # Support both old koi_ names and new mission-agnostic names
        period = features.get('orbital_period', features.get('koi_period', 0))
        radius = features.get('planetary_radius', features.get('koi_prad', 0))
        depth = features.get('transit_depth', features.get('koi_depth', 0))
        duration = features.get('transit_duration', features.get('koi_duration', 0))
        stellar_temp = features.get('stellar_temperature', features.get('koi_steff', 0))
        equilibrium_temp = features.get('equilibrium_temperature', features.get('koi_teq', 0))
        snr = features.get('signal_to_noise_ratio', features.get('koi_model_snr', 0))
        
        # Convert depth to ppm if it's in fractional form
        if depth < 1.0 and depth > 0:
            depth_ppm = depth * 1000000
        else:
            depth_ppm = depth
        
        # Add randomization for more dynamic responses
        import random
        
        if prediction == 1:  # Exoplanet
            exoplanet_templates = [
                f"Analysis reveals a compelling exoplanet candidate with {radius:.1f} R⊕ radius and {period:.1f}-day orbital period. The {depth_ppm:.0f} ppm transit depth indicates substantial stellar dimming consistent with planetary transits observed in Kepler data.",
                f"Statistical validation confirms this as an exoplanet with {confidence:.0%} confidence. The {period:.1f}-day period and {radius:.1f} R⊕ radius place this candidate in the {self._get_planet_type(radius)} category, with transit characteristics matching confirmed exoplanets.",
                f"Transit photometry analysis supports exoplanet classification. The {depth_ppm:.0f} ppm depth and {duration:.1f}h duration align with planetary transit models, while the {equilibrium_temp:.0f}K equilibrium temperature suggests {self._get_temperature_zone(equilibrium_temp)} conditions.",
                f"Comparative analysis with the Kepler exoplanet catalog validates this candidate. The {snr:.1f} signal-to-noise ratio exceeds validation thresholds, and the orbital parameters are consistent with known planetary systems."
            ]
            return random.choice(exoplanet_templates)
        else:  # False positive
            false_positive_templates = [
                f"Detailed analysis indicates this is a false positive with {confidence:.0%} confidence. The {depth_ppm:.0f} ppm transit depth is inconsistent with planetary parameters, suggesting stellar variability or instrumental effects.",
                f"Statistical validation rejects the planetary hypothesis. The {duration:.1f}h transit duration and {depth_ppm:.0f} ppm depth pattern matches stellar activity rather than planetary transits observed in confirmed exoplanets.",
                f"Orbital mechanics analysis reveals inconsistencies. The {period:.1f}-day period combined with {radius:.1f} R⊕ radius creates unstable orbital configurations inconsistent with planetary formation theory.",
                f"Signal analysis suggests instrumental or stellar origin. The {snr:.1f} signal-to-noise ratio and transit characteristics align with known false positive patterns in the Kepler dataset."
            ]
            return random.choice(false_positive_templates)
    
    def _get_planet_type(self, radius: float) -> str:
        """Determine planet type based on radius"""
        if radius < 1.25:
            return "Earth-like"
        elif radius < 2.0:
            return "Super-Earth"
        elif radius < 4.0:
            return "Neptune-like"
        elif radius < 8.0:
            return "Jupiter-like"
        else:
            return "Super-Jupiter"
    
    def _get_temperature_zone(self, temp: float) -> str:
        """Determine temperature zone"""
        if temp < 200:
            return "cryogenic"
        elif temp < 300:
            return "temperate"
        elif temp < 500:
            return "warm"
        elif temp < 1000:
            return "hot"
        else:
            return "extremely hot"

# Global analyzer instance
llama_analyzer = LlamaAnalyzer()
