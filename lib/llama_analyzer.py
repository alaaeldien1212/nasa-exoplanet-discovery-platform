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
        
        period = features.get('koi_period', 0)
        radius = features.get('koi_prad', 0)
        stellar_temp = features.get('koi_steff', 0)
        duration = features.get('koi_duration', 0)
        depth = features.get('koi_depth', 0)
        equilibrium_temp = features.get('koi_teq', 0)
        
        prompt = f"""As a NASA exoplanet scientist, analyze this candidate and explain why it was classified as {'an exoplanet' if prediction == 1 else 'a false positive'} with {confidence:.1%} confidence.

Planetary Parameters:
- Orbital Period: {period:.2f} days
- Planetary Radius: {radius:.2f} Earth radii
- Stellar Temperature: {stellar_temp:.0f} K
- Transit Duration: {duration:.2f} hours
- Transit Depth: {depth:.0f} ppm
- Equilibrium Temperature: {equilibrium_temp:.0f} K

Provide a concise scientific explanation (2-3 sentences) focusing on the key factors that led to this classification. Consider:
1. Transit characteristics (depth, duration)
2. Orbital parameters (period, radius)
3. Stellar properties
4. Physical plausibility

Response format: Brief scientific analysis explaining the classification decision."""
        
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
        
        period = features.get('koi_period', 0)
        radius = features.get('koi_prad', 0)
        depth = features.get('koi_depth', 0)
        duration = features.get('koi_duration', 0)
        
        if prediction == 1:  # Exoplanet
            if radius > 0 and period > 0:
                if radius < 2.0 and period < 50:
                    return f"Strong exoplanet candidate with Earth-like radius ({radius:.1f} R⊕) and short orbital period ({period:.1f} days). Transit depth of {depth:.0f} ppm indicates significant stellar dimming consistent with planetary transit."
                elif radius > 2.0:
                    return f"Large exoplanet candidate with radius {radius:.1f} R⊕ and orbital period {period:.1f} days. Deep transit ({depth:.0f} ppm) suggests substantial planetary body causing significant stellar dimming."
                else:
                    return f"Exoplanet candidate with orbital period {period:.1f} days and radius {radius:.1f} R⊕. Transit characteristics (depth: {depth:.0f} ppm, duration: {duration:.1f}h) are consistent with planetary transit."
            else:
                return f"Exoplanet classification based on transit depth ({depth:.0f} ppm) and duration ({duration:.1f}h). High confidence ({confidence:.1%}) suggests planetary characteristics."
        else:  # False positive
            if depth < 100:
                return f"False positive likely due to shallow transit depth ({depth:.0f} ppm) inconsistent with planetary transit. Short duration ({duration:.1f}h) may indicate stellar variability or instrumental noise."
            elif duration < 1.0:
                return f"False positive classification due to extremely short transit duration ({duration:.1f}h) inconsistent with planetary orbital mechanics. Transit depth of {depth:.0f} ppm suggests stellar activity rather than planetary transit."
            else:
                return f"False positive classification based on transit characteristics inconsistent with planetary parameters. Duration ({duration:.1f}h) and depth ({depth:.0f} ppm) patterns suggest stellar variability or instrumental effects."

# Global analyzer instance
llama_analyzer = LlamaAnalyzer()
