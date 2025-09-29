# NASA Planet Expert LLM Training

This project trains a specialized Large Language Model (LLM) using Ollama with NASA's comprehensive exoplanet dataset. The trained model will be an expert on planetary discovery, characterization, and habitability assessment.

## 🚀 Features

- **Comprehensive NASA Data**: Uses data from NASA's Exoplanet Archive including:
  - Cumulative exoplanet discoveries
  - TESS Object of Interest (TOI) data
  - K2 and confirmed planets data
- **Specialized Training**: Fine-tuned for exoplanet analysis and discovery
- **Multiple Training Approaches**: Both basic and advanced training scripts
- **Evaluation Tools**: Comprehensive model testing and evaluation
- **Production Ready**: Optimized for real-world exoplanet research applications

## 📊 Dataset Overview

The training data includes information about:
- **5,000+ confirmed exoplanets**
- **Physical properties** (radius, mass, density)
- **Orbital characteristics** (period, semi-major axis)
- **Stellar properties** (temperature, radius, mass)
- **Discovery methods** (transit, radial velocity, direct imaging)
- **Habitability indicators** (equilibrium temperature, habitable zone)

## 🛠️ Installation

### Prerequisites

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai/
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama installation**:
   ```bash
   ollama --version
   ```

### Setup

1. **Clone or download this repository**
2. **Ensure your NASA data is in the correct location**:
   ```
   nasa data planet/
   ├── cumulative_2025.09.27_12.55.48.csv
   ├── TOI_2025.09.27_12.56.11.csv
   └── k2pandc_2025.09.27_12.56.23.csv
   ```

## 🚀 Quick Start

### Step 1: Preprocess Data
```bash
python3 data_preprocessor.py
```

### Step 2: Train the Model
```bash
# Option A: Simple training
./train_ollama_model.sh

# Option B: Advanced training with custom parameters
python3 advanced_training.py
```

### Step 3: Evaluate the Model
```bash
python3 evaluate_model.py
```

## 📋 Training Process

### Data Preprocessing
The `data_preprocessor.py` script:
1. **Loads NASA CSV files** from the Exoplanet Archive
2. **Cleans and standardizes** data across different datasets
3. **Generates training examples** in conversation format
4. **Creates multiple question types**:
   - Planet descriptions
   - Discovery information
   - Physical properties
   - Habitability assessment

### Training Examples
Each planet generates multiple training examples:
```
Instruction: "Tell me about the planet Kepler-452b."
Response: "Kepler-452b is an exoplanet orbiting the star Kepler-452..."

Instruction: "Is Kepler-452b potentially habitable?"
Response: "Based on its equilibrium temperature of 265 K..."
```

## 🎯 Model Capabilities

After training, the model can:
- **Identify exoplanets** by name and provide detailed information
- **Explain discovery methods** (transit, radial velocity, etc.)
- **Assess habitability** based on temperature and orbital parameters
- **Compare planetary systems** and their characteristics
- **Analyze stellar properties** and their impact on planetary systems
- **Provide scientific context** for exoplanet discoveries

## 📁 File Structure

```
nasa-planet-llm-training/
├── data_preprocessor.py          # Data preprocessing pipeline
├── ollama_training_script.py     # Main training orchestrator
├── advanced_training.py          # Advanced training with custom parameters
├── evaluate_model.py             # Model evaluation and testing
├── train_ollama_model.sh         # Shell script for simple training
├── Modelfile                     # Generated Ollama Modelfile
├── requirements.txt              # Python dependencies
├── training_data/                # Processed training data
│   ├── nasa_planet_training_data.json
│   ├── nasa_planet_training_data.jsonl
│   └── training_stats.json
└── README.md                     # This file
```

## 🔧 Configuration

### Model Parameters
- **Base Model**: `llama3.2:3b` (optimized for fine-tuning)
- **Temperature**: 0.7 (balanced creativity and accuracy)
- **Top-p**: 0.9 (good diversity)
- **Top-k**: 40 (focused responses)

### Training Parameters
- **Training Examples**: 100-500 per run (adjustable)
- **System Prompt**: Specialized for NASA exoplanet expertise
- **Template Format**: Instruction-response pairs

## 📊 Evaluation

The evaluation script tests the model with questions about:
- Specific exoplanets (Kepler-452b, TRAPPIST-1b, etc.)
- Discovery methods and techniques
- Habitability assessment
- Planetary system analysis
- Scientific concepts

### Evaluation Metrics
- **Response Success Rate**: Percentage of successful responses
- **Average Response Time**: Speed of model inference
- **Answer Quality**: Relevance and accuracy of responses

## 🚀 Usage Examples

### Basic Usage
```bash
# Start Ollama service
ollama serve

# Run the trained model
ollama run nasa-planet-expert "Tell me about Kepler-452b"

# Ask about discovery methods
ollama run nasa-planet-expert "How was TRAPPIST-1b discovered?"
```

### Advanced Queries
```bash
# Habitability assessment
ollama run nasa-planet-expert "Is Proxima Centauri b potentially habitable?"

# Planetary comparison
ollama run nasa-planet-expert "Compare the TRAPPIST-1 system to our solar system"

# Discovery techniques
ollama run nasa-planet-expert "Explain the transit method for exoplanet detection"
```

## 🔬 Scientific Applications

This trained model can assist with:
- **Research**: Quick access to exoplanet data and properties
- **Education**: Teaching exoplanet science concepts
- **Discovery**: Analyzing new planetary candidates
- **Mission Planning**: Understanding target planetary systems
- **Public Outreach**: Explaining exoplanet discoveries to general audiences

## 📈 Performance

Expected performance metrics:
- **Response Time**: < 5 seconds per query
- **Accuracy**: High accuracy on NASA archive data
- **Coverage**: 5,000+ exoplanets in training data
- **Model Size**: ~3B parameters (efficient inference)

## 🤝 Contributing

To improve the model:
1. **Add more training data** from additional NASA sources
2. **Refine training examples** for better instruction-response pairs
3. **Optimize model parameters** for specific use cases
4. **Expand evaluation metrics** for comprehensive testing

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Exoplanet Discovery Methods](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/)

## 📄 License

This project uses NASA's publicly available exoplanet data. Please refer to NASA's data usage policies for commercial applications.

---

**Happy Planet Hunting! 🪐✨**

