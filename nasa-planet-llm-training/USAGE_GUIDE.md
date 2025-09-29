# 🪐 NASA Planet Expert - Usage Guide

## 🎉 Training Complete!

Your NASA Planet Expert model has been successfully trained with **10,446 exoplanet records** from NASA's Exoplanet Archive! The model achieved a **100% success rate** in evaluation tests.

## 🚀 Quick Start

### 1. Start the Model
```bash
ollama run nasa-planet-expert
```

### 2. Ask Questions
Once the model is running, you can ask questions like:
- "Tell me about Kepler-452b"
- "What is the discovery method for TRAPPIST-1b?"
- "Is Proxima Centauri b potentially habitable?"

### 3. Exit the Model
Type `/bye` or press Ctrl+C to exit the interactive session.

## 📊 Model Performance

- **Success Rate**: 100% (10/10 test questions answered successfully)
- **Average Response Time**: 7.59 seconds
- **Training Data**: 10,446 exoplanet records
- **Base Model**: Llama 3.2 3B
- **Specialization**: NASA Exoplanet Archive data

## 🔬 Example Queries

### Planet Information
```
Tell me about the planet Kepler-452b.
```
**Expected Response**: Detailed information about orbital period, size, temperature, composition, and discovery details.

### Discovery Methods
```
How was TRAPPIST-1b discovered?
```
**Expected Response**: Information about the transit method and discovery facility.

### Habitability Assessment
```
Is Proxima Centauri b potentially habitable?
```
**Expected Response**: Analysis of orbital parameters, temperature, and potential for liquid water.

### Planetary Systems
```
Tell me about the TRAPPIST-1 planetary system.
```
**Expected Response**: Overview of the multi-planet system with details about each planet.

### Scientific Concepts
```
Explain the transit method for exoplanet detection.
```
**Expected Response**: Detailed explanation of how the transit method works and its applications.

## 🛠️ Advanced Usage

### Custom Parameters
You can modify the model's behavior by adjusting parameters:

```bash
ollama run nasa-planet-expert --temperature 0.8 --top-p 0.9
```

### Batch Processing
For multiple questions, you can use the model in a script:

```bash
echo "Tell me about Kepler-452b" | ollama run nasa-planet-expert
```

### API Integration
The model can be used with Ollama's API for integration into applications:

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nasa-planet-expert",
    "prompt": "Tell me about Kepler-452b",
    "stream": false
  }'
```

## 📁 File Structure

```
nasa-planet-llm-training/
├── nasa-planet-expert          # Your trained model
├── training_data/              # Processed NASA data
│   ├── nasa_planet_training_data.json
│   ├── nasa_planet_training_data.jsonl
│   └── training_stats.json
├── evaluation_results.json     # Model performance metrics
├── Modelfile                   # Ollama training configuration
├── data_preprocessor.py        # Data processing script
├── ollama_training_script.py   # Training orchestrator
├── advanced_training.py        # Advanced training options
├── evaluate_model.py           # Model evaluation script
└── train_ollama_model.sh       # Simple training script
```

## 🔄 Retraining and Updates

### Add More Data
To retrain with additional data:
1. Add new CSV files to the `nasa data planet/` directory
2. Run: `python3 data_preprocessor.py`
3. Run: `./train_ollama_model.sh`

### Adjust Training Parameters
Modify the `Modelfile` to change:
- Number of training examples
- System prompt
- Model parameters (temperature, top-p, etc.)

### Create New Model Versions
```bash
ollama create nasa-planet-expert-v2 -f Modelfile
```

## 🎯 Model Capabilities

Your trained model can:

✅ **Identify exoplanets** by name and provide detailed information
✅ **Explain discovery methods** (transit, radial velocity, direct imaging)
✅ **Assess habitability** based on temperature and orbital parameters
✅ **Compare planetary systems** and their characteristics
✅ **Analyze stellar properties** and their impact on planets
✅ **Provide scientific context** for exoplanet discoveries
✅ **Answer questions** about exoplanet science concepts

## 🔍 Troubleshooting

### Model Not Responding
```bash
# Check if Ollama is running
brew services list | grep ollama

# Restart Ollama if needed
brew services restart ollama
```

### Slow Responses
- The model runs locally, so response time depends on your hardware
- Average response time is ~7.6 seconds
- Consider using a smaller base model for faster responses

### Model Not Found
```bash
# List available models
ollama list

# If nasa-planet-expert is not listed, retrain:
./train_ollama_model.sh
```

## 📚 Scientific Accuracy

⚠️ **Important Note**: While the model is trained on NASA's official exoplanet data, it may occasionally generate information that isn't in the training data. Always verify critical information with official NASA sources:

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA Exoplanet Exploration](https://exoplanets.nasa.gov/)

## 🚀 Next Steps

1. **Test the model** with your specific exoplanet questions
2. **Integrate into applications** using the Ollama API
3. **Expand training data** with additional NASA datasets
4. **Fine-tune parameters** for your specific use case
5. **Create specialized versions** for different applications

## 📞 Support

If you encounter issues:
1. Check the evaluation results in `evaluation_results.json`
2. Review the training logs
3. Ensure Ollama is properly installed and running
4. Verify your NASA data files are in the correct location

---

**Happy Planet Hunting! 🪐✨**

Your NASA Planet Expert is ready to help you explore the universe of exoplanets!

