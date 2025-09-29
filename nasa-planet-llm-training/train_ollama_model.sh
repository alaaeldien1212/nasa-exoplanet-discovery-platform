#!/bin/bash

# NASA Planet Expert Model Training Script
# This script trains a custom Ollama model using NASA exoplanet data

echo 'Starting NASA Planet Expert Model Training...'

# Check if Ollama is running
if ! pgrep -x 'ollama' > /dev/null; then
    echo 'Starting Ollama service...'
    ollama serve &
    sleep 5
fi

# Create the model
echo 'Creating model nasa-planet-expert...'
ollama create nasa-planet-expert -f Modelfile

# Test the model
echo 'Testing model nasa-planet-expert...'
ollama run nasa-planet-expert 'Tell me about the planet Kepler-452b.'

echo 'Training complete!'
echo 'You can now use the model with: ollama run nasa-planet-expert'
