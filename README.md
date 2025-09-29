# 🪐 NASA Space Apps Challenge 2025 - Exoplanet Discovery Platform

## Challenge #2: AI/ML Model for Exoplanet Discovery

**Team:** NASA Exoplanet Discovery Platform Development Team  
**Event:** 2025 NASA Space Apps Challenge  
**Difficulty:** Advanced  
**Subjects:** Artificial Intelligence & Machine Learning, Data Analysis, Data Visualization, Software

---

## 🚀 Project Overview

An advanced AI-powered platform for exoplanet classification and discovery, combining machine learning algorithms with NASA's comprehensive exoplanet datasets to classify planetary candidates as confirmed planets, candidates, or false positives.

### 🏆 **Key Achievement: 86.15% Cross-Validation Accuracy**

---

## ✨ Features

### 🤖 **Machine Learning Classification**
- **Multiple ML Models**: Gradient Boosting, Random Forest, Extra Trees, SVM, Histogram GB
- **Advanced Preprocessing**: Feature engineering, outlier detection, class imbalance handling
- **Ensemble Methods**: Voting classifier with 5 high-performance models
- **Real-time Predictions**: Instant classification of planetary candidates

### 🌐 **Interactive Web Interface**
- **Manual Prediction**: Input planetary parameters for single predictions
- **Batch Processing**: Upload CSV files for bulk classification
- **Drag & Drop**: Easy file upload with validation
- **Responsive Design**: Works on desktop, tablet, and mobile

### 📊 **Analytics Dashboard**
- **Model Performance**: Accuracy metrics and comparison charts
- **Feature Importance**: Understanding which parameters matter most
- **Data Visualization**: Interactive charts and distributions
- **Real-time Monitoring**: Live model status and performance tracking

### 🧠 **AI Explanations**
- **NASA Planet Expert LLM**: Trained on NASA data for scientific explanations
- **Detailed Insights**: Why a classification was made
- **Educational Content**: Learning about exoplanet science
- **Confidence Scoring**: Probability distributions for predictions

---

## 📊 **Performance Metrics**

### **Model Performance**
- **Cross-Validation Accuracy**: 86.15%
- **Test Accuracy**: 79.54%
- **Training Data**: 21,267 exoplanet records
- **Features**: 42 planetary parameters
- **Processing Speed**: < 1 second per prediction

### **Data Coverage**
- **NASA Missions**: Kepler, K2, TESS
- **Training Records**: 21,267 exoplanet observations
- **Feature Dimensions**: 42 parameters
- **Classification Types**: 3 categories (CONFIRMED, CANDIDATE, FALSE_POSITIVE)

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- NASA exoplanet data files
- Ollama (optional, for AI explanations)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/rhythmlab/nasa-exoplanet-discovery-platform.git
   cd nasa-exoplanet-discovery-platform
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the platform**
   ```bash
   python start_platform.py
   ```

4. **Open your browser**
   ```
   http://localhost:8000
   ```

---

## 📁 **Project Structure**

```
nasa-exoplanet-discovery-platform/
├── app.py                      # FastAPI web application
├── final_high_accuracy_classifier.py  # ML classifier (86.15% accuracy)
├── start_platform.py           # Platform startup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── NASA_Space_Apps_Challenge_2_Submission_Report.md  # Submission report
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   ├── predict.html           # Prediction interface
│   ├── upload.html            # Data upload interface
│   └── dashboard.html         # Analytics dashboard
├── static/                     # Static files (CSS, JS)
│   └── nasa-logo.png          # NASA logo
└── nasa-planet-llm-training/   # AI explanation system
    ├── ollama_training_script.py
    └── training_data/
```

---

## 🎯 **Usage**

### **Manual Prediction**
1. Navigate to **Predict** page
2. Enter planetary parameters:
   - Orbital Period (days)
   - Planetary Radius (Earth radii)
   - Stellar Temperature (K)
   - Stellar Radius (Solar radii)
   - Stellar Mass (Solar masses)
   - Distance (parsecs)
3. Select ML model
4. Click "Classify Exoplanet"
5. View results with confidence scores and explanations

### **Batch Processing**
1. Navigate to **Upload** page
2. Upload CSV file with planetary data
3. Select ML model and processing options
4. Download results with classifications

### **Analytics Dashboard**
1. Navigate to **Dashboard** page
2. View model performance metrics
3. Analyze feature importance
4. Perform hyperparameter tuning
5. Monitor data distributions

---

## 🔬 **Scientific Applications**

### **Research Use Cases**
- **Planetary Discovery**: Classify new exoplanet candidates
- **Data Validation**: Verify existing classifications
- **Mission Planning**: Prioritize follow-up observations
- **Statistical Analysis**: Understand planetary populations

### **Educational Applications**
- **Student Projects**: Learn exoplanet science through AI
- **Research Training**: Understand ML in astronomy
- **Data Analysis**: Practice with real NASA datasets
- **Scientific Method**: Hypothesis testing with AI

### **Professional Tools**
- **Observatory Operations**: Rapid candidate screening
- **Grant Proposals**: Demonstrate discovery potential
- **Publication Support**: Statistical validation of results
- **Collaboration**: Share standardized classifications

---

## 🛠️ **API Endpoints**

### **Prediction**
- `POST /api/predict` - Manual prediction
- `POST /api/upload` - Batch prediction
- `GET /api/models/status` - Model status

### **Analytics**
- `GET /api/model-performance` - Performance metrics
- `GET /api/feature-importance` - Feature analysis
- `POST /api/hyperparameter-tuning` - Model optimization

### **Management**
- `GET /api/retrain` - Retrain models
- `GET /download/{filename}` - Download results
- `GET /api/llm/explain` - AI explanations

---

## 📚 **References**

### **NASA Data Sources**
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/)
- [TESS Mission](https://tess.mit.edu/)
- [K2 Mission](https://www.nasa.gov/feature/ames/kepler/k2-mission)

### **Scientific Papers**
- "The Exoplanet Archive: Data and Tools for Exoplanet Research"
- "Machine Learning for Exoplanet Classification"
- "Transit Detection and Characterization"

---

## 🏆 **Challenge Requirements Met**

✅ **AI/ML Model Trained on NASA Data** - 86.15% accuracy achieved  
✅ **Data Analysis for New Planet Identification** - Comprehensive feature engineering  
✅ **Web Interface for User Interaction** - Full-featured web application  
✅ **Bonus Features**: Batch processing, model monitoring, AI explanations

---

## 📄 **License**

This project uses NASA's publicly available exoplanet data. Please refer to NASA's data usage policies for commercial applications.

---

## 🆘 **Support**

### **Getting Help**
- Check the submission report: `NASA_Space_Apps_Challenge_2_Submission_Report.md`
- Review the platform summary: `PLATFORM_SUMMARY.md`
- Open an issue on GitHub
- Contact the development team

---

**Happy Planet Hunting! 🪐✨**

*Built with ❤️ for the exoplanet discovery community*

**Ready for NASA Space Apps Challenge Submission!**
