# NASA Space Apps Challenge 2025 - Exoplanet AI Detection Platform

## 🚀 Project Overview

This project creates an AI-powered exoplanet detection platform using NASA's Kepler, K2, and TESS mission datasets. The platform combines machine learning with a modern web interface to automatically classify exoplanet candidates.

## 🎯 Challenge Requirements Met

✅ **AI/ML Model**: Trained on NASA exoplanet datasets  
✅ **Web Interface**: Modern Next.js application with high-contrast design  
✅ **Data Analysis**: Processes Kepler, K2, and TESS data  
✅ **User Interaction**: Upload data and get real-time predictions  
✅ **Vercel Compatible**: Ready for deployment  

## 🧠 AI Model Performance

- **Training Samples**: 7,809 exoplanet candidates
- **Test Accuracy**: 83.23%
- **Model Type**: Gradient Boosting Classifier
- **Features**: 15 key astronomical parameters
- **Data Sources**: NASA Kepler, K2, and TESS missions

## 🛠️ Technical Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Modern icon library
- **High Contrast Design** - Black/white theme for accessibility

### Backend
- **Python 3.8+** - Data processing and ML
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation
- **joblib** - Model serialization

### Machine Learning
- **Random Forest** - Ensemble method
- **Gradient Boosting** - Advanced ensemble method
- **Feature Engineering** - Log transformations, mission encoding
- **Cross-validation** - 5-fold validation for robust evaluation

## 📊 Dataset Processing

### Data Sources
1. **Kepler Cumulative** (9,564 rows) - Primary mission data
2. **K2 Planets & Candidates** (4,004 rows) - Extended mission data  
3. **TESS Objects of Interest** (7,699 rows) - Transit survey data

### Feature Engineering
- **Orbital Period** - Planet's year length
- **Transit Duration** - Time spent transiting star
- **Transit Depth** - Light dimming during transit
- **Planetary Radius** - Size relative to Earth
- **Stellar Properties** - Temperature, gravity, radius
- **Mission Encoding** - Categorical mission data

### Data Quality
- **Cleaned Dataset**: 7,809 high-quality samples
- **Exoplanet Ratio**: 56.1% confirmed exoplanets
- **Missing Data**: Handled with median imputation
- **Outliers**: Removed beyond 3 standard deviations

## 🌐 Web Application Features

### Pages
1. **Home Page** - Landing page with project overview
2. **Prediction Interface** - Upload data and get AI analysis
3. **Dashboard** - Model performance and statistics

### Key Features
- **Real-time Prediction** - Instant exoplanet classification
- **Confidence Scores** - Probability-based confidence levels
- **Model Information** - Training statistics and performance
- **Responsive Design** - Works on all device sizes
- **Accessibility** - High contrast, keyboard navigation

## 🔧 API Endpoints

### GET /api/predict
Returns model information and metadata

### POST /api/predict
Accepts exoplanet candidate data and returns classification:
```json
{
  "koi_period": 365.25,
  "koi_prad": 1.0,
  "koi_steff": 5778,
  "mission": "Kepler"
}
```

## 🚀 Deployment

### Vercel Deployment
1. Push code to GitHub repository
2. Connect repository to Vercel
3. Automatic deployment on push to main branch
4. Environment variables handled automatically

### Local Development
```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Prepare data and train model
python scripts/prepare_data.py
python scripts/train_model.py

# Start development server
npm run dev
```

## 📈 Model Evaluation

### Performance Metrics
- **Precision**: 84% (True Positives / All Positives)
- **Recall**: 87% (True Positives / All Actual Positives)
- **F1-Score**: 85% (Harmonic mean of precision and recall)
- **Cross-validation**: 82.87% ± 0.74%

### Classification Report
```
              precision    recall  f1-score   support
         0.0       0.83      0.78      0.80       686
         1.0       0.84      0.87      0.85       876
    accuracy                           0.83      1562
   macro avg       0.83      0.83      0.83      1562
weighted avg       0.83      0.83      0.83      1562
```

## 🎨 Design Philosophy

### High Contrast Theme
- **Background**: Pure black (#000000)
- **Text**: Pure white (#FFFFFF)
- **Accents**: Gray gradients for depth
- **Accessibility**: WCAG AA compliant contrast ratios

### User Experience
- **Intuitive Interface** - Clear data input forms
- **Visual Feedback** - Confidence indicators and status
- **Responsive Layout** - Mobile-first design
- **Fast Loading** - Optimized assets and code splitting

## 🔬 Scientific Accuracy

### Data Validation
- **NASA Sources** - Official exoplanet archive data
- **Quality Control** - Multiple validation steps
- **Feature Selection** - Astronomically relevant parameters
- **Cross-mission** - Consistent data across missions

### Model Interpretability
- **Feature Importance** - Understanding which parameters matter
- **Confidence Scores** - Uncertainty quantification
- **Error Analysis** - Understanding model limitations
- **Validation** - Independent test set evaluation

## 🏆 NASA Space Apps Challenge Alignment

This project directly addresses the challenge requirements:

1. **AI/ML Model** ✅ - Trained on NASA datasets with 83% accuracy
2. **Web Interface** ✅ - Modern, accessible web application
3. **Data Analysis** ✅ - Processes multiple NASA mission datasets
4. **User Interaction** ✅ - Upload data and get predictions
5. **Research Impact** ✅ - Helps identify new exoplanets

## 🔮 Future Enhancements

- **Real-time Data** - Connect to live telescope feeds
- **Advanced Models** - Deep learning approaches
- **Multi-class** - Distinguish planet types
- **Visualization** - Interactive data plots
- **Collaboration** - Share findings with researchers

## 📚 Resources

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **Kepler Mission**: https://www.nasa.gov/mission_pages/kepler/
- **TESS Mission**: https://tess.mit.edu/
- **Project Repository**: [GitHub Link]

---

**Built for NASA Space Apps Challenge 2025**  
*Advancing exoplanet discovery through artificial intelligence*
