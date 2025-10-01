# NASA Exoplanet AI Detection Platform

A Next.js application for AI-powered exoplanet detection using NASA datasets.

## Features

- **AI-Powered Detection**: Machine learning models trained on NASA Kepler, K2, and TESS data
- **Real-time Analysis**: Upload transit data and get instant exoplanet classification
- **High Accuracy**: 83%+ accuracy on test data
- **Modern UI**: Black and white high-contrast design optimized for accessibility
- **Vercel Ready**: Configured for seamless deployment

## Tech Stack

- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: Python, scikit-learn, pandas
- **ML Models**: Random Forest, Gradient Boosting
- **Data**: NASA Kepler, K2, and TESS datasets

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.8+
- pip

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. Prepare data and train models:
   ```bash
   python scripts/prepare_data.py
   python scripts/train_model.py
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

## Project Structure

```
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── dashboard/          # Model dashboard page
│   ├── predict/            # Prediction interface
│   └── page.tsx            # Home page
├── scripts/                # Python scripts
│   ├── prepare_data.py     # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── predict.py          # Prediction script
│   └── model_info.py       # Model information
├── lib/                    # Python libraries
│   └── predictor.py        # Prediction class
├── models/                 # Trained models
├── data/                   # Processed datasets
└── nasa data planet/      # Original NASA datasets
```

## API Endpoints

- `GET /api/predict` - Get model information
- `POST /api/predict` - Make exoplanet prediction

## Deployment

The project is configured for Vercel deployment:

1. Connect your GitHub repository to Vercel
2. Deploy automatically on push to main branch
3. Environment variables are handled automatically

## Model Performance

- **Training Samples**: 7,809 exoplanet candidates
- **Test Accuracy**: 83.23%
- **Features**: 15 key astronomical parameters
- **Data Sources**: Kepler, K2, and TESS missions

## Contributing

This project was created for the 2025 NASA Space Apps Challenge. Contributions are welcome!

## License

MIT License - see LICENSE file for details.
