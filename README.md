# BARLO: Bayani Alert and Response for Local Operations

**We can forecast where a storm will go, but not how much it will destroy.**

Predict a storm's economic impact from typhoon forecast data. Get insights on how to pre-emptively place logistics. 
Deployed on Streamlit, this Python app uses a machine learning model trained with PyTorch and skLearn on historical storm data

## Key Features

- **Live Forecast Impact Assessment** - Get the latest storm forecast from sources such [US Naval Meteorology and Oceanography Command](https://www.metoc.navy.mil/) and get insights hours before landfall.
- **Economic Impact Modeling**: Calculate estimated economic, humanitarian, and  losses based on storm intensity, proximity, and direction, detected based on the plain language warnings of the [Joint Typhoon Warning Center](https://www.metoc.navy.mil/jtwc/jtwc.html)
- **Comprehensive Insights**: Charts, maps, and tables for comprehensive analysis based on historical data.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/smmariquit/pjdsc-economic-impact
cd pjdsc

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run src/app/main.py
```

Visit: http://localhost:8501

**Online Deployment**: https://barlo-pjdsc.streamlit.app

### Run Real-Time Storm Prediction

```bash
# Fetch live JTWC data and predict impacts
python -m src.pipeline.deploy_both_models --storm-id wp3025

# Or use a sample forecast
python -m src.pipeline.deploy_both_models --forecast data/samples/storm_forecast.txt
```

## Project Structure

```
pjdsc/
├── src/                    # Source code
│   ├── app/               # Streamlit application
│   ├── pipeline/          # ML inference pipeline
│   ├── training/          # Model training
│   ├── utils/             # Shared utilities
│   └── scripts/           # Utility scripts
│
├── data/                   # Data files
│   ├── raw/               # Original data (storms, impacts, weather)
│   ├── processed/         # Processed features
│   └── samples/           # Sample files
│
├── models/                 # Trained models
│   ├── persons/           # Persons affected model
│   ├── houses/            # Houses damaged model
│   ├── optimized/         # Optimized models
│   └── minimal/           # Minimal models
│
├── outputs/                # Generated predictions
│
├── docs/                   # Documentation
│   ├── pipeline/          # Pipeline documentation
│   └── feature_engineering/  # Feature engineering docs
│
├── analysis/               # Data analysis scripts
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## Model Performance

### Persons Affected Model
- **Classifier**: AUC-PR 0.9834, F1 0.9762
- **Regressor**: R-squared 0.7538, MAE 0.3312

### Houses Damaged Model
- **Classifier**: AUC-PR 0.9983, F1 0.9928
- **Regressor**: R-squared 0.6502, MAE 0.7125

Dataset: 285 storms (2010-2024), 23,653 storm-province pairs

## Documentation

- **[Quick Start](docs/QUICK_START.md)** - Get started quickly
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Train new models
- **[ML Pipeline](docs/ML_PIPELINE_README.md)** - Complete ML pipeline docs
- **[Pipeline Documentation](docs/pipeline/README.md)** - Pipeline API reference
- **[Data Flow](docs/DATA_FLOW_DIAGRAM.md)** - System architecture

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set main file: `src/app/main.py`
4. Add secrets (OpenAI API key) in Streamlit dashboard

### Local Development

```bash
pip install -r requirements.txt
streamlit run src/app/main.py
```
