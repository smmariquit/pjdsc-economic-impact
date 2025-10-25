# 🌪️ Storm Impact Prediction System

A complete machine learning pipeline for predicting tropical storm impacts on Philippine provinces using real-time forecast data.

---

## 📋 **Overview**

This system predicts **dual humanitarian impacts** from tropical storms at the province level:
1. **Persons Affected** - Number of people requiring assistance
2. **Houses Damaged** - Number of houses destroyed or damaged

**Data Sources:**
- JTWC forecast bulletins (storm track & intensity)
- Open-Meteo weather forecasts (wind, precipitation)
- Historical storm data and impacts
- Province characteristics (population, geography, vulnerability)

**Model Architecture:** Two independent two-stage cascades
- **Stage 1 (Classifier):** Impact vs No Impact (binary classification)
- **Stage 2 (Regressor):** Magnitude of impact (persons or houses)

**Confidence Level:** HIGH (with complete weather data)

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Install dependencies
pip install -r requirements.txt
```

### **2. Real-Time Prediction (Dual Models)**

#### **Option A: Live JTWC Data (One Command)** ⭐ **RECOMMENDED**

```bash
# Fetch live data and predict BOTH persons + houses in one step
python pipeline/deploy_both_models.py --storm-id wp3025
```

#### **Option B: Using Sample/Local Forecast**

```bash
# Use included sample (Tropical Storm Fengshen snapshot)
python pipeline/deploy_both_models.py --forecast storm_forecast.txt
```

**Data Source:** [JTWC (Joint Typhoon Warning Center)](https://www.metoc.navy.mil/jtwc/products/wp3025web.txt)

**Output:** All results in `output/` folder
- `predictions_DUAL_STORM_YEAR.csv` - Full predictions (both models)
- `summary_DUAL_STORM_YEAR.txt` - Human-readable report
- `intermediate/` - Processing files (track, weather)

---

## 📂 **Project Structure**

```
Final_Transform/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── storm_forecast.txt                 ← Sample JTWC forecast (snapshot)
├── fetch_jtwc_bulletin.py             ← Fetch live JTWC data
│
├── 📁 Data Directories
│   ├── Storm_data/                    ← Historical storm tracks
│   ├── Weather_location_data/         ← Historical weather (per storm)
│   ├── Impact_data/                   ← Historical impacts (labels)
│   ├── Population_data/               ← Population & density
│   ├── Location_data/                 ← Province coordinates
│   └── Feature_Engineering_Data/      ← Pre-computed features
│
├── 📁 Feature Engineering
│   └── Feature_Engineering/
│       ├── group1/ - Distance features (26)
│       ├── group2/ - Weather features (20)
│       ├── group3/ - Intensity features (7)
│       ├── group6/ - Motion features (10)
│       ├── group7/ - Interaction features (6)
│       └── group8/ - Multi-storm features (6)
│
├── 📁 Pipeline (Real-Time Inference)
│   └── pipeline/
│       ├── deploy_both_models.py          ← Main (DUAL MODEL - Persons + Houses) ⭐
│       ├── complete_forecast_pipeline.py  ← Single model (Persons only)
│       ├── parse_jtwc_forecast.py         ← JTWC parser
│       ├── fetch_forecast_weather.py      ← Weather API client
│       ├── unified_pipeline.py            ← Feature extraction
│       └── README.md                      ← Pipeline documentation
│
├── 📁 Training
│   └── training/
│       ├── data_loader.py                 ← Load & merge features
│       ├── split_utils.py                 ← Train/val/test split
│       ├── train.py                       ← Train persons model
│       └── train_houses.py                ← Train houses model
│
├── 📁 Model Artifacts
│   ├── artifacts/                         ← Persons Model
│   │   ├── stage1_classifier.joblib       ← Classifier
│   │   ├── stage2_regressor.joblib        ← Regressor
│   │   └── feature_columns.json           ← Feature list
│   └── artifacts_houses/                  ← Houses Model
│       ├── stage1_classifier_houses.joblib
│       ├── stage2_regressor_houses.joblib
│       └── feature_columns_houses.json
│
├── 📁 Output (Generated)
│   └── output/
│       ├── predictions_*.csv              ← Full predictions
│       ├── alerts_*.json                  ← Alert format
│       ├── summary_*.txt                  ← Report
│       └── intermediate/                  ← Processing files
│
└── 📁 Documentation
    ├── QUICK_START.md                     ← Quick reference guide
    ├── TRAINING_GUIDE.md                  ← Model training guide
    ├── DATA_FLOW_DIAGRAM.md               ← System architecture
    ├── DEPLOYMENT_READINESS.md            ← Deployment checklist
    ├── Implementation_Plan_and_Analysis.md ← Project analysis
    └── Project_desc.md                    ← Detailed project description
```

---

## 🎯 **Key Features**

### **Real-Time Forecasting**
- ✅ Parse JTWC forecast bulletins
- ✅ Fetch weather forecasts (Open-Meteo API)
- ✅ Extract 80 features (6 groups + population + vulnerability)
- ✅ Generate predictions for all 83 provinces
- ✅ Export in CSV & JSON formats

### **Complete Feature Engineering**
- **GROUP 1 (26):** Distance & proximity (min/max/mean distance, hours under threshold, etc.)
- **GROUP 2 (20):** Weather exposure (wind, precipitation, hazard combinations)
- **GROUP 3 (7):** Storm intensity (peak wind, pressure, intensification rate)
- **GROUP 6 (10):** Storm motion (speed, direction, track shape)
- **GROUP 7 (6):** Interactions (distance × weather × intensity)
- **GROUP 8 (6):** Multi-storm context (recent storms, concurrent storms)
- **Population (2):** Demographics per province
- **Vulnerability (3):** Historical impact patterns

### **Production Ready**
- ✅ Single command execution
- ✅ Automatic data caching
- ✅ Error handling & validation
- ✅ Multiple output formats
- ✅ Comprehensive logging

---

## 📊 **Model Performance**

**Test Set (Stratified storm split):**

### **Persons Affected Model**

| Model | Metric | Value |
|-------|--------|-------|
| **Classifier** | AUC-PR | 0.9834 |
| | Precision | 0.9650 |
| | Recall | 0.9876 |
| | F1-Score | 0.9762 |
| **Regressor** | RMSE | 0.5842 |
| | MAE | 0.3312 |
| | R² | 0.7538 |

### **Houses Damaged Model**

| Model | Metric | Value |
|-------|--------|-------|
| **Classifier** | AUC-PR | **0.9983** ⭐ |
| | Precision | 0.9981 |
| | Recall | 0.9876 |
| | F1-Score | 0.9928 |
| **Regressor** | RMSE | 2.6287 |
| | MAE | 0.7125 |
| | R² | 0.6502 |

**Dataset:** 285 storms (2010-2024), 23,653 storm-province pairs
**Both models:** Same 80 features, independent predictions

---

## 🔧 **Usage Examples**

### **Real-Time Prediction (DUAL MODELS - Recommended)**

```bash
# Live mode (ONE COMMAND - fetch and predict BOTH models)
python pipeline/deploy_both_models.py --storm-id wp3025

# File mode (use sample or local file)
python pipeline/deploy_both_models.py --forecast storm_forecast.txt
```

**Output:**
- `output/predictions_DUAL_FENGSHEN_2025.csv` - Predictions for all 83 provinces
- `output/summary_DUAL_FENGSHEN_2025.txt` - Human-readable report
- `output/intermediate/` - Processing files (track, weather)

### **Single Model Prediction (Legacy)**

```bash
# Persons only (legacy single model pipeline)
python pipeline/complete_forecast_pipeline.py --storm-id wp3025
```

### **Training New Models**

```bash
# Train PERSONS model (recommended settings)
python -m training.train \
  --split_mode stratified_storm \
  --out_dir artifacts

# Train HOUSES model
python -m training.train_houses \
  --split_mode stratified_storm \
  --out_dir artifacts_houses

# Train with temporal split (alternative)
python -m training.train \
  --split_mode temporal \
  --train_range 2010-2018 \
  --val_range 2019-2020 \
  --test_range 2021-2024
```

### **Process Historical Storm**

```bash
# Extract features for a past storm
python pipeline/unified_pipeline.py \
  --year 2024 \
  --storm Kristine \
  --output features_kristine.csv
```

---

## 📈 **Data Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ REAL-TIME FORECAST                                          │
│ (JTWC Bulletin)                                             │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Parse Forecast                                      │
│  • Extract track points (LAT/LON/time)                      │
│  • Extract wind intensity                                   │
│  • Extract forecast periods (12hr, 24hr, 36hr, etc.)        │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Fetch Weather Forecast                              │
│  • Open-Meteo API (16-day forecast)                         │
│  • Daily wind & precipitation per province                  │
│  • Cache results (1-hour TTL)                               │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Extract Features                                    │
│  • GROUP 1: Distance features (track → provinces)           │
│  • GROUP 2: Weather features (API data)                     │
│  • GROUP 3: Intensity features (track data)                 │
│  • GROUP 6: Motion features (track evolution)               │
│  • GROUP 7: Interaction features (1×2×3)                    │
│  • GROUP 8: Multi-storm features (historical database)      │
│  • Population & Vulnerability (static data)                 │
│  → TOTAL: 80 features × 83 provinces                        │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Model Inference                                     │
│  • Stage 1: Classifier (impact yes/no)                      │
│  • Stage 2: Regressor (magnitude)                           │
│  • Combine: Final predictions                               │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Generate Outputs                                    │
│  • CSV: Full predictions (all provinces)                    │
│  • JSON: Alerts (API-ready format)                          │
│  • TXT: Summary report (human-readable)                     │
│  → Saved to: output/                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 **Documentation**

| Document | Description |
|----------|-------------|
| **QUICK_START.md** | Quick reference for common tasks |
| **TRAINING_GUIDE.md** | Complete model training guide |
| **DATA_FLOW_DIAGRAM.md** | System architecture & data flow |
| **pipeline/README.md** | Pipeline usage & API reference |
| **pipeline/REALTIME_FORECAST_GUIDE.md** | Real-time prediction guide |
| **pipeline/JTWC_DATA_SOURCE.md** | ⭐ JTWC data fetching guide |
| **pipeline/FORECAST_ANALYSIS.md** | Forecast data analysis |
| **pipeline/OUTPUT_STRUCTURE.md** | Output folder structure |
| **Implementation_Plan_and_Analysis.md** | Project planning & analysis |
| **Project_desc.md** | Detailed project description |

---

## ⚙️ **Configuration**

### **Dependencies**

```txt
pandas>=1.5.0
numpy<2.0,>=1.24
scikit-learn>=1.3.0
xgboost<2.0,>=1.7
joblib>=1.3.0
openmeteo-requests>=1.0.0
requests-cache>=1.0.0
retry-requests>=2.0.0
```

### **System Requirements**

- **Python:** 3.8+
- **Memory:** 4GB+ RAM
- **Storage:** 5GB+ (for historical data)
- **Network:** Internet access (for weather API)

### **API Configuration**

**Open-Meteo API:**
- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Rate Limit:** 10,000 requests/day (free tier)
- **Our Usage:** ~10 requests per storm (batched)
- **API Key:** Not required

---

## 🔍 **Model Details**

### **Training Data**

- **Period:** 2010-2024
- **Storms:** 285 named storms
- **Observations:** 23,653 storm-province pairs
- **Features:** 80 engineered features
- **Labels:** Persons affected per province

### **Data Split (Stratified Storm)**

- **Train:** 80% (226 storms, 18,756 observations)
- **Validation:** 10% (27 storms, 2,241 observations)
- **Test:** 10% (32 storms, 2,656 observations)

**Split Strategy:** By storm (not time), stratified by impact severity to ensure balanced representation across splits.

### **Feature Importance**

**Top 10 Features:**
1. `min_distance_km` - Minimum distance to storm
2. `max_wind_gust_kmh` - Peak wind gusts
3. `total_precipitation_mm` - Total rainfall
4. `hist_max_affected` - Historical vulnerability
5. `proximity_peak` - Maximum proximity metric
6. `max_wind_in_track_kt` - Storm peak intensity
7. `integrated_proximity` - Cumulative exposure
8. `compound_hazard_score` - Combined wind+rain
9. `Population` - Province population
10. `hours_under_200km` - Duration of proximity

---

## 🚨 **Known Limitations**

### **Data Quality**

1. **Weather Data Dependency:** Predictions require real-time weather forecasts (Open-Meteo). Track-only mode available but less accurate (~6x underestimation).

2. **Historical Bias:** Model trained on 2010-2024 data. Storm patterns/impacts may change over time due to climate change, infrastructure improvements, etc.

3. **Vulnerability Features:** Currently default to 0 for new storms. Could be improved by computing from historical database.

### **Model Limitations**

1. **High Predictions:** Model may show high confidence (>99%) for some provinces, indicating potential over-confidence or class imbalance issues.

2. **Province-Level Only:** Predictions are at province level, not municipal/barangay level.

3. **Impact Definition:** Model predicts "persons affected" (evacuated, displaced, etc.), not casualties or specific damage types.

### **Operational Considerations**

1. **Forecast Accuracy:** Model depends on JTWC forecast accuracy. Forecast errors propagate to predictions.

2. **Lead Time:** Requires forecast bulletin (typically 5-7 days before landfall). Not suitable for very short-term (0-12 hour) predictions.

3. **API Dependency:** Requires internet access for weather API. Cached for resilience.

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**Error: "Storm not found"**
```bash
# Check storm name spelling (case-sensitive, use Philippine name)
grep "Kristine" Storm_data/ph_storm_data.csv
```

**Error: "Weather file not found"**
```bash
# Ensure using complete_forecast_pipeline.py (fetches automatically)
# Or manually fetch weather:
python pipeline/fetch_forecast_weather.py --from-jtwc storm_forecast.txt
```

**Error: "No module named openmeteo_requests"**
```bash
# Install missing dependency
pip install openmeteo-requests requests-cache retry-requests
```

**Low/High Predictions**
- Verify weather data was included (check `feature_mode` in output)
- Compare COMPLETE vs TRACK-ONLY modes
- Check model artifacts are present in `artifacts/`

---

## 📖 **Citation**

If you use this system in your research or operations, please cite:

```
Storm Impact Prediction System for the Philippines
[Your Organization/Institution]
2024
```

---

## 📞 **Support**

- **Documentation:** See `/docs` folder
- **Issues:** Check `TROUBLESHOOTING` section in guides
- **Updates:** Check git history for latest changes

---

## ⚖️ **License**

[Your License Here]

---

## 🙏 **Acknowledgments**

- **Data Sources:**
  - [JTWC (Joint Typhoon Warning Center)](https://www.metoc.navy.mil/jtwc/) - Storm forecasts & tracks
  - [Open-Meteo API](https://open-meteo.com/) - Weather forecasts
  - NDRRMC - Historical impact data
  - PSA - Population statistics

- **Technologies:**
  - Python, pandas, scikit-learn, XGBoost
  - Open-Meteo API

---

**Last Updated:** 2025-10-23  
**Version:** 1.0  
**Status:** ✅ Production Ready


