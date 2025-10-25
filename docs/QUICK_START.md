# 🚀 Quick Start Guide

## **30 Second Start - DUAL MODELS** ⭐

```bash
# One command - fetch live data and predict BOTH persons + houses!
python pipeline/deploy_both_models.py --storm-id wp3025

# That's it! Check output/ folder for results
```

---

## **Common Commands**

### **Real-Time Prediction (DUAL MODELS - Recommended)**

```bash
# Live mode (ONE COMMAND - BOTH predictions)
python pipeline/deploy_both_models.py --storm-id wp3025

# File mode (use sample forecast)
python pipeline/deploy_both_models.py --forecast storm_forecast.txt
```

### **Single Model (Legacy - Persons Only)**

```bash
# Live mode
python pipeline/complete_forecast_pipeline.py --storm-id wp3025

# File mode
python pipeline/complete_forecast_pipeline.py --forecast storm_forecast.txt

# Custom output directory
python pipeline/complete_forecast_pipeline.py \
  --storm-id wp3025 \
  --output-dir results_2025
```

### **Advanced: Separate Fetch (Optional)**

  ```bash
  # If you want to fetch bulletin separately
  python fetch_jtwc_bulletin.py --storm-id wp3025

# Then use it
python pipeline/complete_forecast_pipeline.py --forecast storm_forecast_wp3025.txt
```

### **Train Models**

```bash
# Train PERSONS model (stratified storm split - recommended)
python -m training.train --split_mode stratified_storm

# Train HOUSES model
python -m training.train_houses --split_mode stratified_storm

# Temporal split (alternative)
python -m training.train \
  --split_mode temporal \
  --train_range 2010-2018 \
  --val_range 2019-2020 \
  --test_range 2021-2024
```

### **Historical Storm**

```bash
# Extract features
python pipeline/unified_pipeline.py --year 2024 --storm Kristine

# Generate predictions
python pipeline/example_deployment.py --year 2024 --storm Kristine
```

---

## **Output**

### **Dual Model Output** (in `output/` folder):
- `predictions_DUAL_STORM_YEAR.csv` - Full predictions (persons + houses)
- `summary_DUAL_STORM_YEAR.txt` - Combined report
- `intermediate/` - Processing files

### **Single Model Output** (legacy):
- `predictions_STORM_YEAR.csv` - Persons only
- `alerts_STORM_YEAR.json` - Alert format  
- `summary_STORM_YEAR.txt` - Report

---

## **Next Steps**

- **Full Guide:** See `README.md`
- **Training:** See `TRAINING_GUIDE.md`
- **Pipeline:** See `pipeline/README.md`
- **JTWC Data:** See `pipeline/JTWC_DATA_SOURCE.md` ⭐
- **Real-time Guide:** See `pipeline/REALTIME_FORECAST_GUIDE.md`

