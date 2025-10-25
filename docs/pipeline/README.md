# 🌊 Single Storm Feature Extraction Pipeline

## Overview

This pipeline takes **RAW INPUT** for one storm and generates **ML-ready features** for all 81 provinces.

```
RAW DATA → FEATURE ENGINEERING → MODEL INPUT
```

---

## 📥 **Input Data Requirements**

### **Required Files:**

1. **Storm Track** (from `Storm_data/ph_storm_data.csv`)
   - Contains: LAT, LON, datetime, wind, pressure, etc.
   - Format: 3-6 hour intervals

2. **Weather Data** (from `Weather_location_data/{year}/{year}_{storm}.csv`)
   - Contains: Daily wind, precipitation per province
   - Format: One row per province per day

3. **Static Data:**
   - Population: `Population_data/population_density_all_years.csv`
   - Locations: `Location_data/locations_latlng.csv`

4. **Historical Context** (for Group8 only):
   - Past storms from `Storm_data/ph_storm_data.csv`
   - Automatically extracted for past 90 days

---

## 🚀 **Usage**

### **Quick Start - Process One Storm:**

```bash
python pipeline/unified_pipeline.py --year 2024 --storm Kristine --output features_kristine.csv
```

### **Without Group8 (faster, no historical context):**

```bash
python pipeline/unified_pipeline.py --year 2024 --storm Kristine --no-group8
```

### **Batch Process Multiple Storms:**

```bash
# Example: Process all 2024 storms
for storm in Kristine Leon Marce Nika Ofel Pepito; do
  python pipeline/unified_pipeline.py --year 2024 --storm $storm --output "batch_output/2024_${storm}.csv"
done
```

---

## 📦 **Output Format**

**Columns:**
```
Year, Storm, Province, 
group1__min_distance_km, group1__max_distance_km, ... (26 distance features)
group2__max_wind_gust_kmh, group2__total_precipitation_mm, ... (20 weather features)
group3__max_wind_in_track_kt, ... (7 intensity features)
group6__mean_forward_speed, ... (10 motion features)
group7__distance_x_wind, ... (6 interaction features)
group8__storms_past_30_days, ... (6 multi-storm features)
Population, PopulationDensity
```

**Rows:**
- One row per province (81 rows for Philippines)
- All features for that storm-province combination

---

## 🏗️ **Architecture**

### **Pipeline Flow:**

```
1. LOAD RAW DATA
   ├─ Storm track (3-6hr positions)
   ├─ Weather data (daily per province)
   └─ Static data (population, locations)

2. EXTRACT FEATURES BY GROUP
   ├─ GROUP 1: Distance/proximity (26 features)
   │   └─ Uses: group1_distance_feature_engineering.py
   │
   ├─ GROUP 2: Weather exposure (20 features)
   │   └─ Uses: group2_weather_feature_engineering.py
   │
   ├─ GROUP 3: Storm intensity (7 features)
   │   └─ Uses: group3_storm_intensity_features.py
   │
   ├─ GROUP 6: Storm motion (10 features)
   │   └─ Uses: group6_storm_motion_features.py
   │
   ├─ GROUP 7: Interactions (6 features)
   │   └─ Uses: group7_interaction_features.py
   │
   └─ GROUP 8: Multi-storm context (6 features) *OPTIONAL*
       └─ Uses: group8_multistorm_features.py
       └─ Requires: Historical storms (past 90 days)

3. MERGE ALL GROUPS
   └─ Join on [Year, Storm, Province]

4. ADD POPULATION
   └─ Year-specific population lookup

5. OUTPUT
   └─ 81 provinces × 75+ features
```

---

## 🔍 **Feature Groups Explained**

| Group | Features | Type | Data Source |
|-------|----------|------|-------------|
| **Group1** | 26 | Distance/proximity | Storm track + Province centroids |
| **Group2** | 20 | Weather exposure | Open-Meteo daily weather |
| **Group3** | 7 | Storm intensity | Storm track (wind/pressure) |
| **Group6** | 10 | Storm motion | Storm track (speed/direction) |
| **Group7** | 6 | Interactions | Distance × Weather |
| **Group8** | 6 | Multi-storm | Historical storm database |
| **Static** | 2 | Population | Census data |

**Total: 77 features**

---

## ⚠️ **Important Notes**

### **Group8 (Multi-Storm) is STATIC Context**

**Question:** *"Does Group8 require input of recent storm data?"*

**Answer:** **YES**, but it's **STATIC** relative to the current storm.

- Group8 features look at storms that **already happened** (past 30/60/90 days)
- These are **historical context** features
- They are **computed at prediction time** but use **past data only**

**Example:**
```
Current storm: 2024 Kristine (starts Oct 20, 2024)

Group8 computes:
  - storms_past_30_days: Count storms Sept 20 - Oct 19, 2024
  - storms_past_60_days: Count storms Aug 20 - Oct 19, 2024
  - concurrent_storms: Other storms active on Oct 20, 2024

All use ONLY storms that started/ended BEFORE Kristine.
```

**No data leakage** - we're not looking into the future!

---

## 🎯 **For Deployment**

When deploying for **real-time predictions** on a new storm:

```python
from pipeline.unified_pipeline import StormFeaturePipeline

# Initialize pipeline
pipeline = StormFeaturePipeline()

# Process new storm
features = pipeline.process_storm(
    year=2025,
    storm_name="NewStorm",
    include_group8=True  # Include recent storm context
)

# Load trained models
import joblib
clf = joblib.load("artifacts/stage1_classifier.joblib")
reg = joblib.load("artifacts/stage2_regressor.joblib")

# Predict
impact_proba = clf.predict_proba(features[feature_cols])[:, 1]
affected_pred = reg.predict(features[feature_cols])

# Get top-risk provinces
features['impact_probability'] = impact_proba
features['predicted_affected'] = np.expm1(affected_pred)
top_10 = features.nlargest(10, 'impact_probability')
```

---

## 📊 **Example Output**

```csv
Year,Storm,Province,group1__min_distance_km,group2__max_wind_gust_kmh,...,Population
2024,Kristine,Albay,45.2,120.5,...,1379398
2024,Kristine,Batangas,180.3,65.2,...,2994795
2024,Kristine,Camarines Sur,30.1,145.8,...,2063314
...
```

---

## 🛠️ **Troubleshooting**

### **Error: "Storm not found"**
- Check spelling of storm name (case-sensitive)
- Verify year is correct
- Ensure storm exists in `Storm_data/ph_storm_data.csv`

### **Error: "Weather file not found"**
- Ensure file exists: `Weather_location_data/{year}/{year}_{storm}.csv`
- Check filename matches Philippine name exactly

### **Missing Group8 features**
- Use `--no-group8` flag if historical data unavailable
- Group8 requires past 90 days of storm data

---

## 🔗 **Integration with Training**

After extracting features:

```bash
# 1. Extract features for all historical storms (2010-2024)
python scripts/batch_extract_all_storms.py

# 2. Merge with impact labels
python training/data_loader.py

# 3. Train models
python -m training.train
```

---

**Questions?** See `TRAINING_GUIDE.md` and `DEPLOYMENT_READINESS.md`

