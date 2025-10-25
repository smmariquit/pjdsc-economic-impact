# 📊 Complete Data Flow: Raw Input → Model Predictions

## 🗂️ **Data Architecture**

```
Final_Transform/
├── RAW INPUT DATA
│   ├── Storm_data/
│   │   └── ph_storm_data.csv           [Storm tracks: LAT/LON/time/wind]
│   ├── Weather_location_data/
│   │   └── {year}/{year}_{storm}.csv   [Daily weather per province]
│   ├── Population_data/
│   │   └── population_density_all_years.csv  [Census data]
│   └── Location_data/
│       └── locations_latlng.csv        [Province centroids]
│
├── FEATURE ENGINEERING (Process: Raw → Features)
│   ├── Feature_Engineering/
│   │   ├── group1/  [Distance features]
│   │   ├── group2/  [Weather features]
│   │   ├── group3/  [Intensity features]
│   │   ├── group6/  [Motion features]
│   │   ├── group7/  [Interaction features]
│   │   └── group8/  [Multi-storm features]
│   │
│   └── pipeline/
│       ├── unified_pipeline.py         [🆕 Single storm processor]
│       └── README.md
│
├── FEATURE DATA (Output from engineering)
│   └── Feature_Engineering_Data/
│       ├── group1/distance_features_group1.csv
│       ├── group2/weather_features_group2.csv
│       ├── group3/intensity_features_group3.csv
│       ├── group6/motion_features_group6.csv
│       ├── group7/interaction_features_group7.csv
│       └── group8/multistorm_features_group8.csv
│
├── LABELS (Ground truth)
│   └── Impact_data/
│       └── people_affected_all_years.csv  [Actual impacts]
│
├── TRAINING (Process: Features + Labels → Models)
│   └── training/
│       ├── data_loader.py              [Merge features + labels]
│       ├── split_utils.py              [Train/val/test split]
│       └── train.py                    [🆕 Two-stage training]
│
└── MODELS (Trained artifacts)
    └── artifacts/
        ├── stage1_classifier.joblib    [Impact yes/no]
        ├── stage2_regressor.joblib     [Magnitude prediction]
        └── feature_columns.json        [Feature list]
```

---

## 🌊 **Data Flow for Single Storm**

### **Scenario: New Storm "Kristine" in 2024**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: RAW DATA COLLECTION                                    │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│ Storm Track     │  │ Weather Data │  │ Static Data     │
│ (JTWC/IBTrACS)  │  │ (Open-Meteo) │  │ (Census/Coords) │
│                 │  │              │  │                 │
│ 2024_track.csv  │  │ 2024_Krist.. │  │ population.csv  │
│ LAT, LON, time  │  │ wind, rain   │  │ locations.csv   │
│ wind, pressure  │  │ per province │  │ year: 2024      │
└─────────────────┘  └──────────────┘  └─────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: FEATURE ENGINEERING                                     │
│ python pipeline/unified_pipeline.py --year 2024 --storm Kristine│
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ GROUP 1     │     │ GROUP 2     │ ... │ GROUP 8     │
│ Distance    │     │ Weather     │     │ Multi-storm │
│ 26 features │     │ 20 features │     │ 6 features  │
└─────────────┘     └─────────────┘     └─────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │ Merge on [Year, Storm, Province]
                             ▼
                   ┌─────────────────┐
                   │ FEATURE MATRIX  │
                   │ 81 provinces ×  │
                   │ 75+ features    │
                   └─────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: MODEL PREDICTION                                        │
│ python predict.py --features features_kristine.csv              │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌───────────────┐   ┌──────────────────┐
│ Stage 1      │    │ Stage 2       │   │ Final Output     │
│ Classifier   │ → │ Regressor     │ → │ Top 10 Provinces │
│              │    │               │   │ with predictions │
│ P(impact)    │    │ log(affected) │   │                  │
└──────────────┘    └───────────────┘   └──────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │ ALERT SYSTEM    │
                   │ - Albay: 85%    │
                   │ - Camarines: 78%│
                   │ - Quezon: 65%   │
                   └─────────────────┘
```

---

## 🔄 **Training Data Flow (Historical)**

```
┌─────────────────────────────────────────────────────────────────┐
│ HISTORICAL DATA (2010-2024)                                     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ Feature Engineering      │
              │ (batch process all years)│
              └──────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌────────────────┐   ┌──────────────┐
│ Engineered   │    │ Impact Labels  │   │ Population   │
│ Features     │    │ (ground truth) │   │ Historical   │
│ 285 storms   │    │ 140 storms     │   │ Vulnerability│
└──────────────┘    └────────────────┘   └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ data_loader.py           │
              │ - Merge all sources      │
              │ - Add vulnerability      │
              │ - Create labels          │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ split_utils.py           │
              │ - Stratified storm split │
              │ - 80/10/10 train/val/test│
              └──────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌────────────────┐   ┌──────────────┐
│ Train        │    │ Validation     │   │ Test         │
│ 18,756 rows  │    │ 2,241 rows     │   │ 2,656 rows   │
│ 226 storms   │    │ 27 storms      │   │ 32 storms    │
└──────────────┘    └────────────────┘   └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ train.py                 │
              │ - Train classifier       │
              │ - Train regressor        │
              │ - Save models            │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ artifacts/               │
              │ - stage1_classifier      │
              │ - stage2_regressor       │
              │ - feature_columns.json   │
              └──────────────────────────┘
```

---

## 🎯 **Key Concepts**

### **1. Raw Data → Features (Pipeline)**

**Purpose:** Transform messy raw data into clean ML features

**Input:**
- Storm track CSV (messy, sparse, 3-6hr intervals)
- Weather CSV (daily per province)
- Static CSVs (population, coordinates)

**Output:**
- One row per storm-province pair
- 75+ engineered features
- Ready for model input

**Tool:** `pipeline/unified_pipeline.py`

---

### **2. Features + Labels → Models (Training)**

**Purpose:** Learn patterns from historical storms

**Input:**
- Engineered features (from pipeline)
- Impact labels (actual damage reports)
- Population & vulnerability (added by data_loader)

**Output:**
- Trained classifier (impact yes/no)
- Trained regressor (magnitude)

**Tool:** `training/train.py`

---

### **3. New Storm → Predictions (Deployment)**

**Purpose:** Predict impact for incoming storm

**Input:**
- JTWC forecast (storm track)
- Open-Meteo forecast (weather)
- Historical storm database (for group8)

**Process:**
1. Run feature pipeline on new storm
2. Load trained models
3. Predict for all 81 provinces
4. Rank by risk, generate alerts

**Tool:** `pipeline/unified_pipeline.py` + `artifacts/models`

---

## ✅ **Is Group8 "Static"?**

**Answer: YES, relative to the current storm.**

Group8 features use **historical context**:

```python
Current storm: 2024-10-20 Kristine

Group8 computes:
  - storms_past_30_days: Count(storms between Sep 20 - Oct 19)
  - concurrent_storms: Other storms active on Oct 20
  - days_since_last_storm: Days since most recent storm ended

All use ONLY storms that:
  - Started BEFORE Kristine
  - Ended BEFORE Kristine starts
```

**At prediction time:** You query your storm database for context.

**No future leakage:** We never look at future storms or outcomes.

---

## 🚀 **Quick Commands**

```bash
# 1. Process single storm (raw → features)
python pipeline/unified_pipeline.py --year 2024 --storm Kristine --output features.csv

# 2. Train models (features → models)
python -m training.train --split_mode stratified_storm --out_dir artifacts

# 3. Check for data leakage
python check_leakage.py

# 4. Verify training results
python verify_training.py
```

---

**Complete pipeline is ready!** 🎉

