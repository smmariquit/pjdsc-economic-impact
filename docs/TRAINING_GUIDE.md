# 🎓 Complete Training Guide

TLDR: run

```bash
# Optimize persons model (90 min)
python -m training.train_optimized --n_iter 100 --cv_folds 5

# Optimize houses model (90 min)
python -m training.train_houses_optimized --n_iter 100 --cv_folds 5
```

## 📋 **Prerequisites**

Ensure you have these data files ready:

```
Final_Transform/
├── Feature_Engineering_Data/
│   ├── group1/distance_features_group1.csv
│   ├── group2/weather_features_group2.csv
│   ├── group3/intensity_features_group3.csv
│   ├── group6/motion_features_group6.csv
│   ├── group7/interaction_features_group7.csv
│   └── group8/multistorm_features_group8.csv
├── Population_data/
│   └── population_density_all_years.csv
└── Impact_data/
    ├── people_affected_all_years.csv
    └── houses_all_years.csv
```

---

## 🚀 **Step-by-Step Training**

### **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 2: Run Training (DUAL MODELS)**

```bash
# Train Persons Affected model
python -m training.train --split_mode stratified_storm --random_state 42 --out_dir artifacts

# Train Houses Damaged model
python -m training.train_houses --split_mode stratified_storm --random_state 42 --out_dir artifacts_houses
```

### **Step 3: Verify Results**

Check the output directories:
- `artifacts/` - Persons model
- `artifacts_houses/` - Houses model

Each contains:
- `stage1_classifier*.joblib` - Binary classifier
- `stage2_regressor*.joblib` - Magnitude regressor
- `clf_metrics_*.json` - Classification performance
- `reg_metrics_*.json` - Regression performance
- `split_summary_*.csv` - Data split statistics

---

## ⚙️ **Training Parameters Explained**

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--split_mode` | How to split data | `stratified_storm` | `stratified_storm`, `temporal` |
| `--random_state` | Random seed | `42` | Any integer |
| `--train_frac` | Training fraction | `0.8` | 0.0-1.0 |
| `--val_frac` | Validation fraction | `0.1` | 0.0-1.0 |
| `--test_frac` | Test fraction | `0.1` | 0.0-1.0 |
| `--out_dir` | Output directory | `artifacts` | Any path |

**For temporal split only:**
| Parameter | Description | Format |
|-----------|-------------|--------|
| `--train_range` | Training years | `YYYY-YYYY` (e.g., `2010-2018`) |
| `--val_range` | Validation years | `YYYY-YYYY` (e.g., `2019-2020`) |
| `--test_range` | Test years | `YYYY-YYYY` (e.g., `2021-2024`) |

---

## 🎯 **Training Pipeline Details**

### **What Happens During Training:**

1. **Load Features** (6 groups, 75+ features)
   - Distance features (group1): 26 features
   - Weather features (group2): 20 features
   - Intensity features (group3): 7 features
   - Motion features (group6): 10 features
   - Interaction features (group7): 6 features
   - Multi-storm features (group8): 6 features

2. **Add Contextual Data**
   - Population (yearly, per province)
   - Population density
   - Historical vulnerability (leakage-free):
     - `hist_storms` - # of impactful years in past
     - `hist_avg_affected` - average affected per past storm
     - `hist_max_affected` - worst historical impact

3. **Stratified Storm Split** (80/10/10)
   - Groups storms by severity (none/minor/moderate/major)
   - Randomly assigns storms to train/val/test within each severity bin
   - Ensures all splits have balanced mix of storm types
   - Entire storm-province sets stay together

4. **Stage 1: Binary Classification**
   - **Model**: XGBoost Classifier
   - **Target**: `has_impact` (0 or 1)
   - **Handling class imbalance**:
     - `scale_pos_weight` = ratio of negatives to positives
     - Optimize for AUC-PR (not accuracy)
   - **Hyperparameters**:
     ```python
     n_estimators=400
     max_depth=6
     learning_rate=0.05
     subsample=0.8
     colsample_bytree=0.8
     ```

5. **Stage 2: Regression (on positives only)**
   - **Model**: XGBoost Regressor
   - **Target**: `log1p(Affected)` - log-transformed affected persons
   - **Training data**: Only rows where `has_impact=1`
   - **Hyperparameters**:
     ```python
     n_estimators=600
     max_depth=8
     learning_rate=0.05
     subsample=0.8
     colsample_bytree=0.8
     ```

6. **Preprocessing Pipeline** (built into models)
   - Median imputation for missing values
   - Standard scaling (zero mean, unit variance)
   - Applied to both models automatically

7. **Save Artifacts**
   - Trained models (`.joblib` format)
   - Feature list (`.json`)
   - Metrics (`.json`)
   - Split summary (`.csv`)

---

## 📊 **Expected Performance**

Based on current data (with leakage-free vulnerability features):

| Metric | Target | Achieved |
|--------|--------|----------|
| **Classification AUC-PR** | >0.65 | ~1.00 ⚠️ |
| **Classification F1** | >0.55 | ~1.00 ⚠️ |
| **Regression RMSE (log)** | <3.5 | ~2.8 ✅ |

**⚠️ Note**: Near-perfect classification suggests possible remaining issues:
- May be overfitting to storm-specific patterns
- Splits might be too easy (severity stratification too strong)
- Could indicate subtle feature leakage (investigate group8)

---

## 🔧 **Troubleshooting**

### **Issue: "FileNotFoundError: No feature groups found"**
**Solution**: Check that `Feature_Engineering_Data/` contains the CSV files.

### **Issue: "KeyError: 'Storm'"**
**Solution**: Ensure all feature CSVs have columns: `Year`, `Storm`, `Province`.

### **Issue: "Near-perfect accuracy (>0.99)"**
**Diagnosis**: Possible data leakage or overfitting.
**Actions**:
1. Run `python check_leakage.py` to verify no leakage
2. Check group8 features (multi-storm) for temporal contamination
3. Try different `random_state` seeds to ensure robustness

### **Issue: "Very poor performance (<0.5 AUC-PR)"**
**Diagnosis**: Model not learning.
**Actions**:
1. Check class imbalance (`split_summary.csv`)
2. Verify feature scaling (should be automatic)
3. Increase `n_estimators` or `max_depth`

---

## 🎨 **Training Variations**

### **1. Train Multiple Models (Ensemble)**

```bash
# Train 5 models with different seeds
for seed in 42 123 456 789 1011; do
  python -m training.train --random_state $seed --out_dir artifacts_seed_$seed
done
```

### **2. Ablation Study (Feature Importance)**

Remove feature groups to test importance:

```bash
# Temporarily rename group to skip it
mv Feature_Engineering_Data/group8 Feature_Engineering_Data/group8_disabled
python -m training.train --out_dir artifacts_no_group8
mv Feature_Engineering_Data/group8_disabled Feature_Engineering_Data/group8
```

### **3. Hyperparameter Tuning**

Edit `training/train.py` and modify:

```python
def train_classifier(X, y, feature_cols):
    model = XGBClassifier(
        n_estimators=600,      # Increase trees
        max_depth=8,           # Deeper trees
        learning_rate=0.03,    # Slower learning
        # ... other params
    )
```

---

## 📈 **After Training**

### **Next Steps:**

1. **Evaluate on specific storms:**
   ```python
   import joblib
   import pandas as pd
   
   clf = joblib.load("artifacts/stage1_classifier.joblib")
   reg = joblib.load("artifacts/stage2_regressor.joblib")
   
   # Load test data for Typhoon Yolanda (2013)
   # ... make predictions
   ```

2. **Feature importance analysis:**
   ```python
   import joblib
   import json
   
   model = joblib.load("artifacts/stage1_classifier.joblib")
   with open("artifacts/feature_columns.json") as f:
       features = json.load(f)["feature_columns"]
   
   importances = model.named_steps["model"].feature_importances_
   for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:20]:
       print(f"{feat}: {imp:.4f}")
   ```

3. **Deploy for real-time predictions:**
   - See `DEPLOYMENT_READINESS.md`
   - Set up JTWC API ingestion
   - Configure Open-Meteo real-time weather

---

## 💾 **Output Files Reference**

| File | Description | Used For |
|------|-------------|----------|
| `stage1_classifier.joblib` | Binary classifier | Predict if province will have impact |
| `stage2_regressor.joblib` | Magnitude regressor | Predict # of affected persons (if impact predicted) |
| `feature_columns.json` | Feature list | Ensure correct feature order during inference |
| `split_summary.csv` | Dataset statistics | Verify split balance |
| `clf_metrics_*.json` | Classification metrics | Model evaluation |
| `reg_metrics_*.json` | Regression metrics | Model evaluation |

---

## 🎯 **Quick Commands Cheat Sheet**

```bash
# Full training (recommended)
python -m training.train

# With custom output directory
python -m training.train --out_dir my_models

# Different random seed
python -m training.train --random_state 123

# More training data (85% train, 10% val, 5% test)
python -m training.train --train_frac 0.85 --test_frac 0.05

# Verify results
python verify_training.py

# Check for data leakage
python check_leakage.py
```

---

**Happy Training! 🚀**

