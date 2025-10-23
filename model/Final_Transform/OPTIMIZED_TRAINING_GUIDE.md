# 🚀 Optimized Model Training Guide

## 📋 **Overview**

This guide covers **advanced model training** with hyperparameter optimization to improve model performance beyond the baseline.

### **What's New?**

✅ **Randomized Search CV** - Automated hyperparameter tuning  
✅ **Cross-Validation** - Better generalization (3-5 folds)  
✅ **Extended Search Space** - 9 hyperparameters per model  
✅ **Feature Importance Analysis** - Understand what drives predictions  
✅ **Comprehensive Metrics** - More evaluation metrics (MAPE, R², etc.)  
✅ **Better Regularization** - L1/L2 penalties, min_child_weight, gamma  

---

## 🎯 **Why Optimize?**

### **Current Model Issues**

1. **Classifier "Too Perfect"** (99.8% AUC-PR)
   - Possible overfitting on training data
   - May not generalize to truly novel storms
   - Need better regularization

2. **Regressor Moderate Performance** (RMSE 2.63 for houses, 0.58 for persons)
   - High variance in predictions
   - Could benefit from better hyperparameters
   - May need different tree depth/learning rate

3. **No Systematic Tuning**
   - Current params chosen manually
   - Not optimized for this specific dataset
   - Missing optimal configuration

---

## 🔬 **Hyperparameter Search Space**

### **Classifier Parameters**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_estimators` | [300, 500, 700, 1000] | Number of boosting rounds |
| `max_depth` | [4, 6, 8, 10] | Maximum tree depth |
| `learning_rate` | [0.01, 0.03, 0.05, 0.1] | Step size shrinkage |
| `subsample` | [0.7, 0.8, 0.9] | Row sampling ratio |
| `colsample_bytree` | [0.7, 0.8, 0.9, 1.0] | Column sampling ratio |
| `reg_lambda` | [0.1, 1.0, 5.0, 10.0] | L2 regularization |
| `reg_alpha` | [0.0, 0.1, 1.0, 5.0] | L1 regularization |
| `min_child_weight` | [1, 3, 5] | Minimum sum of instance weight |
| `gamma` | [0, 0.1, 0.3, 0.5] | Minimum loss reduction |

**Total Combinations:** ~100,000 possible configurations  
**Search Method:** Randomized (50-100 iterations with cross-validation)

### **Regressor Parameters**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_estimators` | [400, 600, 800, 1000] | Number of boosting rounds |
| `max_depth` | [6, 8, 10, 12] | Maximum tree depth |
| `learning_rate` | [0.01, 0.03, 0.05, 0.1] | Step size shrinkage |
| `subsample` | [0.7, 0.8, 0.9] | Row sampling ratio |
| `colsample_bytree` | [0.7, 0.8, 0.9, 1.0] | Column sampling ratio |
| `reg_lambda` | [0.1, 1.0, 5.0, 10.0] | L2 regularization |
| `reg_alpha` | [0.0, 0.1, 1.0, 5.0] | L1 regularization |
| `min_child_weight` | [1, 3, 5] | Minimum sum of instance weight |
| `gamma` | [0, 0.1, 0.3, 0.5] | Minimum loss reduction |

---

## 🚀 **Quick Start**

### **Option 1: Fast Tuning (Recommended for Testing)**

```bash
# Persons model (30 iterations, 3-fold CV, ~15-20 minutes)
python -m training.train_optimized \
  --split_mode stratified_storm \
  --n_iter 30 \
  --cv_folds 3 \
  --out_dir artifacts_optimized

# Houses model (30 iterations, 3-fold CV, ~15-20 minutes)
python -m training.train_houses_optimized \
  --split_mode stratified_storm \
  --n_iter 30 \
  --cv_folds 3 \
  --out_dir artifacts_houses_optimized
```

### **Option 2: Thorough Tuning (Recommended for Production)**

```bash
# Persons model (100 iterations, 5-fold CV, ~60-90 minutes)
python -m training.train_optimized \
  --split_mode stratified_storm \
  --n_iter 100 \
  --cv_folds 5 \
  --out_dir artifacts_optimized

# Houses model (100 iterations, 5-fold CV, ~60-90 minutes)
python -m training.train_houses_optimized \
  --split_mode stratified_storm \
  --n_iter 100 \
  --cv_folds 5 \
  --out_dir artifacts_houses_optimized
```

### **Option 3: Extreme Tuning (For Best Results)**

```bash
# Persons model (200 iterations, 5-fold CV, ~2-3 hours)
python -m training.train_optimized \
  --split_mode stratified_storm \
  --n_iter 200 \
  --cv_folds 5 \
  --out_dir artifacts_optimized

# Houses model (200 iterations, 5-fold CV, ~2-3 hours)
python -m training.train_houses_optimized \
  --split_mode stratified_storm \
  --n_iter 200 \
  --cv_folds 5 \
  --out_dir artifacts_houses_optimized
```

---

## 📊 **What to Expect**

### **Training Output**

```
======================================================================
OPTIMIZED MODEL TRAINING
======================================================================
Output directory: artifacts_optimized
RandomizedSearch iterations: 50
Cross-validation folds: 3
Random state: 42

Loading dataset...
✓ Dataset: 23653 samples, 80 features

Splitting data (mode: stratified_storm)...
✓ Data split complete

======================================================================
CLASSIFIER TRAINING WITH HYPERPARAMETER TUNING
======================================================================
Class imbalance ratio: 4.23
Positive samples: 4521 / 23653 (19.1%)

Starting RandomizedSearchCV:
  - Iterations: 50
  - CV Folds: 3
  - Metric: AUC-PR (average precision)
  - Search space: 100000+ combinations

Training...
Fitting 3 folds for each of 50 candidates, totalling 150 fits
[CV] END ...model__n_estimators=700, model__max_depth=6, ... (1 min 23s)
[CV] END ...model__n_estimators=500, model__max_depth=8, ... (1 min 45s)
...

✓ Best CV Score (AUC-PR): 0.9812
✓ Best Parameters:
    model__n_estimators: 700
    model__max_depth: 6
    model__learning_rate: 0.03
    model__subsample: 0.8
    model__colsample_bytree: 0.9
    model__reg_lambda: 5.0
    model__reg_alpha: 0.1
    model__min_child_weight: 3
    model__gamma: 0.1

======================================================================
CLASSIFIER EVALUATION
======================================================================

Validation Metrics:
  auc_pr              : 0.9845
  best_threshold      : 0.4532
  precision           : 0.9702
  recall              : 0.9843
  f1_score            : 0.9772
  accuracy            : 0.9756

Test Metrics:
  auc_pr              : 0.9821
  best_threshold      : 0.4532
  precision           : 0.9654
  recall              : 0.9876
  f1_score            : 0.9764
  accuracy            : 0.9723

📊 Top 20 Most Important Features:
======================================================================
  hist_max_affected                        0.1234
  min_distance_km                          0.0987
  avg_wind_speed_kmh                       0.0856
  max_precipitation_mm                     0.0743
  ...

✓ Classifier saved to: artifacts_optimized/stage1_classifier.joblib

======================================================================
REGRESSOR TRAINING WITH HYPERPARAMETER TUNING
======================================================================
...
```

### **Output Files**

```
artifacts_optimized/
├── stage1_classifier.joblib          ← Optimized classifier
├── stage2_regressor.joblib           ← Optimized regressor
├── feature_columns.json              ← Feature list
├── clf_metrics_val.json              ← Classifier validation metrics
├── clf_metrics_test.json             ← Classifier test metrics
├── clf_tuning_results.json           ← Hyperparameter search results
├── reg_metrics_val.json              ← Regressor validation metrics
├── reg_metrics_test.json             ← Regressor test metrics
├── reg_tuning_results.json           ← Hyperparameter search results
├── feature_importance_classifier.csv ← Feature rankings (classifier)
├── feature_importance_regressor.csv  ← Feature rankings (regressor)
└── split_summary.csv                 ← Data split statistics
```

---

## 🔍 **Analyzing Results**

### **1. Check Tuning Results**

```bash
# View best hyperparameters found
cat artifacts_optimized/clf_tuning_results.json

# Expected improvements:
# - Better regularization (higher reg_lambda/reg_alpha)
# - Deeper trees for regressor (max_depth 10-12)
# - More conservative learning rate (0.01-0.03)
# - Better sampling ratios
```

### **2. Compare with Baseline**

```python
import json

# Baseline
with open("artifacts/clf_metrics_test.json") as f:
    baseline = json.load(f)

# Optimized
with open("artifacts_optimized/clf_metrics_test.json") as f:
    optimized = json.load(f)

print(f"Baseline AUC-PR: {baseline['auc_pr']:.4f}")
print(f"Optimized AUC-PR: {optimized['auc_pr']:.4f}")
print(f"Improvement: {(optimized['auc_pr'] - baseline['auc_pr'])*100:.2f}%")
```

### **3. Feature Importance Analysis**

```bash
# Top features for classifier
head -20 artifacts_optimized/feature_importance_classifier.csv

# Top features for regressor
head -20 artifacts_optimized/feature_importance_regressor.csv
```

**Expected Insights:**
- `hist_max_affected` / `hist_max_houses` likely most important
- `min_distance_km` strong predictor
- Weather features (`avg_wind_speed_kmh`, `max_precipitation_mm`) important
- Some features may have near-zero importance → candidates for removal

---

## 📈 **Expected Performance Improvements**

### **Classifier**

| Metric | Baseline | Expected Optimized | Improvement |
|--------|----------|-------------------|-------------|
| AUC-PR | 0.9834 | 0.9750-0.9850 | ± 0.5% |
| Precision | 0.9650 | 0.9700-0.9800 | +1-2% |
| Recall | 0.9876 | 0.9850-0.9900 | ± 0.5% |
| F1-Score | 0.9762 | 0.9780-0.9850 | +0.5-1% |

**Note:** Baseline already very high, improvements may be marginal.  
**Goal:** Better generalization, not just higher test scores.

### **Regressor**

| Metric | Baseline (Persons) | Expected Optimized | Improvement |
|--------|-------------------|-------------------|-------------|
| RMSE (log) | 0.5842 | 0.5200-0.5600 | -5-10% |
| MAE (log) | 0.3312 | 0.2900-0.3200 | -5-10% |
| R² | 0.7538 | 0.7800-0.8200 | +3-6% |

| Metric | Baseline (Houses) | Expected Optimized | Improvement |
|--------|------------------|-------------------|-------------|
| RMSE (log) | 2.6287 | 2.3000-2.5000 | -5-12% |
| MAE (log) | 0.7125 | 0.6200-0.6800 | -5-13% |
| R² | 0.6502 | 0.6800-0.7300 | +4-8% |

**Goal:** Significant improvements expected for regressors!

---

## ⚙️ **Advanced Options**

### **Skip Classifier (Only Tune Regressor)**

```bash
# If classifier already good, focus on regressor
python -m training.train_optimized \
  --skip_classifier \
  --n_iter 100 \
  --cv_folds 5
```

### **Skip Regressor (Only Tune Classifier)**

```bash
python -m training.train_optimized \
  --skip_regressor \
  --n_iter 100 \
  --cv_folds 5
```

### **Custom Search Iterations**

```bash
# More thorough search (slower but potentially better)
python -m training.train_optimized \
  --n_iter 150 \
  --cv_folds 5

# Faster search (for prototyping)
python -m training.train_optimized \
  --n_iter 20 \
  --cv_folds 3
```

---

## 🎯 **Recommendations**

### **For Production Deployment**

1. **Run thorough tuning** (100+ iterations, 5-fold CV)
2. **Compare with baseline** on independent test set
3. **Check feature importance** - consider removing low-importance features
4. **Validate on recent storms** (2024-2025 data)
5. **Monitor cross-validation scores** - ensure no overfitting (train vs test gap)

### **Expected Tuning Time**

| Configuration | Persons Model | Houses Model | Total |
|--------------|---------------|--------------|-------|
| Fast (30 iter, 3-fold) | ~15 min | ~15 min | ~30 min |
| Medium (50 iter, 3-fold) | ~25 min | ~25 min | ~50 min |
| Thorough (100 iter, 5-fold) | ~90 min | ~90 min | ~3 hours |
| Extreme (200 iter, 5-fold) | ~180 min | ~180 min | ~6 hours |

**Hardware:** Based on modern CPU (8+ cores recommended)

---

## 🔧 **Troubleshooting**

### **Issue: "Memory Error"**

```bash
# Reduce CV folds or iterations
python -m training.train_optimized \
  --n_iter 30 \
  --cv_folds 2
```

### **Issue: "Taking Too Long"**

```bash
# Reduce iterations
python -m training.train_optimized \
  --n_iter 20 \
  --cv_folds 3
```

### **Issue: "Overfitting" (CV score >> test score)**

**Solution:** Use optimized model - it has better regularization!
- Check `reg_lambda` and `reg_alpha` in best params
- Consider increasing regularization in search space
- Reduce `max_depth`

---

## 📚 **Next Steps After Optimization**

1. **Replace baseline models** with optimized ones:
   ```bash
   # Backup old models
   mv artifacts artifacts_baseline
   mv artifacts_optimized artifacts
   
   mv artifacts_houses artifacts_houses_baseline
   mv artifacts_houses_optimized artifacts_houses
   ```

2. **Test with deployment pipeline**:
   ```bash
   python pipeline/deploy_both_models.py --storm-id wp3025
   ```

3. **Document improvements** in `Project_desc.md`

4. **Retrain if needed** with different hyperparameter ranges

---

## ✅ **Success Criteria**

**Optimization is successful if:**

✅ Regressor RMSE improves by >5%  
✅ Regressor R² increases by >3%  
✅ No significant drop in classifier metrics  
✅ Cross-validation scores stable (low std)  
✅ Feature importance makes sense  
✅ Models generalize to test set  

---

## 📊 **Comparison Table Template**

| Model | Baseline | Optimized | Improvement |
|-------|----------|-----------|-------------|
| **Persons Classifier** | | | |
| - AUC-PR | 0.9834 | ___ | ___ |
| - F1-Score | 0.9762 | ___ | ___ |
| **Persons Regressor** | | | |
| - RMSE | 0.5842 | ___ | ___ |
| - R² | 0.7538 | ___ | ___ |
| **Houses Classifier** | | | |
| - AUC-PR | 0.9983 | ___ | ___ |
| - F1-Score | 0.9928 | ___ | ___ |
| **Houses Regressor** | | | |
| - RMSE | 2.6287 | ___ | ___ |
| - R² | 0.6502 | ___ | ___ |

---

**Ready to optimize? Start with the fast tuning option and scale up!** 🚀


