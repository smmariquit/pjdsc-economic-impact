"""
MINIMAL SKLEARN MODEL FOR ECONOMIC IMPACT PREDICTION
=====================================================
Uses only 6 key features with RandomForest
Fast training (~2 minutes), easy to debug
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
import json

def load_and_prepare_data():
    """Load pre-engineered features and merge with impact data"""
    script_dir = Path(__file__).parent.parent
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    # Load feature files (already computed at province-storm level)
    print("Loading distance features...")
    distance_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group1" / "distance_features_group1.csv")
    
    print("Loading weather features...")
    weather_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group2" / "weather_features_group2.csv")
    
    print("Loading intensity features...")
    intensity_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group3" / "intensity_features_group3.csv")
    
    print("Loading population data...")
    pop_df = pd.read_csv(script_dir / "Population_data" / "population_density_all_years.csv")
    
    # Get average population per province (across years)
    pop_avg = pop_df.groupby('Province')['Population'].mean().reset_index()
    
    print("Loading impact data...")
    impact_df = pd.read_csv(script_dir / "Impact_data" / "people_affected_all_years.csv")
    
    # Merge all features
    print("\nMerging features...")
    df = distance_df.merge(weather_df, on=['Year', 'Storm', 'Province'], how='inner')
    df = df.merge(intensity_df, on=['Year', 'Storm', 'Province'], how='inner')
    df = df.merge(pop_avg[['Province', 'Population']], on='Province', how='left')
    
    # Merge impact data
    df = df.merge(
        impact_df[['Year', 'Storm', 'Province', 'Affected']],
        on=['Year', 'Storm', 'Province'],
        how='left'
    )
    
    # Fill missing
    df['Affected'] = df['Affected'].fillna(0)
    df['Population'] = df['Population'].fillna(df['Population'].median())
    
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Records with impact: {(df['Affected'] > 0).sum()} ({100*(df['Affected'] > 0).mean():.1f}%)")
    
    return df

def select_minimal_features(df):
    """Select only 6 most important features"""
    
    print("\n" + "="*70)
    print("STEP 2: FEATURE SELECTION (6 KEY FEATURES)")
    print("="*70)
    
    # Only keep the most predictive features
    feature_cols = [
        'min_distance_km',           # How close the storm got
        'max_wind_gust_kmh',          # Peak wind intensity
        'total_precipitation_mm',     # Total rainfall
        'max_wind_in_track_kt',       # Storm strength in track
        'hours_under_100km',          # Duration of exposure
        'Population',                 # Vulnerability
    ]
    
    # Check which features exist
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    
    if missing:
        print(f"⚠️  Missing features: {missing}")
        print("Available columns sample:", df.columns[:20].tolist())
        raise ValueError("Required features not found in data!")
    
    print("Selected features:")
    for i, feat in enumerate(available, 1):
        print(f"  {i}. {feat}")
    
    # Create feature matrix
    X = df[available].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Target variables
    y_impact = (df['Affected'] > 0).astype(int)  # Binary: yes/no impact
    y_count = df['Affected'].copy()               # Count: how many affected
    
    print(f"\n✓ Feature matrix: {X.shape}")
    print(f"✓ Impact rate: {100*y_impact.mean():.1f}%")
    print(f"✓ Avg affected (when >0): {y_count[y_count>0].mean():.0f} people")
    
    return X, y_impact, y_count, available

def split_by_year(df, X, y_impact, y_count):
    """Simple temporal split: train on 2010-2020, test on 2021-2024"""
    
    print("\n" + "="*70)
    print("STEP 3: TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*70)
    
    train_mask = df['Year'] <= 2020
    test_mask = df['Year'] > 2020
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_impact_train = y_impact[train_mask]
    y_impact_test = y_impact[test_mask]
    y_count_train = y_count[train_mask]
    y_count_test = y_count[test_mask]
    
    print(f"Train set: {len(X_train)} records (2010-2020)")
    print(f"  - Impact rate: {100*y_impact_train.mean():.1f}%")
    print(f"Test set:  {len(X_test)} records (2021-2024)")
    print(f"  - Impact rate: {100*y_impact_test.mean():.1f}%")
    
    return X_train, X_test, y_impact_train, y_impact_test, y_count_train, y_count_test

def train_models(X_train, y_impact_train, y_count_train):
    """Train simple RandomForest models"""
    
    print("\n" + "="*70)
    print("STEP 4: TRAINING MODELS")
    print("="*70)
    
    # Stage 1: Binary classifier (will there be impact?)
    print("\n1. Training impact classifier...")
    clf = RandomForestClassifier(
        n_estimators=50,        # Fewer trees = faster
        max_depth=8,            # Shallow = less overfit
        min_samples_split=20,   # Don't split small nodes
        min_samples_leaf=10,    # At least 10 samples per leaf
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_impact_train)
    print("   ✓ Classifier trained")
    
    # Stage 2: Regressor (how many affected?)
    # Only train on samples WITH impact
    print("\n2. Training impact regressor...")
    mask_with_impact = y_impact_train == 1
    X_train_impact = X_train[mask_with_impact]
    y_count_train_impact = y_count_train[mask_with_impact]
    
    # Use log transform for better distribution
    y_count_log = np.log1p(y_count_train_impact)
    
    reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train_impact, y_count_log)
    print("   ✓ Regressor trained")
    
    return clf, reg

def evaluate_models(clf, reg, X_train, X_test, y_impact_train, y_impact_test, y_count_train, y_count_test):
    """Evaluate model performance"""
    
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("="*70)
    
    # Classifier evaluation
    print("\n1. CLASSIFIER PERFORMANCE")
    print("-" * 70)
    
    # Training set
    y_pred_train = clf.predict(X_train)
    print("Training Set:")
    print(classification_report(y_impact_train, y_pred_train, target_names=['No Impact', 'Impact']))
    
    # Test set
    y_pred_test = clf.predict(X_test)
    print("\nTest Set:")
    print(classification_report(y_impact_test, y_pred_test, target_names=['No Impact', 'Impact']))
    
    # Show probability distribution
    prob_test = clf.predict_proba(X_test)[:, 1]
    print(f"\nProbability Distribution on Test Set:")
    print(f"  Min:    {prob_test.min():.3f}")
    print(f"  Q1:     {np.percentile(prob_test, 25):.3f}")
    print(f"  Median: {np.percentile(prob_test, 50):.3f}")
    print(f"  Q3:     {np.percentile(prob_test, 75):.3f}")
    print(f"  Max:    {prob_test.max():.3f}")
    
    # Regressor evaluation (only on samples WITH impact)
    print("\n2. REGRESSOR PERFORMANCE")
    print("-" * 70)
    
    mask_test_impact = y_impact_test == 1
    if mask_test_impact.sum() > 0:
        X_test_impact = X_test[mask_test_impact]
        y_count_test_impact = y_count_test[mask_test_impact]
        y_count_test_log = np.log1p(y_count_test_impact)
        
        # Predict
        pred_log = reg.predict(X_test_impact)
        pred_count = np.expm1(pred_log)
        
        mae = mean_absolute_error(y_count_test_impact, pred_count)
        r2 = r2_score(y_count_test_impact, pred_count)
        
        print(f"Test Set (only samples with impact):")
        print(f"  MAE:  {mae:.0f} people")
        print(f"  R²:   {r2:.3f}")
        print(f"  Mean actual:     {y_count_test_impact.mean():.0f}")
        print(f"  Mean predicted:  {pred_count.mean():.0f}")
    
    # Feature importance
    print("\n3. FEATURE IMPORTANCE")
    print("-" * 70)
    feature_names = X_train.columns
    importances = clf.feature_importances_
    
    for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:30s} {imp:.3f} {'█' * int(imp * 50)}")

def save_models(clf, reg, feature_names):
    """Save models and metadata"""
    
    print("\n" + "="*70)
    print("STEP 6: SAVING MODELS")
    print("="*70)
    
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "artifacts_minimal"
    output_dir.mkdir(exist_ok=True)
    
    # Save models
    joblib.dump(clf, output_dir / "classifier_minimal.joblib")
    joblib.dump(reg, output_dir / "regressor_minimal.joblib")
    
    # Save feature list
    with open(output_dir / "features_minimal.json", 'w') as f:
        json.dump({'features': list(feature_names)}, f, indent=2)
    
    print(f"✓ Models saved to: {output_dir}")
    print(f"  - classifier_minimal.joblib")
    print(f"  - regressor_minimal.joblib")
    print(f"  - features_minimal.json")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("MINIMAL SKLEARN MODEL TRAINING")
    print("Fast & Simple: 6 features, RandomForest, ~2 minutes")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Select features
    X, y_impact, y_count, feature_names = select_minimal_features(df)
    
    # Split
    X_train, X_test, y_impact_train, y_impact_test, y_count_train, y_count_test = split_by_year(df, X, y_impact, y_count)
    
    # Train
    clf, reg = train_models(X_train, y_impact_train, y_count_train)
    
    # Evaluate
    evaluate_models(clf, reg, X_train, X_test, y_impact_train, y_impact_test, y_count_train, y_count_test)
    
    # Save
    save_models(clf, reg, feature_names)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check artifacts_minimal/ folder for saved models")
    print("2. Update app_ml.py to load these minimal models")
    print("3. Test predictions - they should be MUCH more varied now!")

if __name__ == "__main__":
    main()
