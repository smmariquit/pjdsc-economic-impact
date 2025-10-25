"""
Simple Fast Training Script - Economic Impact from Storm Forecast

Uses only the MOST IMPORTANT features for quick, reliable predictions:
- Distance to province (closest approach)
- Storm intensity (wind speed, pressure)
- Weather forecast (rainfall, wind)
- Population (vulnerability)

Trains in ~5 minutes with good accuracy!
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, average_precision_score
from sklearn.model_selection import train_test_split

def load_data():
    """Load training data from existing dataset."""
    print("Loading data...")
    
    # Get base directory
    script_dir = Path(__file__).parent.parent  # Go up to Final_Transform/
    
    # Load pre-engineered feature files (already at province-storm level)
    distance_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group1" / "distance_features_group1.csv")
    weather_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group2" / "weather_features_group2.csv")
    intensity_df = pd.read_csv(script_dir / "Feature_Engineering_Data" / "group3" / "intensity_features_group3.csv")
    
    # Load population data
    pop_df = pd.read_csv(script_dir / "Population_data" / "provinces_with_population.csv")
    
    # Load impact data
    impact_file = script_dir / "Impact_data" / "people_affected_all_years.csv"
    impact_df = pd.read_csv(impact_file)
    
    # Merge features
    df = distance_df.merge(weather_df, on=['Year', 'Storm', 'Province'], how='inner')
    df = df.merge(intensity_df, on=['Year', 'Storm', 'Province'], how='inner')
    
    # Add population data
    df = df.merge(pop_df[['Province', 'Population', 'PopulationDensity']], 
                  on='Province', how='left')
    
    # Merge impact data
    merged = df.merge(
        impact_df[['Year', 'Storm', 'Province', 'Affected']],
        on=['Year', 'Storm', 'Province'],
        how='left'
    )
    
    # Fill missing impacts with 0
    merged['Affected'] = merged['Affected'].fillna(0)
    merged['has_impact'] = (merged['Affected'] > 0).astype(int)
    
    print(f"✓ Loaded {len(merged)} province-storm pairs")
    print(f"✓ {merged['has_impact'].sum()} with impact ({100*merged['has_impact'].mean():.1f}%)")
    
    return merged


def select_simple_features(df):
    """Select only the MOST IMPORTANT features - easy to engineer from forecast."""
    
    # Key features that strongly predict impact:
    important_features = [
        # Distance (GROUP 1) - #1 predictor
        'min_distance_km',
        'hours_under_100km',
        'hours_under_200km',
        
        # Weather (GROUP 2) - Direct impact drivers
        'max_wind_gust_kmh',
        'total_precipitation_mm',
        'max_daily_precip_mm',
        
        # Intensity (GROUP 3) - Storm strength
        'max_wind_in_track_kt',
        'wind_at_closest_approach_kt',
        
        # Population - Vulnerability
        'Population',
        'PopulationDensity',
    ]
    
    # Check which features exist
    available = [f for f in important_features if f in df.columns]
    missing = [f for f in important_features if f not in df.columns]
    
    if missing:
        print(f"⚠️ Missing features (will use defaults): {missing}")
        for feat in missing:
            df[feat] = 0
    
    print(f"✓ Using {len(important_features)} key features")
    
    return df, important_features


def train_simple_model(X_train, y_train, X_test, y_test, feature_names, model_type='classifier'):
    """Train a simple RandomForest model (fast, interpretable)."""
    
    if model_type == 'classifier':
        print("\n🎯 Training Impact Classifier (Yes/No impact)...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_proba)
        
        print(f"✓ F1 Score: {f1:.3f}")
        print(f"✓ AUC-PR: {auc_pr:.3f}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 5 Most Important Features:")
        for _, row in importances.head(5).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:.3f}")
        
        return model, {'f1': f1, 'auc_pr': auc_pr}
    
    else:  # regressor
        print("\n💰 Training Impact Regressor (How many affected)...")
        
        # Log transform target
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_log)
        
        # Evaluate
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ MAE: {mae:.0f} persons")
        print(f"✓ R² Score: {r2:.3f}")
        
        return model, {'mae': mae, 'r2': r2}


def main():
    print("="*70)
    print("SIMPLE ECONOMIC IMPACT MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Select simple features
    df, features = select_simple_features(df)
    
    # Split data (temporal: train on 2010-2020, test on 2021-2024)
    train_df = df[df['Year'] <= 2020]
    test_df = df[df['Year'] > 2020]
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_df)} samples (2010-2020)")
    print(f"   Test:  {len(test_df)} samples (2021-2024)")
    
    # Prepare train/test sets
    X_train = train_df[features]
    X_test = test_df[features]
    y_train_clf = train_df['has_impact']
    y_test_clf = test_df['has_impact']
    
    # Stage 1: Classifier (will there be impact?)
    clf, clf_metrics = train_simple_model(
        X_train, y_train_clf, 
        X_test, y_test_clf, 
        features, 
        'classifier'
    )
    
    # Stage 2: Regressor (how many affected?) - only on impacted samples
    train_pos = train_df[train_df['has_impact'] == 1]
    test_pos = test_df[test_df['has_impact'] == 1]
    
    X_train_reg = train_pos[features]
    X_test_reg = test_pos[features]
    y_train_reg = train_pos['Affected']
    y_test_reg = test_pos['Affected']
    
    print(f"\nRegressor training samples: {len(train_pos)} (only impacted provinces)")
    
    reg, reg_metrics = train_simple_model(
        X_train_reg, y_train_reg,
        X_test_reg, y_test_reg,
        features,
        'regressor'
    )
    
    # Save models
    script_dir = Path(__file__).parent.parent  # Final_Transform/
    out_dir = script_dir / "artifacts_simple"
    out_dir.mkdir(exist_ok=True)
    
    joblib.dump(clf, out_dir / "classifier_simple.joblib")
    joblib.dump(reg, out_dir / "regressor_simple.joblib")
    
    # Save features
    with open(out_dir / "features_simple.json", 'w') as f:
        json.dump({'features': features}, f, indent=2)
    
    # Save metrics
    with open(out_dir / "metrics_simple.json", 'w') as f:
        json.dump({
            'classifier': clf_metrics,
            'regressor': reg_metrics
        }, f, indent=2)
    
    print(f"\n✅ Models saved to: {out_dir}/")
    print(f"   - classifier_simple.joblib")
    print(f"   - regressor_simple.joblib")
    print(f"   - features_simple.json")
    
    print("\n🎉 Training complete! Ready to deploy in Streamlit.")


if __name__ == "__main__":
    main()
