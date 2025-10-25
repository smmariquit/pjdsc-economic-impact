"""
MINIMAL SKLEARN MODELS FOR ALL IMPACT TYPES
=====================================================
Trains 3 separate models:
1. People Affected
2. Houses Damaged  
3. Economic Cost (money)

Uses only 6 key features with RandomForest
Fast training (~5 minutes), easy to debug
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
import json

def load_and_prepare_data():
    """Load pre-engineered features and merge with ALL impact data"""
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
    pop_avg = pop_df.groupby('Province')['Population'].mean().reset_index()
    
    print("Loading impact data (people)...")
    people_df = pd.read_csv(script_dir / "Impact_data" / "people_affected_all_years.csv")
    
    print("Loading impact data (houses)...")
    houses_df = pd.read_csv(script_dir / "Impact_data" / "houses_all_years.csv")
    
    # Merge all features
    print("\nMerging features...")
    df = distance_df.merge(weather_df, on=['Year', 'Storm', 'Province'], how='inner')
    df = df.merge(intensity_df, on=['Year', 'Storm', 'Province'], how='inner')
    df = df.merge(pop_avg[['Province', 'Population']], on='Province', how='left')
    
    # Merge impact data - people
    df = df.merge(
        people_df[['Year', 'Storm', 'Province', 'Affected']].rename(columns={'Affected': 'People_Affected'}),
        on=['Year', 'Storm', 'Province'],
        how='left'
    )
    
    # Merge impact data - houses
    df = df.merge(
        houses_df[['Year', 'Storm', 'Province', 'Total Houses']].rename(columns={'Total Houses': 'Houses_Damaged'}),
        on=['Year', 'Storm', 'Province'],
        how='left'
    )
    
    # Fill missing
    df['People_Affected'] = df['People_Affected'].fillna(0)
    df['Houses_Damaged'] = df['Houses_Damaged'].fillna(0)
    df['Population'] = df['Population'].fillna(df['Population'].median())
    
    # Calculate economic cost estimate
    # Assumptions: $5000 per person affected, $20000 per house damaged
    df['Economic_Cost_USD'] = (df['People_Affected'] * 5000 + df['Houses_Damaged'] * 20000)
    
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Records with people impact: {(df['People_Affected'] > 0).sum()} ({100*(df['People_Affected'] > 0).mean():.1f}%)")
    print(f"✓ Records with house damage: {(df['Houses_Damaged'] > 0).sum()} ({100*(df['Houses_Damaged'] > 0).mean():.1f}%)")
    print(f"✓ Records with economic cost: {(df['Economic_Cost_USD'] > 0).sum()} ({100*(df['Economic_Cost_USD'] > 0).mean():.1f}%)")
    
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
        raise ValueError("Required features not found in data!")
    
    print("Selected features:")
    for i, feat in enumerate(available, 1):
        print(f"  {i}. {feat}")
    
    # Create feature matrix
    X = df[available].copy()
    X = X.fillna(X.median())
    
    # Target variables for each impact type
    targets = {
        'people': {
            'impact': (df['People_Affected'] > 0).astype(int),
            'count': df['People_Affected'].copy()
        },
        'houses': {
            'impact': (df['Houses_Damaged'] > 0).astype(int),
            'count': df['Houses_Damaged'].copy()
        },
        'cost': {
            'impact': (df['Economic_Cost_USD'] > 0).astype(int),
            'count': df['Economic_Cost_USD'].copy()
        }
    }
    
    print(f"\n✓ Feature matrix: {X.shape}")
    for name, data in targets.items():
        impact_rate = data['impact'].mean()
        avg_when_nonzero = data['count'][data['count'] > 0].mean()
        print(f"✓ {name.upper()}: {100*impact_rate:.1f}% impact rate, avg when >0: {avg_when_nonzero:.0f}")
    
    return X, targets, available

def split_by_year(df, X, targets):
    """Simple temporal split: train on 2010-2020, test on 2021-2024"""
    
    print("\n" + "="*70)
    print("STEP 3: TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*70)
    
    train_mask = df['Year'] <= 2020
    test_mask = df['Year'] > 2020
    
    splits = {
        'X_train': X[train_mask],
        'X_test': X[test_mask]
    }
    
    for name, data in targets.items():
        splits[f'y_impact_{name}_train'] = data['impact'][train_mask]
        splits[f'y_impact_{name}_test'] = data['impact'][test_mask]
        splits[f'y_count_{name}_train'] = data['count'][train_mask]
        splits[f'y_count_{name}_test'] = data['count'][test_mask]
    
    print(f"Train set: {len(splits['X_train'])} records (2010-2020)")
    print(f"Test set:  {len(splits['X_test'])} records (2021-2024)")
    
    return splits

def train_model_pair(X_train, y_impact_train, y_count_train, name):
    """Train classifier + regressor pair for one impact type"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {name.upper()} MODELS")
    print(f"{'='*70}")
    
    # Stage 1: Binary classifier
    print(f"1. Training {name} impact classifier...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_impact_train)
    print("   ✓ Classifier trained")
    
    # Stage 2: Regressor
    print(f"2. Training {name} count regressor...")
    mask_with_impact = y_impact_train == 1
    X_train_impact = X_train[mask_with_impact]
    y_count_train_impact = y_count_train[mask_with_impact]
    
    # Use log transform
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

def evaluate_model_pair(clf, reg, X_train, X_test, y_impact_train, y_impact_test, y_count_train, y_count_test, name):
    """Evaluate classifier + regressor pair"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: {name.upper()}")
    print(f"{'='*70}")
    
    # Classifier
    print("\n1. CLASSIFIER PERFORMANCE")
    print("-" * 70)
    y_pred_test = clf.predict(X_test)
    print("Test Set:")
    print(classification_report(y_impact_test, y_pred_test, target_names=['No Impact', 'Impact']))
    
    prob_test = clf.predict_proba(X_test)[:, 1]
    print(f"Probability Distribution:")
    print(f"  Min: {prob_test.min():.3f}, Median: {np.median(prob_test):.3f}, Max: {prob_test.max():.3f}")
    
    # Regressor
    print("\n2. REGRESSOR PERFORMANCE")
    print("-" * 70)
    mask_test_impact = y_impact_test == 1
    if mask_test_impact.sum() > 0:
        X_test_impact = X_test[mask_test_impact]
        y_count_test_impact = y_count_test[mask_test_impact]
        
        pred_log = reg.predict(X_test_impact)
        pred_count = np.expm1(pred_log)
        
        mae = mean_absolute_error(y_count_test_impact, pred_count)
        r2 = r2_score(y_count_test_impact, pred_count)
        
        print(f"  MAE:  {mae:.0f}")
        print(f"  R²:   {r2:.3f}")
        print(f"  Mean actual:     {y_count_test_impact.mean():.0f}")
        print(f"  Mean predicted:  {pred_count.mean():.0f}")

def save_all_models(models, feature_names):
    """Save all models and metadata"""
    
    print("\n" + "="*70)
    print("STEP: SAVING ALL MODELS")
    print("="*70)
    
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "artifacts_minimal"
    output_dir.mkdir(exist_ok=True)
    
    # Save each model pair
    for name, (clf, reg) in models.items():
        joblib.dump(clf, output_dir / f"classifier_{name}.joblib")
        joblib.dump(reg, output_dir / f"regressor_{name}.joblib")
        print(f"✓ Saved {name} models")
    
    # Save feature list
    with open(output_dir / "features_minimal.json", 'w') as f:
        json.dump({'features': list(feature_names)}, f, indent=2)
    
    print(f"\n✓ All models saved to: {output_dir}")
    print(f"  - classifier_people.joblib / regressor_people.joblib")
    print(f"  - classifier_houses.joblib / regressor_houses.joblib")
    print(f"  - classifier_cost.joblib / regressor_cost.joblib")
    print(f"  - features_minimal.json")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("MINIMAL SKLEARN MODELS - ALL IMPACT TYPES")
    print("People, Houses, Economic Cost")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Select features
    X, targets, feature_names = select_minimal_features(df)
    
    # Split
    splits = split_by_year(df, X, targets)
    
    # Train all three model pairs
    models = {}
    for impact_type in ['people', 'houses', 'cost']:
        clf, reg = train_model_pair(
            splits['X_train'],
            splits[f'y_impact_{impact_type}_train'],
            splits[f'y_count_{impact_type}_train'],
            impact_type
        )
        models[impact_type] = (clf, reg)
        
        # Evaluate
        evaluate_model_pair(
            clf, reg,
            splits['X_train'], splits['X_test'],
            splits[f'y_impact_{impact_type}_train'], splits[f'y_impact_{impact_type}_test'],
            splits[f'y_count_{impact_type}_train'], splits[f'y_count_{impact_type}_test'],
            impact_type
        )
    
    # Save
    save_all_models(models, feature_names)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check artifacts_minimal/ folder for all 6 model files")
    print("2. App will now predict people, houses, AND economic cost!")
    print("3. All predictions use the same 6 features")

if __name__ == "__main__":
    main()
