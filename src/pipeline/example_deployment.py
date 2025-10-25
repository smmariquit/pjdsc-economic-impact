"""
Example: Full deployment workflow for a new storm.

Shows how to:
  1. Extract features from raw data
  2. Load trained models
  3. Make predictions
  4. Rank provinces by risk
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from unified_pipeline import StormFeaturePipeline


def add_group_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add group prefixes to feature columns to match training format.
    
    Training features have format: group1__feature_name
    Pipeline features have format: feature_name
    
    This function maps pipeline columns to training columns.
    """
    # Define feature mappings based on deployment function returns
    group1_features = [
        'min_distance_km', 'mean_distance_km', 'max_distance_km', 'distance_range_km',
        'distance_std_km', 'hours_under_50km', 'hours_under_100km', 'hours_under_200km',
        'hours_under_300km', 'hours_under_500km', 'distance_at_current', 'distance_at_12hr',
        'distance_at_24hr', 'distance_at_48hr', 'distance_at_72hr', 'integrated_proximity',
        'weighted_exposure_hours', 'proximity_peak', 'approach_speed_kmh', 'departure_speed_kmh',
        'time_approaching_hours', 'time_departing_hours', 'bearing_at_closest_deg',
        'bearing_variability_deg', 'did_cross_province', 'approach_angle_deg'
    ]
    
    group2_features = [
        'max_wind_gust_kmh', 'max_wind_speed_kmh', 'total_precipitation_mm',
        'total_precipitation_hours', 'days_with_rain', 'consecutive_rain_days',
        'max_daily_precip_mm', 'max_hourly_precip_mm', 'mean_daily_precipitation_mm',
        'precip_variability', 'precipitation_concentration_index', 'days_with_heavy_rain',
        'days_with_very_heavy_rain', 'days_with_strong_wind', 'days_with_damaging_wind',
        'wind_gust_persistence_score', 'wind_rain_product', 'compound_hazard_score',
        'compound_hazard_days', 'rain_during_closest_approach'
    ]
    
    group3_features = [
        'max_wind_in_track_kt', 'min_pressure_in_track_hpa', 'wind_at_closest_approach_kt',
        'pressure_at_closest_hpa', 'wind_change_approaching_kt', 'intensification_rate_kt_per_day',
        'is_intensifying'
    ]
    
    group6_features = [
        'mean_forward_speed', 'max_forward_speed', 'min_forward_speed',
        'speed_at_closest_approach', 'mean_direction', 'direction_variability',
        'track_sinuosity', 'is_slow_moving', 'is_fast_moving', 'is_recurving'
    ]
    
    group7_features = [
        'min_distance_x_max_wind', 'proximity_intensity_product', 'intensity_per_km',
        'rainfall_distance_ratio', 'close_approach_rainfall', 'distant_rainfall'
    ]
    
    group8_features = [
        'has_concurrent_storm', 'concurrent_storms_count', 'nearest_concurrent_storm_distance',
        'concurrent_storms_combined_intensity', 'days_since_last_storm', 'storms_past_30_days'
    ]
    
    # Create rename mapping
    rename_map = {}
    for feat in group1_features:
        if feat in df.columns:
            rename_map[feat] = f'group1__{feat}'
    for feat in group2_features:
        if feat in df.columns:
            rename_map[feat] = f'group2__{feat}'
    for feat in group3_features:
        if feat in df.columns:
            rename_map[feat] = f'group3__{feat}'
    for feat in group6_features:
        if feat in df.columns:
            rename_map[feat] = f'group6__{feat}'
    for feat in group7_features:
        if feat in df.columns:
            rename_map[feat] = f'group7__{feat}'
    for feat in group8_features:
        if feat in df.columns:
            rename_map[feat] = f'group8__{feat}'
    
    # Rename columns
    df_renamed = df.rename(columns=rename_map)
    
    return df_renamed


def predict_storm_impact(year: int, storm_name: str, artifacts_dir: str = "artifacts"):
    """
    End-to-end prediction for a new storm.
    
    Args:
        year: Storm year
        storm_name: Philippine storm name
        artifacts_dir: Directory with trained models
    
    Returns:
        DataFrame with predictions for all provinces
    """
    print(f"\n{'='*70}")
    print(f"DEPLOYMENT PREDICTION: {year} {storm_name}")
    print(f"{'='*70}\n")
    
    # Step 1: Extract features
    print("1. Extracting features...")
    pipeline = StormFeaturePipeline()
    features = pipeline.process_storm(year, storm_name, include_group8=True, verbose=False)
    
    # Add group prefixes to match training format
    features = add_group_prefixes(features)
    print(f"   ✓ Features extracted: {features.shape}")
    
    # Step 2: Load models and feature columns
    print("\n2. Loading trained models...")
    artifacts_path = Path(artifacts_dir)
    
    clf = joblib.load(artifacts_path / "stage1_classifier.joblib")
    reg = joblib.load(artifacts_path / "stage2_regressor.joblib")
    
    import json
    with open(artifacts_path / "feature_columns.json") as f:
        feature_data = json.load(f)
        # Handle both dict format and list format
        if isinstance(feature_data, dict):
            feature_cols = feature_data['feature_columns']
        else:
            feature_cols = feature_data
    
    print(f"   ✓ Classifier loaded")
    print(f"   ✓ Regressor loaded")
    print(f"   ✓ Feature list: {len(feature_cols)} features")
    
    # Step 3: Prepare features (align with training)
    print("\n3. Preparing features...")
    
    # Add vulnerability features if missing (for new storms, default to 0)
    if 'hist_storms' not in features.columns:
        features['hist_storms'] = 0.0
    if 'hist_avg_affected' not in features.columns:
        features['hist_avg_affected'] = 0.0
    if 'hist_max_affected' not in features.columns:
        features['hist_max_affected'] = 0.0
    
    # Select only the features needed for prediction
    X = features[feature_cols].copy()
    
    # Handle any remaining missing features
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0  # Default value
            print(f"   ⚠ Missing feature '{col}' - using default value 0.0")
    
    # Reorder to match training
    X = X[feature_cols]
    
    print(f"   ✓ Feature matrix: {X.shape}")
    print(f"   ✓ Missing values: {X.isnull().sum().sum()}")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    
    # Stage 1: Impact probability
    impact_proba = clf.predict_proba(X)[:, 1]
    
    # Stage 2: Impact magnitude (only for high-probability cases)
    log_affected = reg.predict(X)
    affected_persons = np.expm1(log_affected)  # Convert from log scale
    
    # Combine: zero out low-probability predictions
    final_affected = np.where(impact_proba > 0.1, affected_persons, 0)
    
    print(f"   ✓ Predicted {(impact_proba > 0.5).sum()} provinces with impact")
    
    # Step 5: Create results DataFrame
    results = features[['Year', 'Storm', 'Province']].copy()
    results['impact_probability'] = impact_proba
    results['predicted_affected_raw'] = affected_persons
    results['predicted_affected_final'] = final_affected
    results['risk_score'] = impact_proba * np.log1p(affected_persons)
    
    # Add some key features for context
    if 'min_distance_km' in features.columns:
        results['min_distance_km'] = features['min_distance_km']
    if 'max_wind_gust_kmh' in features.columns:
        results['max_wind_gust_kmh'] = features['max_wind_gust_kmh']
    if 'Population' in features.columns:
        results['population'] = features['Population']
    
    # Sort by risk
    results = results.sort_values('risk_score', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}\n")
    
    return results


def print_top_risk_provinces(results: pd.DataFrame, top_n: int = 10):
    """Print top N provinces at risk."""
    print(f"\n🚨 TOP {top_n} PROVINCES AT RISK\n")
    print(f"{'Rank':<6} {'Province':<20} {'Impact %':<10} {'Predicted':<12} {'Distance':<12}")
    print("=" * 70)
    
    for i, row in results.head(top_n).iterrows():
        print(
            f"{i+1:<6} "
            f"{row['Province']:<20} "
            f"{row['impact_probability']*100:>6.1f}%    "
            f"{row['predicted_affected_final']:>10,.0f}  "
            f"{row.get('min_distance_km', 0):>10.1f} km"
        )
    
    print()


def export_alert_json(results: pd.DataFrame, output_file: str, threshold: float = 0.3):
    """Export high-risk provinces to JSON for alert system."""
    import json
    
    high_risk = results[results['impact_probability'] >= threshold].copy()
    
    alerts = []
    for _, row in high_risk.iterrows():
        alerts.append({
            'province': row['Province'],
            'impact_probability': float(row['impact_probability']),
            'predicted_affected': int(row['predicted_affected_final']),
            'risk_score': float(row['risk_score']),
            'alert_level': 'HIGH' if row['impact_probability'] > 0.7 else 'MEDIUM'
        })
    
    output = {
        'storm': f"{results['Year'].iloc[0]} {results['Storm'].iloc[0]}",
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_provinces_at_risk': len(alerts),
        'alerts': alerts
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Alert JSON exported: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict impact for a new storm")
    parser.add_argument("--year", type=int, required=True, help="Storm year")
    parser.add_argument("--storm", type=str, required=True, help="Storm name")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Models directory")
    parser.add_argument("--top", type=int, default=10, help="Number of top provinces to show")
    parser.add_argument("--export-csv", type=str, help="Export full results to CSV")
    parser.add_argument("--export-json", type=str, help="Export alerts to JSON")
    
    args = parser.parse_args()
    
    # Run prediction
    results = predict_storm_impact(args.year, args.storm, args.artifacts)
    
    # Display top risk provinces
    print_top_risk_provinces(results, top_n=args.top)
    
    # Export if requested
    if args.export_csv:
        results.to_csv(args.export_csv, index=False)
        print(f"✅ Full results exported: {args.export_csv}")
    
    if args.export_json:
        export_alert_json(results, args.export_json)

