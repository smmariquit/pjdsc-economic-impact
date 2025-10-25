"""
Example: Using the Storm Intensity Feature Engineering Pipeline for Deployment

This demonstrates how to use the modular functions for:
1. Training: Processing historical storm intensity data
2. Deployment: Real-time feature extraction from forecast/observation data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from group3_storm_intensity_features import (
    extract_intensity_features_for_deployment,
    compute_all_intensity_features,
    MISSING_VALUE
)


# ============================================================================
# EXAMPLE 1: Deployment Scenario - Real-time Intensity Feature Extraction
# ============================================================================

def example_realtime_intensity():
    """
    Simulate processing real-time storm intensity data for deployment.
    """
    print("=" * 60)
    print("EXAMPLE 1: Real-time Intensity Feature Extraction")
    print("=" * 60)
    
    # Simulated storm track data (e.g., from JTWC)
    track_data = pd.DataFrame({
        'TOKYO_WIND': [25, 35, 45, 55, 65, 70, 65, 55],
        'USA_WIND': ['', '', '', '', '', '', '', ''],  # Not always available
        'TOKYO_PRES': [1005, 1002, 998, 994, 990, 988, 992, 996],
        'PH_DAY': ['2024-11-15'] * 8,
        'PH_TIME': ['00:00:00', '03:00:00', '06:00:00', '09:00:00', 
                   '12:00:00', '15:00:00', '18:00:00', '21:00:00']
    })
    
    # Closest approach at index 5 (15:00:00)
    closest_approach_idx = 5
    
    # Extract features
    features = extract_intensity_features_for_deployment(
        track_data=track_data,
        closest_approach_idx=closest_approach_idx
    )
    
    print("\nExtracted Intensity Features:")
    print("-" * 60)
    for feature_name, value in sorted(features.items()):
        if value == MISSING_VALUE:
            print(f"  {feature_name:40s}: MISSING")
        else:
            print(f"  {feature_name:40s}: {value:>12.2f}")
    
    print("\n" + "=" * 60)
    return features


# ============================================================================
# EXAMPLE 2: Processing a Single Storm
# ============================================================================

def example_single_storm():
    """
    Process intensity features for a specific storm.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Single Storm Intensity Features")
    print("=" * 60)
    
    # Load storm track data
    storm_data_file = '../../Storm_data/ph_storm_data.csv'
    
    if not Path(storm_data_file).exists():
        print(f"Storm data file not found: {storm_data_file}")
        return None
    
    storm_data = pd.read_csv(storm_data_file)
    
    # Filter for a specific storm (e.g., 2021 Odette)
    storm_track = storm_data[
        (storm_data['SEASON'] == 2021) &
        (storm_data['PHNAME'] == 'Odette')
    ].copy()
    
    print(f"\nStorm: 2021 Odette")
    print(f"Track points: {len(storm_track)}")
    
    # Compute features (middle of track as closest approach)
    features = compute_all_intensity_features(
        storm_track,
        province='Cebu',  # Example province
        distance_features_df=None
    )
    
    print("\nIntensity Features:")
    print("-" * 60)
    for feature_name, value in sorted(features.items()):
        if value == MISSING_VALUE:
            print(f"  {feature_name:40s}: MISSING")
        else:
            print(f"  {feature_name:40s}: {value:>12.2f}")
    
    print("\n" + "=" * 60)
    return features


# ============================================================================
# EXAMPLE 3: Analyzing Data Completeness
# ============================================================================

def example_data_completeness():
    """
    Analyze data completeness for intensity features.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Data Completeness Analysis")
    print("=" * 60)
    
    features_file = '../../Feature_Engineering_Data/group3/intensity_features_group3.csv'
    
    if not Path(features_file).exists():
        print(f"Features file not found: {features_file}")
        print("Run the main script first to generate features.")
        return None
    
    # Load features
    features_df = pd.read_csv(features_file)
    
    print(f"\nDataset loaded:")
    print(f"  Total records: {len(features_df):,}")
    print(f"  Years: {features_df['Year'].min()} - {features_df['Year'].max()}")
    print(f"  Storms: {features_df['Storm'].nunique()}")
    print(f"  Provinces: {features_df['Province'].nunique()}")
    
    # Analyze completeness
    print("\nData Completeness by Feature:")
    print("-" * 60)
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Year', 'Storm', 'Province']]
    
    for col in feature_cols:
        total = len(features_df)
        missing = (features_df[col] == MISSING_VALUE).sum()
        complete_pct = ((total - missing) / total) * 100
        print(f"  {col:40s}: {complete_pct:>5.1f}% complete ({total-missing:,}/{total:,})")
    
    # Show statistics for complete data only
    print("\n\nFeature Statistics (complete data only):")
    print("-" * 60)
    
    for col in feature_cols:
        valid_data = features_df[features_df[col] != MISSING_VALUE][col]
        if len(valid_data) > 0:
            print(f"\n{col}:")
            print(f"  Count: {len(valid_data):,}")
            print(f"  Mean: {valid_data.mean():.2f}")
            print(f"  Std: {valid_data.std():.2f}")
            print(f"  Min: {valid_data.min():.2f}")
            print(f"  25%: {valid_data.quantile(0.25):.2f}")
            print(f"  50%: {valid_data.median():.2f}")
            print(f"  75%: {valid_data.quantile(0.75):.2f}")
            print(f"  Max: {valid_data.max():.2f}")
    
    print("\n" + "=" * 60)
    return features_df


# ============================================================================
# EXAMPLE 4: Combining with Other Feature Groups
# ============================================================================

def example_combine_all_features():
    """
    Demonstrate combining all three feature groups.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Combining All Feature Groups (1+2+3)")
    print("=" * 60)
    
    distance_file = '../../Feature_Engineering_Data/group1/distance_features_group1.csv'
    weather_file = '../../Feature_Engineering_Data/group2/weather_features_group2.csv'
    intensity_file = '../../Feature_Engineering_Data/group3/intensity_features_group3.csv'
    
    files_exist = all([
        Path(distance_file).exists(),
        Path(weather_file).exists(),
        Path(intensity_file).exists()
    ])
    
    if not files_exist:
        print("One or more feature files not found.")
        print("Run all feature engineering scripts first.")
        return None
    
    # Load all feature sets
    distance_df = pd.read_csv(distance_file)
    weather_df = pd.read_csv(weather_file)
    intensity_df = pd.read_csv(intensity_file)
    
    print(f"\nFeature Groups Loaded:")
    print(f"  GROUP 1 (Distance): {len(distance_df.columns) - 3} features")
    print(f"  GROUP 2 (Weather): {len(weather_df.columns) - 3} features")
    print(f"  GROUP 3 (Intensity): {len(intensity_df.columns) - 3} features")
    
    # Merge all on Year, Storm, Province
    combined_df = distance_df.merge(
        weather_df,
        on=['Year', 'Storm', 'Province'],
        how='inner'
    ).merge(
        intensity_df,
        on=['Year', 'Storm', 'Province'],
        how='inner'
    )
    
    print(f"\nCombined Dataset:")
    print(f"  Total records: {len(combined_df):,}")
    print(f"  Total features: {len(combined_df.columns) - 3}")
    print(f"  Years: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
    
    # Sample correlations between feature groups
    print("\nCross-Group Feature Correlations:")
    print("-" * 60)
    
    # Check correlation between distance and intensity
    distance_intensity_corr = combined_df[
        combined_df['wind_at_closest_approach_kt'] != MISSING_VALUE
    ][['min_distance_km', 'wind_at_closest_approach_kt']].corr()
    
    print("\nDistance vs Intensity:")
    print(distance_intensity_corr)
    
    # Check correlation between weather and intensity
    weather_intensity_corr = combined_df[
        combined_df['max_wind_in_track_kt'] != MISSING_VALUE
    ][['max_wind_gust_kmh', 'max_wind_in_track_kt']].corr()
    
    print("\nObserved Weather vs Storm Intensity:")
    print(weather_intensity_corr)
    
    print("\n" + "=" * 60)
    return combined_df


# ============================================================================
# EXAMPLE 5: Handling Missing Data for ML
# ============================================================================

def example_handle_missing_data():
    """
    Demonstrate strategies for handling missing intensity data in ML pipeline.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Handling Missing Data for ML")
    print("=" * 60)
    
    intensity_file = '../../Feature_Engineering_Data/group3/intensity_features_group3.csv'
    
    if not Path(intensity_file).exists():
        print(f"Features file not found: {intensity_file}")
        return None
    
    features_df = pd.read_csv(intensity_file)
    
    print("\nStrategy 1: Replace -999 with NaN for sklearn")
    print("-" * 60)
    
    features_clean = features_df.copy()
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Year', 'Storm', 'Province']]
    
    for col in feature_cols:
        features_clean.loc[features_clean[col] == MISSING_VALUE, col] = np.nan
    
    print(f"  Converted {MISSING_VALUE} to NaN")
    print(f"  Can now use sklearn's SimpleImputer")
    
    print("\nStrategy 2: Create 'data available' indicator features")
    print("-" * 60)
    
    for col in feature_cols:
        indicator_col = f"{col}_available"
        features_clean[indicator_col] = (~features_clean[col].isna()).astype(int)
        print(f"  Created: {indicator_col}")
    
    print("\nStrategy 3: Impute with median (for models that need it)")
    print("-" * 60)
    
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy='median')
    features_imputed = features_clean.copy()
    features_imputed[feature_cols] = imputer.fit_transform(features_clean[feature_cols])
    
    print(f"  Imputed missing values with median")
    print(f"  Ready for models that don't handle missing data")
    
    print("\nRecommendation:")
    print("-" * 60)
    print("  • Tree-based models (XGBoost, Random Forest): Can handle -999 natively")
    print("  • Linear models: Use Strategy 2 (indicators) + Strategy 3 (imputation)")
    print("  • Neural networks: Use Strategy 2 + 3, normalize features")
    
    print("\n" + "=" * 60)
    return features_clean


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print(" STORM INTENSITY FEATURE ENGINEERING - USAGE EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Run all examples
    example_realtime_intensity()
    example_single_storm()
    example_data_completeness()
    example_combine_all_features()
    example_handle_missing_data()
    
    print("\n" + "*" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("*" * 60)
    print("\n")



