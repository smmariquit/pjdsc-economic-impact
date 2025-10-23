"""
Example: Using the Distance Feature Engineering Pipeline for Deployment

This demonstrates how to use the modular functions for:
1. Training: Processing historical storm data
2. Deployment: Real-time feature extraction from forecast data
"""

import pandas as pd
from group1_distance_feature_engineering import (
    extract_features_for_deployment,
    compute_all_distance_features,
    process_single_storm_file
)


# ============================================================================
# EXAMPLE 1: Deployment Scenario - Real-time Forecast Processing
# ============================================================================

def example_realtime_forecast():
    """
    Simulate processing a real-time storm forecast for deployment.
    """
    print("=" * 60)
    print("EXAMPLE 1: Real-time Forecast Feature Extraction")
    print("=" * 60)
    
    # Simulated forecast data from JTWC
    # (In production, this would come from your API/data source)
    forecast_distances = [1250.5, 1100.3, 950.8, 750.2, 600.1]  # km
    forecast_bearings = [85.5, 82.3, 78.9, 75.1, 71.8]  # degrees
    forecast_times = [
        '2024-11-15 00:00:00',
        '2024-11-15 12:00:00',
        '2024-11-16 00:00:00',
        '2024-11-16 12:00:00',
        '2024-11-17 00:00:00'
    ]
    
    # Extract features for this province-storm pair
    features = extract_features_for_deployment(
        distances=forecast_distances,
        bearings=forecast_bearings,
        timestamps=forecast_times,
        time_interval_hours=12.0  # Forecast intervals are 12 hours
    )
    
    print("\nExtracted Features:")
    print("-" * 60)
    for feature_name, value in sorted(features.items()):
        print(f"  {feature_name:30s}: {value:>12.2f}")
    
    print("\n" + "=" * 60)
    return features


# ============================================================================
# EXAMPLE 2: Processing a Single Storm-Province Pair
# ============================================================================

def example_single_storm_province():
    """
    Process features for a specific storm and province.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Single Storm-Province Feature Extraction")
    print("=" * 60)
    
    # Load the storm location data for a specific storm
    storm_df = pd.read_csv('../../Feature_Engineering_Data/group0/storm_location_data/2010/2010_Agaton.csv')
    
    # Filter for a specific province
    province = 'Metro Manila'
    province_df = storm_df[storm_df['Province'] == province].copy()
    
    print(f"\nStorm: 2010 Agaton")
    print(f"Province: {province}")
    print(f"Data points: {len(province_df)}")
    
    # Compute all features
    features = compute_all_distance_features(province_df, time_interval_hours=3.0)
    
    print("\nKey Features:")
    print("-" * 60)
    key_features = [
        'min_distance_km',
        'mean_distance_km', 
        'hours_under_300km',
        'approach_speed_kmh',
        'bearing_at_closest_deg',
        'did_cross_province'
    ]
    
    for feature in key_features:
        if feature in features:
            print(f"  {feature:30s}: {features[feature]:>12.2f}")
    
    print("\n" + "=" * 60)
    return features


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Provinces for One Storm
# ============================================================================

def example_batch_one_storm():
    """
    Process all provinces for a single storm.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing All Provinces for One Storm")
    print("=" * 60)
    
    from pathlib import Path
    
    # Process single storm file
    storm_file = Path('../../Feature_Engineering_Data/group0/storm_location_data/2021/2021_Odette.csv')
    
    if storm_file.exists():
        features_df = process_single_storm_file(storm_file)
        
        print(f"\nStorm: {storm_file.stem}")
        print(f"Provinces processed: {len(features_df)}")
        print(f"Features per province: {len(features_df.columns) - 3}")
        
        # Show top 5 most impacted provinces by minimum distance
        print("\nTop 5 Most Impacted Provinces (by closest approach):")
        print("-" * 60)
        top_5 = features_df.nsmallest(5, 'min_distance_km')[
            ['Province', 'min_distance_km', 'hours_under_300km', 'did_cross_province']
        ]
        print(top_5.to_string(index=False))
        
        print("\n" + "=" * 60)
        return features_df
    else:
        print(f"Storm file not found: {storm_file}")
        return None


# ============================================================================
# EXAMPLE 4: Loading Pre-computed Features for ML Pipeline
# ============================================================================

def example_load_for_ml():
    """
    Load the pre-computed features for machine learning.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Loading Features for ML Pipeline")
    print("=" * 60)
    
    # Load the full feature dataset
    features_df = pd.read_csv('../../Feature_Engineering_Data/group1/distance_features_group1.csv')
    
    print(f"\nDataset loaded:")
    print(f"  Total records: {len(features_df):,}")
    print(f"  Years: {features_df['Year'].min()} - {features_df['Year'].max()}")
    print(f"  Storms: {features_df['Storm'].nunique()}")
    print(f"  Provinces: {features_df['Province'].nunique()}")
    print(f"  Features: {len(features_df.columns) - 3}")
    
    # Show feature correlations (example)
    print("\nFeature Statistics:")
    print("-" * 60)
    key_features = [
        'min_distance_km',
        'mean_distance_km',
        'hours_under_300km',
        'approach_speed_kmh'
    ]
    print(features_df[key_features].describe())
    
    # Filter for a specific year (e.g., for training/testing split)
    train_df = features_df[features_df['Year'] <= 2020]
    test_df = features_df[features_df['Year'] > 2020]
    
    print(f"\nTrain/Test Split by Year:")
    print(f"  Training set (<=2020): {len(train_df):,} records")
    print(f"  Test set (>2020): {len(test_df):,} records")
    
    print("\n" + "=" * 60)
    return features_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print(" DISTANCE FEATURE ENGINEERING - USAGE EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Run all examples
    example_realtime_forecast()
    example_single_storm_province()
    example_batch_one_storm()
    example_load_for_ml()
    
    print("\n" + "*" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("*" * 60)
    print("\n")

