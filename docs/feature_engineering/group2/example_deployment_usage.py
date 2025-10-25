"""
Example: Using the Weather Feature Engineering Pipeline for Deployment

This demonstrates how to use the modular functions for:
1. Training: Processing historical weather data
2. Deployment: Real-time feature extraction from forecast/observation data
"""

import pandas as pd
from pathlib import Path
from group2_weather_feature_engineering import (
    extract_weather_features_for_deployment,
    compute_all_weather_features,
    process_single_weather_file
)


# ============================================================================
# EXAMPLE 1: Deployment Scenario - Real-time Weather Feature Extraction
# ============================================================================

def example_realtime_weather():
    """
    Simulate processing real-time weather observations for deployment.
    """
    print("=" * 60)
    print("EXAMPLE 1: Real-time Weather Feature Extraction")
    print("=" * 60)
    
    # Simulated weather observations (e.g., from Open-Meteo API)
    dates = [
        '2024-11-15', '2024-11-16', '2024-11-17', 
        '2024-11-18', '2024-11-19'
    ]
    wind_gusts = [45.2, 78.5, 92.3, 65.4, 38.1]  # km/h
    wind_speeds = [32.1, 58.3, 71.2, 48.5, 28.9]  # km/h
    precipitation_sums = [12.5, 45.8, 87.3, 32.1, 8.4]  # mm
    precipitation_hours = [4.0, 8.0, 12.0, 6.0, 2.0]  # hours
    
    closest_approach_date = '2024-11-17'  # Peak of the storm
    
    # Extract features
    features = extract_weather_features_for_deployment(
        dates=dates,
        wind_gusts=wind_gusts,
        wind_speeds=wind_speeds,
        precipitation_sums=precipitation_sums,
        precipitation_hours=precipitation_hours,
        closest_approach_date=closest_approach_date
    )
    
    print("\nExtracted Weather Features:")
    print("-" * 60)
    for feature_name, value in sorted(features.items()):
        print(f"  {feature_name:35s}: {value:>12.2f}")
    
    print("\n" + "=" * 60)
    return features


# ============================================================================
# EXAMPLE 2: Processing a Single Storm-Province Pair
# ============================================================================

def example_single_storm_province():
    """
    Process weather features for a specific storm and province.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Single Storm-Province Weather Features")
    print("=" * 60)
    
    # Load the weather data for a specific storm
    weather_df = pd.read_csv('Weather_location_data/2010/2010_Agaton.csv')
    
    # Filter for a specific province
    province = 'Metro Manila'
    province_df = weather_df[weather_df['province'] == province].copy()
    
    print(f"\nStorm: 2010 Agaton")
    print(f"Province: {province}")
    print(f"Data points: {len(province_df)}")
    
    # Compute all features
    features = compute_all_weather_features(province_df)
    
    print("\nKey Weather Features:")
    print("-" * 60)
    key_features = [
        'max_wind_gust_kmh',
        'max_wind_speed_kmh',
        'total_precipitation_mm',
        'days_with_heavy_rain',
        'compound_hazard_score'
    ]
    
    for feature in key_features:
        if feature in features:
            print(f"  {feature:35s}: {features[feature]:>12.2f}")
    
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
    
    # Process single weather file
    weather_file = Path('Weather_location_data/2021/2021_Odette.csv')
    
    if weather_file.exists():
        features_df = process_single_weather_file(weather_file)
        
        print(f"\nStorm: {weather_file.stem}")
        print(f"Provinces processed: {len(features_df)}")
        print(f"Features per province: {len(features_df.columns) - 3}")
        
        # Show top 5 most impacted provinces by max wind gust
        print("\nTop 5 Most Impacted Provinces (by max wind gust):")
        print("-" * 60)
        top_5 = features_df.nlargest(5, 'max_wind_gust_kmh')[
            ['Province', 'max_wind_gust_kmh', 'total_precipitation_mm', 'compound_hazard_days']
        ]
        print(top_5.to_string(index=False))
        
        # Show top 5 by rainfall
        print("\nTop 5 Most Impacted Provinces (by total rainfall):")
        print("-" * 60)
        top_5_rain = features_df.nlargest(5, 'total_precipitation_mm')[
            ['Province', 'total_precipitation_mm', 'days_with_heavy_rain', 'max_wind_gust_kmh']
        ]
        print(top_5_rain.to_string(index=False))
        
        print("\n" + "=" * 60)
        return features_df
    else:
        print(f"Weather file not found: {weather_file}")
        return None


# ============================================================================
# EXAMPLE 4: Loading Pre-computed Features for ML Pipeline
# ============================================================================

def example_load_for_ml():
    """
    Load the pre-computed weather features for machine learning.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Loading Weather Features for ML Pipeline")
    print("=" * 60)
    
    features_file = 'Feature_Engineering_Data/group2/weather_features_group2.csv'
    
    if Path(features_file).exists():
        # Load the full feature dataset
        features_df = pd.read_csv(features_file)
        
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
            'max_wind_gust_kmh',
            'total_precipitation_mm',
            'compound_hazard_score',
            'days_with_heavy_rain'
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
    else:
        print(f"Features file not found: {features_file}")
        print("Run the main script first to generate features.")
        return None


# ============================================================================
# EXAMPLE 5: Combining Distance and Weather Features
# ============================================================================

def example_combine_features():
    """
    Demonstrate combining distance and weather features for ML.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Combining Distance and Weather Features")
    print("=" * 60)
    
    distance_file = 'Feature_Engineering_Data/group1/distance_features_group1.csv'
    weather_file = 'Feature_Engineering_Data/group2/weather_features_group2.csv'
    
    if Path(distance_file).exists() and Path(weather_file).exists():
        # Load both feature sets
        distance_df = pd.read_csv(distance_file)
        weather_df = pd.read_csv(weather_file)
        
        # Merge on Year, Storm, Province
        combined_df = pd.merge(
            distance_df,
            weather_df,
            on=['Year', 'Storm', 'Province'],
            how='inner'
        )
        
        print(f"\nCombined Dataset:")
        print(f"  Total records: {len(combined_df):,}")
        print(f"  Distance features: {len(distance_df.columns) - 3}")
        print(f"  Weather features: {len(weather_df.columns) - 3}")
        print(f"  Total features: {len(combined_df.columns) - 3}")
        
        # Show sample correlations between distance and weather
        print("\nSample Feature Correlations:")
        print("-" * 60)
        
        sample_corr = combined_df[[
            'min_distance_km',
            'max_wind_gust_kmh',
            'total_precipitation_mm',
            'compound_hazard_score'
        ]].corr()
        
        print(sample_corr.to_string())
        
        print("\n" + "=" * 60)
        return combined_df
    else:
        print("One or both feature files not found.")
        print("Run both feature engineering scripts first.")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print(" WEATHER FEATURE ENGINEERING - USAGE EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Run all examples
    example_realtime_weather()
    example_single_storm_province()
    example_batch_one_storm()
    example_load_for_ml()
    example_combine_features()
    
    print("\n" + "*" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("*" * 60)
    print("\n")

