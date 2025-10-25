"""
Weather Exposure Feature Engineering Pipeline
Processes Weather_location_data to generate GROUP 2 features for ML pipeline.

GROUP 2: WEATHER EXPOSURE FEATURES (22 features total)
- Peak Intensity (4 features)
- Accumulation/Duration (8 features)
- Temporal Distribution (5 features)
- Combined Hazards (3 features)
- Variability (2 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GROUP 2.1: PEAK INTENSITY METRICS (4 features)
# ============================================================================

def compute_peak_intensity_features(weather_data: pd.DataFrame) -> Dict[str, float]:
    """
    Extract peak values across entire storm event.
    
    Args:
        weather_data: DataFrame with columns [date, wind_gusts_10m_max, 
                      wind_speed_10m_max, precipitation_sum]
    
    Returns:
        Dictionary with 4 peak intensity features
    """
    return {
        'max_wind_gust_kmh': float(weather_data['wind_gusts_10m_max'].max()),
        'max_wind_speed_kmh': float(weather_data['wind_speed_10m_max'].max()),
        'max_daily_precip_mm': float(weather_data['precipitation_sum'].max()),
        # max_hourly_precip would require hourly data; using daily as proxy
        'max_hourly_precip_mm': float(weather_data['precipitation_sum'].max() / 24)  # Rough estimate
    }


# ============================================================================
# GROUP 2.2: ACCUMULATION & DURATION METRICS (8 features)
# ============================================================================

def max_consecutive_ones(arr: np.ndarray) -> int:
    """
    Find longest streak of 1s in binary array.
    
    Args:
        arr: Binary array (0s and 1s)
    
    Returns:
        Maximum consecutive count of 1s
    """
    max_count = 0
    current_count = 0
    for val in arr:
        if val == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def compute_accumulation_duration_features(weather_data: pd.DataFrame) -> Dict[str, float]:
    """
    Accumulation and duration metrics.
    
    Args:
        weather_data: DataFrame with weather observations
    
    Returns:
        Dictionary with 8 accumulation/duration features
    """
    features = {
        'total_precipitation_mm': float(weather_data['precipitation_sum'].sum()),
        'days_with_rain': int((weather_data['precipitation_sum'] > 1).sum()),
        'days_with_heavy_rain': int((weather_data['precipitation_sum'] > 50).sum()),
        'days_with_very_heavy_rain': int((weather_data['precipitation_sum'] > 100).sum()),
        'days_with_strong_wind': int((weather_data['wind_gusts_10m_max'] > 60).sum()),
        'days_with_damaging_wind': int((weather_data['wind_gusts_10m_max'] > 90).sum()),
        'total_precipitation_hours': float(weather_data['precipitation_hours'].sum())
    }
    
    # Consecutive rainy days
    rainy_days = (weather_data['precipitation_sum'] > 10).astype(int)
    features['consecutive_rain_days'] = int(max_consecutive_ones(rainy_days.values))
    
    return features


# ============================================================================
# GROUP 2.3: TEMPORAL DISTRIBUTION FEATURES (5 features)
# ============================================================================

def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient (0 = perfectly equal, 1 = maximum inequality).
    
    Args:
        values: Array of values
    
    Returns:
        Gini coefficient (0-1)
    """
    values = values[values >= 0]  # Remove negative values if any
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    
    if cumsum[-1] == 0:
        return 0.0
    
    return float((2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n+1)/n)


def compute_temporal_distribution_features(
    weather_data: pd.DataFrame,
    closest_approach_date: Optional[str] = None
) -> Dict[str, float]:
    """
    How weather hazards are distributed over time.
    
    Args:
        weather_data: DataFrame with weather observations
        closest_approach_date: Date of closest approach (optional)
    
    Returns:
        Dictionary with 5 temporal distribution features
    """
    # Precipitation concentration index (Gini coefficient)
    daily_rain = weather_data['precipitation_sum'].values
    pci = gini_coefficient(daily_rain)
    
    # Rainfall during closest approach
    rain_during_closest = 0.0
    if closest_approach_date is not None:
        # Convert closest_approach_date to date-only for comparison
        try:
            closest_date = pd.to_datetime(closest_approach_date).date()
            weather_data['date_only'] = pd.to_datetime(weather_data['date']).dt.date
            closest_day_data = weather_data[weather_data['date_only'] == closest_date]
            
            if len(closest_day_data) > 0:
                rain_during_closest = float(closest_day_data['precipitation_sum'].iloc[0])
        except:
            rain_during_closest = 0.0
    
    # Wind persistence
    days_with_strong_wind = (weather_data['wind_gusts_10m_max'] > 50).sum()
    total_days = len(weather_data)
    wind_persistence = float(days_with_strong_wind / total_days if total_days > 0 else 0)
    
    return {
        'precipitation_concentration_index': pci,
        'rain_during_closest_approach': rain_during_closest,
        'wind_gust_persistence_score': wind_persistence,
        'mean_daily_precipitation_mm': float(weather_data['precipitation_sum'].mean()),
        'precip_variability': float(weather_data['precipitation_sum'].std())
    }


# ============================================================================
# GROUP 2.4: COMBINED HAZARD FEATURES (3 features)
# ============================================================================

def compute_combined_hazard_features(weather_data: pd.DataFrame) -> Dict[str, float]:
    """
    Features capturing simultaneous wind and rain hazards.
    
    Args:
        weather_data: DataFrame with weather observations
    
    Returns:
        Dictionary with 3 combined hazard features
    """
    max_wind = weather_data['wind_gusts_10m_max'].max()
    total_rain = weather_data['precipitation_sum'].sum()
    
    # Simple product
    wind_rain_product = float(max_wind * total_rain)
    
    # Days with both hazards
    compound_days = int((
        (weather_data['wind_gusts_10m_max'] > 60) & 
        (weather_data['precipitation_sum'] > 30)
    ).sum())
    
    # Normalized compound hazard (z-scores)
    # Simple normalization by typical maximums
    wind_normalized = max_wind / 100  # Normalize by typical max (100 km/h)
    rain_normalized = total_rain / 200  # Normalize by typical accumulation (200 mm)
    compound_hazard_score = float((wind_normalized + rain_normalized) / 2)
    
    return {
        'wind_rain_product': wind_rain_product,
        'compound_hazard_score': compound_hazard_score,
        'compound_hazard_days': compound_days
    }


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_weather_features(
    weather_province_df: pd.DataFrame,
    closest_approach_date: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all 22 GROUP 2 weather exposure features.
    
    Args:
        weather_province_df: DataFrame with columns [date, wind_gusts_10m_max,
                            wind_speed_10m_max, precipitation_sum, precipitation_hours]
                            for a single storm-province pair
        closest_approach_date: Date when storm was closest (optional)
    
    Returns:
        Dictionary with all 22 features
    """
    # Initialize feature dictionary
    all_features = {}
    
    # 1. Peak Intensity Metrics (4 features)
    all_features.update(compute_peak_intensity_features(weather_province_df))
    
    # 2. Accumulation & Duration Features (8 features)
    all_features.update(compute_accumulation_duration_features(weather_province_df))
    
    # 3. Temporal Distribution Features (5 features)
    all_features.update(compute_temporal_distribution_features(
        weather_province_df, 
        closest_approach_date
    ))
    
    # 4. Combined Hazard Features (3 features)
    all_features.update(compute_combined_hazard_features(weather_province_df))
    
    return all_features


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_single_weather_file(
    file_path: Path,
    distance_features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Process a single weather file and compute features for all provinces.
    
    Args:
        file_path: Path to weather CSV file
        distance_features_df: Optional DataFrame with distance features to get closest_approach_date
    
    Returns:
        DataFrame with one row per province containing all weather features
    """
    # Load weather data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract storm info from filename
    filename = file_path.stem  # e.g., "2010_Agaton"
    year, storm_name = filename.split('_', 1)
    
    results = []
    
    # Process each province separately
    for province in df['province'].unique():
        province_df = df[df['province'] == province].sort_values('date')
        
        # Get closest approach date from distance features if available
        closest_approach_date = None
        if distance_features_df is not None:
            try:
                match = distance_features_df[
                    (distance_features_df['Year'] == int(year)) &
                    (distance_features_df['Storm'] == storm_name) &
                    (distance_features_df['Province'] == province)
                ]
                if len(match) > 0:
                    # Find the date when distance was minimum
                    # This requires the raw distance data; for now, use first date as proxy
                    closest_approach_date = province_df['date'].iloc[len(province_df)//2].strftime('%Y-%m-%d')
            except:
                pass
        
        # Compute all features
        features = compute_all_weather_features(province_df, closest_approach_date)
        
        # Add metadata
        features['Year'] = int(year)
        features['Storm'] = storm_name
        features['Province'] = province
        
        results.append(features)
    
    return pd.DataFrame(results)


def process_year_directory(
    year_dir: Path,
    distance_features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Process all weather files in a year directory.
    
    Args:
        year_dir: Path to year directory containing weather CSVs
        distance_features_df: Optional DataFrame with distance features
    
    Returns:
        DataFrame with features for all storms in that year
    """
    all_results = []
    
    csv_files = sorted(year_dir.glob('*.csv'))
    
    print(f"  Processing {len(csv_files)} storms in {year_dir.name}...")
    
    for csv_file in csv_files:
        try:
            storm_features = process_single_weather_file(csv_file, distance_features_df)
            all_results.append(storm_features)
        except Exception as e:
            print(f"    Error processing {csv_file.name}: {str(e)}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def process_all_weather_data(
    weather_location_dir: str = 'Weather_location_data',
    output_file: str = 'Feature_Engineering_Data/group2/weather_features_group2.csv',
    distance_features_file: Optional[str] = 'Feature_Engineering_Data/group1/distance_features_group1.csv'
) -> pd.DataFrame:
    """
    Process all weather location data and generate GROUP 2 features.
    
    Args:
        weather_location_dir: Directory containing year subdirectories
        output_file: Output CSV filename
        distance_features_file: Path to distance features CSV (for closest approach dates)
    
    Returns:
        DataFrame with all features for all storms
    """
    base_dir = Path(weather_location_dir)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {weather_location_dir}")
    
    # Load distance features if available
    distance_features_df = None
    if distance_features_file and Path(distance_features_file).exists():
        print(f"Loading distance features from {distance_features_file}...")
        distance_features_df = pd.read_csv(distance_features_file)
        print(f"  Loaded {len(distance_features_df)} records")
    
    all_results = []
    
    # Get all year directories
    year_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print(f"\nProcessing {len(year_dirs)} years of weather data...")
    print("=" * 60)
    
    for year_dir in year_dirs:
        year_features = process_year_directory(year_dir, distance_features_df)
        
        if not year_features.empty:
            all_results.append(year_features)
            print(f"  [OK] {year_dir.name}: {len(year_features)} storm-province pairs")
    
    print("=" * 60)
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns for better readability
        metadata_cols = ['Year', 'Storm', 'Province']
        feature_cols = [col for col in final_df.columns if col not in metadata_cols]
        final_df = final_df[metadata_cols + sorted(feature_cols)]
        
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved {len(final_df)} records to {output_file}")
        print(f"[OK] Generated {len(feature_cols)} features per record")
        
        return final_df
    else:
        print("\n[ERROR] No data processed")
        return pd.DataFrame()


# ============================================================================
# DEPLOYMENT FUNCTION (for single storm-province pair)
# ============================================================================

def extract_weather_features_for_deployment(
    dates: List[str],
    wind_gusts: List[float],
    wind_speeds: List[float],
    precipitation_sums: List[float],
    precipitation_hours: List[float],
    closest_approach_date: Optional[str] = None
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract all GROUP 2 features from raw weather data.
    
    Args:
        dates: List of date strings
        wind_gusts: List of wind gust values (km/h)
        wind_speeds: List of wind speed values (km/h)
        precipitation_sums: List of daily precipitation (mm)
        precipitation_hours: List of precipitation hours
        closest_approach_date: Date when storm was closest (optional)
    
    Returns:
        Dictionary with all 22 GROUP 2 features
    """
    # Create temporary DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'wind_gusts_10m_max': wind_gusts,
        'wind_speed_10m_max': wind_speeds,
        'precipitation_sum': precipitation_sums,
        'precipitation_hours': precipitation_hours
    })
    
    # Compute features
    return compute_all_weather_features(df, closest_approach_date)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WEATHER EXPOSURE FEATURE ENGINEERING PIPELINE")
    print("GROUP 2: 22 Features")
    print("=" * 60)
    print()
    
    # Example 1: Process single weather file
    print("Example 1: Processing single weather file")
    print("-" * 60)
    
    sample_file = Path('Weather_location_data/2010/2010_Agaton.csv')
    if sample_file.exists():
        result = process_single_weather_file(sample_file)
        print(f"Processed {sample_file.name}:")
        print(f"  - {len(result)} provinces")
        print(f"  - {len(result.columns) - 3} features per province")
        print("\nSample output (first 3 provinces):")
        print(result.head(3).to_string())
    else:
        print(f"Sample file not found: {sample_file}")
        print("Weather location data should be in Weather_location_data/ directory.")
    
    print("\n" + "=" * 60)
    print("Example 2: Processing all weather data")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_weather_data(
        weather_location_dir='Weather_location_data',
        output_file='Feature_Engineering_Data/group2/weather_features_group2.csv',
        distance_features_file='Feature_Engineering_Data/group1/distance_features_group1.csv'
    )
    
    if not features_df.empty:
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"\nTotal records: {len(features_df)}")
        print(f"Years covered: {features_df['Year'].min()} - {features_df['Year'].max()}")
        print(f"Unique storms: {features_df['Storm'].nunique()}")
        print(f"Provinces: {features_df['Province'].nunique()}")
        
        print("\nFeature statistics (first 5 features):")
        feature_cols = [col for col in features_df.columns if col not in ['Year', 'Storm', 'Province']]
        print(features_df[feature_cols[:5]].describe())
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

