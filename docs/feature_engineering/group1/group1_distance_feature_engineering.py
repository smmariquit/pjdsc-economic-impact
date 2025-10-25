"""
Distance & Proximity Feature Engineering Pipeline
Processes storm_location_data to generate GROUP 1 features for ML pipeline.

GROUP 1: DISTANCE & PROXIMITY FEATURES (26 features total)
- Basic Distance (5 features)
- Duration Thresholds (5 features)
- Forecast Horizons (5 features)
- Integrated Proximity (3 features)
- Approach/Departure (4 features)
- Geometric (4 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GROUP 1.1: BASIC DISTANCE METRICS (5 features)
# ============================================================================

def compute_basic_distance_features(distances: np.ndarray) -> Dict[str, float]:
    """
    Compute fundamental distance statistics.
    
    Args:
        distances: Array of distance measurements (km)
    
    Returns:
        Dictionary with 5 basic distance features
    """
    return {
        'min_distance_km': float(np.min(distances)),
        'mean_distance_km': float(np.mean(distances)),
        'max_distance_km': float(np.max(distances)),
        'distance_range_km': float(np.max(distances) - np.min(distances)),
        'distance_std_km': float(np.std(distances))
    }


# ============================================================================
# GROUP 1.2: THRESHOLD-BASED DURATION FEATURES (5 features)
# ============================================================================

def compute_duration_features(
    distances: np.ndarray, 
    timestamps: pd.Series,
    time_interval_hours: float = 3.0
) -> Dict[str, float]:
    """
    Compute duration within distance thresholds.
    
    Args:
        distances: Array of distance measurements (km)
        timestamps: Series of timestamps
        time_interval_hours: Time between observations (default: 3 hours)
    
    Returns:
        Dictionary with 5 duration features
    """
    features = {}
    
    thresholds = [50, 100, 200, 300, 500]
    
    for threshold in thresholds:
        # Count points within threshold
        count = np.sum(distances < threshold)
        
        # Convert to hours (assuming regular time intervals)
        hours = count * time_interval_hours
        
        features[f'hours_under_{threshold}km'] = float(hours)
    
    return features


# ============================================================================
# GROUP 1.3: FORECAST HORIZON FEATURES (5 features)
# ============================================================================

def compute_forecast_horizon_features(
    distances: np.ndarray,
    timestamps: pd.Series
) -> Dict[str, float]:
    """
    Extract distances at standard forecast intervals.
    Simulates operational forecast structure.
    
    Args:
        distances: Array of distance measurements (km)
        timestamps: Series of timestamps
    
    Returns:
        Dictionary with 5 forecast horizon features
    """
    features = {}
    
    # Convert to numpy for easier indexing
    timestamps_array = pd.to_datetime(timestamps).values
    
    if len(timestamps_array) == 0:
        return {
            'distance_at_current': -999,
            'distance_at_12hr': -999,
            'distance_at_24hr': -999,
            'distance_at_48hr': -999,
            'distance_at_72hr': -999
        }
    
    # Reference time (first timestamp)
    t0 = timestamps_array[0]
    
    # Define forecast horizons
    horizons = {
        'distance_at_current': 0,
        'distance_at_12hr': 12,
        'distance_at_24hr': 24,
        'distance_at_48hr': 48,
        'distance_at_72hr': 72
    }
    
    for feature_name, hours in horizons.items():
        target_time = t0 + np.timedelta64(hours, 'h')
        
        # Find closest timestamp to target
        time_diffs = np.abs(timestamps_array - target_time)
        closest_idx = np.argmin(time_diffs)
        
        # Check if within reasonable tolerance (6 hours)
        if time_diffs[closest_idx] <= np.timedelta64(6, 'h'):
            features[feature_name] = float(distances[closest_idx])
        else:
            # No data point close enough to this horizon
            features[feature_name] = -999  # Missing indicator
    
    return features


# ============================================================================
# GROUP 1.4: INTEGRATED PROXIMITY METRICS (3 features)
# ============================================================================

def compute_integrated_proximity_features(distances: np.ndarray) -> Dict[str, float]:
    """
    Compute weighted proximity metrics that emphasize closer positions.
    
    Args:
        distances: Array of distance measurements (km)
    
    Returns:
        Dictionary with 3 integrated proximity features
    """
    # Add small constant to avoid division by zero
    epsilon = 1.0
    
    # Integrated proximity (sum of inverse-square distances)
    integrated_proximity = float(np.sum(1 / (distances**2 + epsilon)))
    
    # Weighted exposure (exponential decay with distance)
    weighted_exposure = float(np.sum(np.exp(-distances / 100)))
    
    # Peak proximity
    proximity_peak = float(np.max(1 / (distances**2 + epsilon)))
    
    return {
        'integrated_proximity': integrated_proximity,
        'weighted_exposure_hours': weighted_exposure,
        'proximity_peak': proximity_peak
    }


# ============================================================================
# GROUP 1.5: APPROACH/DEPARTURE DYNAMICS (4 features)
# ============================================================================

def compute_approach_departure_features(
    distances: np.ndarray,
    timestamps: pd.Series
) -> Dict[str, float]:
    """
    Analyze storm approach and departure dynamics.
    
    Args:
        distances: Array of distance measurements (km)
        timestamps: Series of timestamps
    
    Returns:
        Dictionary with 4 approach/departure features
    """
    features = {
        'approach_speed_kmh': 0.0,
        'departure_speed_kmh': 0.0,
        'time_approaching_hours': 0.0,
        'time_departing_hours': 0.0
    }
    
    if len(distances) < 2:
        return features
    
    # Find closest approach index
    closest_idx = np.argmin(distances)
    
    timestamps_array = pd.to_datetime(timestamps).values
    
    # Approach phase (before closest)
    if closest_idx > 0:
        approach_distances = distances[:closest_idx + 1]
        approach_times = timestamps_array[:closest_idx + 1]
        
        # Calculate approach speed
        distance_change = float(approach_distances[0] - approach_distances[-1])
        time_change_seconds = (approach_times[-1] - approach_times[0]) / np.timedelta64(1, 's')
        time_change_hours = time_change_seconds / 3600
        
        if time_change_hours > 0:
            features['approach_speed_kmh'] = distance_change / time_change_hours
            features['time_approaching_hours'] = time_change_hours
    
    # Departure phase (after closest)
    if closest_idx < len(distances) - 1:
        departure_distances = distances[closest_idx:]
        departure_times = timestamps_array[closest_idx:]
        
        distance_change = float(departure_distances[-1] - departure_distances[0])
        time_change_seconds = (departure_times[-1] - departure_times[0]) / np.timedelta64(1, 's')
        time_change_hours = time_change_seconds / 3600
        
        if time_change_hours > 0:
            features['departure_speed_kmh'] = distance_change / time_change_hours
            features['time_departing_hours'] = time_change_hours
    
    return features


# ============================================================================
# GROUP 1.6: GEOMETRIC FEATURES (4 features)
# ============================================================================

def compute_geometric_features(
    distances: np.ndarray,
    bearings: np.ndarray
) -> Dict[str, float]:
    """
    Compute geometric relationships between storm and province.
    
    Args:
        distances: Array of distance measurements (km)
        bearings: Array of bearing measurements (degrees)
    
    Returns:
        Dictionary with 4 geometric features
    """
    # Find closest approach index
    closest_idx = np.argmin(distances)
    
    features = {
        'bearing_at_closest_deg': float(bearings[closest_idx]),
        'bearing_variability_deg': float(np.std(bearings)),
        'did_cross_province': 1 if distances[closest_idx] < 50 else 0
    }
    
    # Approach angle calculation requires storm motion direction
    # Since we have bearings from province to storm, we can estimate direction change
    if len(bearings) > 1 and closest_idx > 0:
        # Calculate bearing change near closest approach
        bearing_before = bearings[max(0, closest_idx - 1)]
        bearing_at = bearings[closest_idx]
        
        # Angular difference
        angle_diff = abs(bearing_at - bearing_before)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        features['approach_angle_deg'] = float(angle_diff)
    else:
        features['approach_angle_deg'] = -999  # Unknown
    
    return features


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_distance_features(
    storm_province_df: pd.DataFrame,
    time_interval_hours: float = 3.0
) -> Dict[str, float]:
    """
    Compute all 26 GROUP 1 distance and proximity features.
    
    Args:
        storm_province_df: DataFrame with columns [Timestamp, Distance_KM, Bearing_Degrees]
                          for a single storm-province pair
        time_interval_hours: Time between observations (default: 3 hours)
    
    Returns:
        Dictionary with all 26 features
    """
    # Extract arrays
    distances = storm_province_df['Distance_KM'].values
    bearings = storm_province_df['Bearing_Degrees'].values
    timestamps = storm_province_df['Timestamp']
    
    # Initialize feature dictionary
    all_features = {}
    
    # 1. Basic Distance Metrics (5 features)
    all_features.update(compute_basic_distance_features(distances))
    
    # 2. Duration Features (5 features)
    all_features.update(compute_duration_features(distances, timestamps, time_interval_hours))
    
    # 3. Forecast Horizon Features (5 features)
    all_features.update(compute_forecast_horizon_features(distances, timestamps))
    
    # 4. Integrated Proximity Features (3 features)
    all_features.update(compute_integrated_proximity_features(distances))
    
    # 5. Approach/Departure Features (4 features)
    all_features.update(compute_approach_departure_features(distances, timestamps))
    
    # 6. Geometric Features (4 features)
    all_features.update(compute_geometric_features(distances, bearings))
    
    return all_features


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_single_storm_file(
    file_path: Path,
    time_interval_hours: float = 3.0
) -> pd.DataFrame:
    """
    Process a single storm file and compute features for all provinces.
    
    Args:
        file_path: Path to storm location CSV file
        time_interval_hours: Time between observations
    
    Returns:
        DataFrame with one row per province containing all features
    """
    # Load storm-province distance data
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract storm info from filename
    filename = file_path.stem  # e.g., "2010_Agaton"
    year, storm_name = filename.split('_', 1)
    
    results = []
    
    # Process each province separately
    for province in df['Province'].unique():
        province_df = df[df['Province'] == province].sort_values('Timestamp')
        
        # Compute all features
        features = compute_all_distance_features(province_df, time_interval_hours)
        
        # Add metadata
        features['Year'] = int(year)
        features['Storm'] = storm_name
        features['Province'] = province
        
        results.append(features)
    
    return pd.DataFrame(results)


def process_year_directory(
    year_dir: Path,
    time_interval_hours: float = 3.0
) -> pd.DataFrame:
    """
    Process all storm files in a year directory.
    
    Args:
        year_dir: Path to year directory containing storm CSVs
        time_interval_hours: Time between observations
    
    Returns:
        DataFrame with features for all storms in that year
    """
    all_results = []
    
    csv_files = sorted(year_dir.glob('*.csv'))
    
    print(f"  Processing {len(csv_files)} storms in {year_dir.name}...")
    
    for csv_file in csv_files:
        try:
            storm_features = process_single_storm_file(csv_file, time_interval_hours)
            all_results.append(storm_features)
        except Exception as e:
            print(f"    Error processing {csv_file.name}: {str(e)}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def process_all_storm_data(
    storm_location_dir: str = 'Feature_Engineering_Data/group0/storm_location_data',
    output_file: str = 'Feature_Engineering_Data/group1/distance_features_group1.csv',
    time_interval_hours: float = 3.0
) -> pd.DataFrame:
    """
    Process all storm location data and generate GROUP 1 features.
    
    Args:
        storm_location_dir: Directory containing year subdirectories
        output_file: Output CSV filename
        time_interval_hours: Time between observations
    
    Returns:
        DataFrame with all features for all storms
    """
    base_dir = Path(storm_location_dir)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {storm_location_dir}")
    
    all_results = []
    
    # Get all year directories
    year_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(year_dirs)} years of storm data...")
    print("=" * 60)
    
    for year_dir in year_dirs:
        year_features = process_year_directory(year_dir, time_interval_hours)
        
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

def extract_features_for_deployment(
    distances: List[float],
    bearings: List[float],
    timestamps: List[str],
    time_interval_hours: float = 3.0
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract all GROUP 1 features from raw storm track data.
    
    Args:
        distances: List of distances (km) from province to storm
        bearings: List of bearings (degrees) from province to storm
        timestamps: List of timestamp strings
        time_interval_hours: Time between observations
    
    Returns:
        Dictionary with all 26 GROUP 1 features
    """
    # Create temporary DataFrame
    df = pd.DataFrame({
        'Distance_KM': distances,
        'Bearing_Degrees': bearings,
        'Timestamp': pd.to_datetime(timestamps)
    })
    
    # Compute features
    return compute_all_distance_features(df, time_interval_hours)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DISTANCE & PROXIMITY FEATURE ENGINEERING PIPELINE")
    print("GROUP 1: 26 Features")
    print("=" * 60)
    print()
    
    # Example 1: Process single storm file
    print("Example 1: Processing single storm file")
    print("-" * 60)
    
    sample_file = Path('Feature_Engineering_Data/group0/storm_location_data/2010/2010_Agaton.csv')
    if sample_file.exists():
        result = process_single_storm_file(sample_file)
        print(f"Processed {sample_file.name}:")
        print(f"  - {len(result)} provinces")
        print(f"  - {len(result.columns) - 3} features per province")
        print("\nSample output (first 3 provinces):")
        print(result.head(3).to_string())
    else:
        print(f"Sample file not found: {sample_file}")
        print("Run batch_storm_distance_calc.py first to generate storm location data.")
    
    print("\n" + "=" * 60)
    print("Example 2: Processing all storm data")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_storm_data(
        storm_location_dir='Feature_Engineering_Data/group0/storm_location_data',
        output_file='Feature_Engineering_Data/group1/distance_features_group1.csv',
        time_interval_hours=3.0
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

