"""
Storm Motion & Evolution Feature Engineering Pipeline
Processes Storm_data to generate GROUP 6 features for ML pipeline.

GROUP 6: STORM MOTION & EVOLUTION FEATURES (10 features total)
- Forward Motion (6 features)
- Track Shape & Direction (4 features)

Note: Landfall features (3) excluded due to data unavailability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Constants
MISSING_VALUE = -999


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371.0
    
    return c * r


def safe_float(value, default=MISSING_VALUE):
    """Safely convert value to float, return default if invalid."""
    try:
        if pd.isna(value) or value == '' or value == ' ':
            return default
        return float(value)
    except:
        return default


def circular_mean(angles: np.ndarray) -> float:
    """
    Compute mean of circular data (angles in degrees).
    
    Args:
        angles: Array of angles in degrees
    
    Returns:
        Mean angle in degrees (0-360)
    """
    angles_rad = np.radians(angles)
    mean_angle = np.degrees(
        np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    )
    return mean_angle % 360


def circular_std(angles: np.ndarray) -> float:
    """
    Compute standard deviation of circular data.
    
    Args:
        angles: Array of angles in degrees
    
    Returns:
        Circular standard deviation in degrees
    """
    angles_rad = np.radians(angles)
    R = np.sqrt(np.mean(np.sin(angles_rad))**2 + np.mean(np.cos(angles_rad))**2)
    if R == 0:
        return 0.0
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


# ============================================================================
# GROUP 6.1: FORWARD MOTION FEATURES (6 features)
# ============================================================================

def compute_forward_motion_features(
    track_df: pd.DataFrame,
    province_lat: float,
    province_lon: float
) -> Dict[str, float]:
    """
    Analyze storm translation speed.
    
    Args:
        track_df: DataFrame with storm track data including STORM_SPEED, LAT, LON
        province_lat: Province latitude
        province_lon: Province longitude
    
    Returns:
        Dictionary with 6 forward motion features
    """
    # Extract valid speed data from STORM_SPEED column
    speeds = []
    for _, row in track_df.iterrows():
        speed = safe_float(row.get('STORM_SPEED', ''))
        if speed != MISSING_VALUE:
            speeds.append(speed)
    
    # If STORM_SPEED not available, calculate from positions
    if len(speeds) == 0:
        speeds = []
        for i in range(1, len(track_df)):
            prev_row = track_df.iloc[i-1]
            curr_row = track_df.iloc[i]
            
            dist = haversine_distance(
                prev_row['LAT'], prev_row['LON'],
                curr_row['LAT'], curr_row['LON']
            )
            
            # Calculate time difference
            prev_time = pd.to_datetime(str(prev_row['PH_DAY']) + ' ' + str(prev_row['PH_TIME']))
            curr_time = pd.to_datetime(str(curr_row['PH_DAY']) + ' ' + str(curr_row['PH_TIME']))
            time_diff_hours = (curr_time - prev_time).total_seconds() / 3600
            
            if time_diff_hours > 0:
                speeds.append(dist / time_diff_hours)
    
    # Default values if no speed data
    if len(speeds) == 0:
        return {
            'mean_forward_speed': 20.0,  # Default typical speed
            'max_forward_speed': 30.0,
            'min_forward_speed': 10.0,
            'speed_at_closest_approach': 20.0,
            'is_slow_moving': 0,
            'is_fast_moving': 0
        }
    
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    min_speed = float(np.min(speeds))
    
    # Find closest approach to province
    distances = []
    for _, row in track_df.iterrows():
        dist = haversine_distance(
            row['LAT'], row['LON'],
            province_lat, province_lon
        )
        distances.append(dist)
    
    closest_idx = np.argmin(distances)
    
    # Speed at closest approach
    if closest_idx < len(speeds):
        speed_at_closest = float(speeds[closest_idx])
    else:
        speed_at_closest = mean_speed
    
    return {
        'mean_forward_speed': mean_speed,
        'max_forward_speed': max_speed,
        'min_forward_speed': min_speed,
        'speed_at_closest_approach': speed_at_closest,
        'is_slow_moving': 1 if mean_speed < 15 else 0,
        'is_fast_moving': 1 if mean_speed > 40 else 0
    }


# ============================================================================
# GROUP 6.2: TRACK SHAPE & DIRECTION FEATURES (4 features)
# ============================================================================

def compute_track_shape_features(track_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze storm track geometry and direction changes.
    
    Args:
        track_df: DataFrame with storm track data including STORM_DIR, LAT, LON
    
    Returns:
        Dictionary with 4 track shape features
    """
    # Extract valid direction data
    directions = []
    for _, row in track_df.iterrows():
        direction = safe_float(row.get('STORM_DIR', ''))
        if direction != MISSING_VALUE:
            directions.append(direction)
    
    # Direction features
    if len(directions) >= 2:
        mean_direction = circular_mean(np.array(directions))
        direction_std = circular_std(np.array(directions))
        
        # Check for recurving (major direction change)
        direction_changes = []
        for i in range(1, len(directions)):
            # Calculate smallest angle difference
            diff = abs(directions[i] - directions[i-1])
            if diff > 180:
                diff = 360 - diff
            direction_changes.append(diff)
        
        max_change = max(direction_changes) if direction_changes else 0
        is_recurving = 1 if max_change > 45 else 0
    else:
        mean_direction = 270.0  # Default westward
        direction_std = 0.0
        is_recurving = 0
    
    # Track sinuosity (path complexity)
    if len(track_df) >= 2:
        # Calculate total path length
        total_distance = 0.0
        for i in range(1, len(track_df)):
            prev_row = track_df.iloc[i-1]
            curr_row = track_df.iloc[i]
            dist = haversine_distance(
                prev_row['LAT'], prev_row['LON'],
                curr_row['LAT'], curr_row['LON']
            )
            total_distance += dist
        
        # Calculate straight-line distance
        first_row = track_df.iloc[0]
        last_row = track_df.iloc[-1]
        straight_line_distance = haversine_distance(
            first_row['LAT'], first_row['LON'],
            last_row['LAT'], last_row['LON']
        )
        
        # Sinuosity: 1.0 = perfectly straight, >1.0 = curved
        if straight_line_distance > 0:
            sinuosity = total_distance / straight_line_distance
        else:
            sinuosity = 1.0
    else:
        sinuosity = 1.0
    
    return {
        'mean_direction': float(mean_direction),
        'direction_variability': float(direction_std),
        'is_recurving': int(is_recurving),
        'track_sinuosity': float(sinuosity)
    }


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_motion_features(
    storm_track_df: pd.DataFrame,
    province_lat: float,
    province_lon: float
) -> Dict[str, float]:
    """
    Compute all 10 GROUP 6 storm motion and evolution features.
    
    Args:
        storm_track_df: DataFrame with storm track data
        province_lat: Province latitude
        province_lon: Province longitude
    
    Returns:
        Dictionary with all 10 features
    """
    # Initialize all features
    all_features = {}
    
    # 1. Forward Motion Features (6 features)
    all_features.update(compute_forward_motion_features(
        storm_track_df, province_lat, province_lon
    ))
    
    # 2. Track Shape & Direction Features (4 features)
    all_features.update(compute_track_shape_features(storm_track_df))
    
    return all_features


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_storm_for_all_provinces(
    storm_data_df: pd.DataFrame,
    year: int,
    storm_name: str,
    provinces_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a single storm and compute motion features for all provinces.
    
    Args:
        storm_data_df: DataFrame with storm track data
        year: Year of the storm
        storm_name: Philippine name of the storm
        provinces_df: DataFrame with province locations [Province, Lat, Lng]
    
    Returns:
        DataFrame with one row per province containing all motion features
    """
    results = []
    
    # For each province, compute features
    for _, prov_row in provinces_df.iterrows():
        province = prov_row['Province']
        prov_lat = prov_row['Lat']
        prov_lng = prov_row['Lng']
        
        # Skip if coordinates are missing
        if pd.isna(prov_lat) or pd.isna(prov_lng):
            continue
        
        # Compute features
        features = compute_all_motion_features(
            storm_data_df,
            prov_lat,
            prov_lng
        )
        
        # Add metadata
        features['Year'] = int(year)
        features['Storm'] = storm_name
        features['Province'] = province
        
        results.append(features)
    
    return pd.DataFrame(results)


def process_all_storm_motion_data(
    storm_data_file: str = 'Storm_data/ph_storm_data.csv',
    output_file: str = 'Feature_Engineering_Data/group6/motion_features_group6.csv',
    province_list_file: str = 'Location_data/locations_latlng.csv'
) -> pd.DataFrame:
    """
    Process all storm data and generate GROUP 6 features.
    
    Args:
        storm_data_file: Path to storm data CSV
        output_file: Output CSV filename
        province_list_file: Path to province list CSV
    
    Returns:
        DataFrame with all features for all storms
    """
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    # Load storm data
    if not Path(storm_data_file).exists():
        raise FileNotFoundError(f"Storm data file not found: {storm_data_file}")
    
    storm_data = pd.read_csv(storm_data_file)
    print(f"Loaded {len(storm_data)} storm track points")
    
    # Load province list
    if not Path(province_list_file).exists():
        raise FileNotFoundError(f"Province list file not found: {province_list_file}")
    
    provinces_df = pd.read_csv(province_list_file)
    provinces_df = provinces_df[provinces_df['Province'].notna()]
    print(f"Loaded {len(provinces_df)} provinces")
    
    # Get unique storms
    storms = storm_data[['SEASON', 'PHNAME']].drop_duplicates()
    
    print(f"\nProcessing {len(storms)} storms...")
    print("=" * 60)
    
    all_results = []
    
    for idx, (_, storm_row) in enumerate(storms.iterrows(), 1):
        year = int(storm_row['SEASON'])
        storm_name = storm_row['PHNAME']
        
        try:
            # Get track for this storm
            storm_track = storm_data[
                (storm_data['SEASON'] == year) &
                (storm_data['PHNAME'] == storm_name)
            ].copy()
            
            if len(storm_track) == 0:
                continue
            
            # Process this storm for all provinces
            storm_features = process_storm_for_all_provinces(
                storm_track,
                year,
                storm_name,
                provinces_df
            )
            
            all_results.append(storm_features)
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(storms)} storms...")
        
        except Exception as e:
            print(f"  Error processing {year}_{storm_name}: {str(e)}")
            continue
    
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
# DEPLOYMENT FUNCTION (for single storm)
# ============================================================================

def extract_motion_features_for_deployment(
    track_data: pd.DataFrame,
    province_lat: float,
    province_lon: float
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract all GROUP 6 features from storm track data.
    
    Args:
        track_data: DataFrame with columns [LAT, LON, STORM_SPEED, STORM_DIR, PH_DAY, PH_TIME]
        province_lat: Province latitude
        province_lon: Province longitude
    
    Returns:
        Dictionary with all 10 GROUP 6 features
    """
    return compute_all_motion_features(track_data, province_lat, province_lon)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STORM MOTION & EVOLUTION FEATURE ENGINEERING PIPELINE")
    print("GROUP 6: 10 Features")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_storm_motion_data(
        storm_data_file='Storm_data/ph_storm_data.csv',
        output_file='Feature_Engineering_Data/group6/motion_features_group6.csv',
        province_list_file='Location_data/locations_latlng.csv'
    )
    
    if not features_df.empty:
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"\nTotal records: {len(features_df)}")
        print(f"Years covered: {features_df['Year'].min()} - {features_df['Year'].max()}")
        print(f"Unique storms: {features_df['Storm'].nunique()}")
        print(f"Provinces: {features_df['Province'].nunique()}")
        
        print("\nFeature statistics:")
        feature_cols = [col for col in features_df.columns if col not in ['Year', 'Storm', 'Province']]
        print(features_df[feature_cols].describe())
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


