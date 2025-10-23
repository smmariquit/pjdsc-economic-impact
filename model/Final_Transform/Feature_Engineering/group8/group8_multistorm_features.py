"""
Multi-Storm & Compound Impact Feature Engineering Pipeline
Processes Storm_data to generate GROUP 8 features for ML pipeline.

GROUP 8: MULTI-STORM & COMPOUND IMPACT FEATURES (6 features total)
- Concurrent Storm Detection (4 features)
- Recent Storm History (2 features)

Requires: Storm track data with temporal information
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# Constants
MISSING_VALUE = -999
PAR_BOUNDS = {
    'lat_min': 4.0,
    'lat_max': 21.0,
    'lon_min': 115.0,
    'lon_max': 135.0
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371.0 * c


def is_in_par(lat: float, lon: float) -> bool:
    """Check if coordinates are within Philippine Area of Responsibility."""
    return (PAR_BOUNDS['lat_min'] <= lat <= PAR_BOUNDS['lat_max'] and
            PAR_BOUNDS['lon_min'] <= lon <= PAR_BOUNDS['lon_max'])


def safe_float(value, default=MISSING_VALUE):
    """Safely convert value to float."""
    try:
        if pd.isna(value) or value == '' or value == ' ':
            return default
        return float(value)
    except:
        return default


# ============================================================================
# GROUP 8.1: CONCURRENT STORM DETECTION (4 features)
# ============================================================================

def compute_concurrent_storm_features(
    storm_id: str,
    storm_date: datetime,
    all_storms_data: pd.DataFrame,
    storm_location: Tuple[float, float]
) -> Dict[str, float]:
    """
    Detect and analyze concurrent storms active at the same time.
    
    Args:
        storm_id: Unique identifier for current storm (SEASON_PHNAME)
        storm_date: Specific date to check for concurrent storms
        all_storms_data: Full storm dataset
        storm_location: (lat, lon) of current storm on this date
    
    Returns:
        Dictionary with 4 concurrent storm features
    """
    # Find all other storms active on the same date
    concurrent_storms = all_storms_data[
        (all_storms_data['timestamp'].dt.date == storm_date.date()) &
        (all_storms_data['storm_id'] != storm_id)
    ]
    
    if len(concurrent_storms) == 0:
        return {
            'has_concurrent_storm': 0,
            'concurrent_storms_count': 0,
            'nearest_concurrent_storm_distance': MISSING_VALUE,
            'concurrent_storms_combined_intensity': 0.0
        }
    
    # Get unique concurrent storms
    concurrent_storm_ids = concurrent_storms['storm_id'].unique()
    
    # Calculate distances to each concurrent storm
    distances = []
    total_intensity = 0.0
    
    for other_storm_id in concurrent_storm_ids:
        # Get position of other storm on this date
        other_storm_data = concurrent_storms[
            concurrent_storms['storm_id'] == other_storm_id
        ].iloc[0]
        
        # Calculate distance
        dist = haversine_distance(
            storm_location[0], storm_location[1],
            other_storm_data['LAT'], other_storm_data['LON']
        )
        distances.append(dist)
        
        # Get intensity (use USA_WIND or default)
        wind = safe_float(other_storm_data.get('USA_WIND', ''), default=0)
        if wind == MISSING_VALUE or wind == 0:
            wind = safe_float(other_storm_data.get('TOKYO_WIND', ''), default=0)
        if wind != MISSING_VALUE and wind != 0:
            # Convert knots to km/h
            total_intensity += wind * 1.852
    
    return {
        'has_concurrent_storm': 1,
        'concurrent_storms_count': int(len(concurrent_storm_ids)),
        'nearest_concurrent_storm_distance': float(min(distances)),
        'concurrent_storms_combined_intensity': float(total_intensity)
    }


# ============================================================================
# GROUP 8.2: RECENT STORM HISTORY (2 features)
# ============================================================================

def compute_recent_storm_history(
    storm_start_date: datetime,
    province: str,
    all_storms_summary: pd.DataFrame
) -> Dict[str, float]:
    """
    Analyze recent storm history for rapid succession patterns.
    
    Args:
        storm_start_date: Start date of current storm
        province: Province name
        all_storms_summary: Summary of all storms with start/end dates
    
    Returns:
        Dictionary with 2 recent history features
    """
    # Find storms that ended before current storm started
    previous_storms = all_storms_summary[
        all_storms_summary['end_date'] < storm_start_date
    ].sort_values('end_date', ascending=False)
    
    if len(previous_storms) == 0:
        return {
            'days_since_last_storm': 365.0,  # No recent storm
            'storms_past_30_days': 0
        }
    
    # Days since last storm
    last_storm_end = previous_storms.iloc[0]['end_date']
    days_since = (storm_start_date - last_storm_end).days
    
    # Count storms in past 30 days
    thirty_days_ago = storm_start_date - timedelta(days=30)
    recent_storms = previous_storms[
        previous_storms['end_date'] >= thirty_days_ago
    ]
    storms_count = len(recent_storms)
    
    return {
        'days_since_last_storm': float(max(0, days_since)),
        'storms_past_30_days': int(storms_count)
    }


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_multistorm_features(
    storm_id: str,
    storm_track_df: pd.DataFrame,
    all_storms_data: pd.DataFrame,
    all_storms_summary: pd.DataFrame,
    province: str
) -> List[Dict[str, float]]:
    """
    Compute all 6 GROUP 8 multi-storm features for each timestep of storm.
    
    Args:
        storm_id: Unique identifier (SEASON_PHNAME)
        storm_track_df: Track data for this storm
        all_storms_data: Full dataset with all storms
        all_storms_summary: Summary with storm start/end dates
        province: Province name
    
    Returns:
        List of feature dictionaries, one per timestep
    """
    results = []
    
    storm_start_date = storm_track_df['timestamp'].min()
    
    # Compute history features once (same for all timesteps of this storm)
    history_features = compute_recent_storm_history(
        storm_start_date, province, all_storms_summary
    )
    
    # For each timestep, compute concurrent features
    for _, row in storm_track_df.iterrows():
        storm_location = (row['LAT'], row['LON'])
        
        concurrent_features = compute_concurrent_storm_features(
            storm_id,
            row['timestamp'],
            all_storms_data,
            storm_location
        )
        
        # Combine all features
        all_features = {**concurrent_features, **history_features}
        all_features['timestamp'] = row['timestamp']
        
        results.append(all_features)
    
    return results


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def prepare_storm_data(storm_data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare storm data with timestamps and summary information.
    
    Args:
        storm_data_file: Path to storm data CSV
    
    Returns:
        Tuple of (full_data_with_timestamps, storm_summary)
    """
    print("Loading and preparing storm data...")
    
    # Load full dataset
    df = pd.read_csv(storm_data_file)
    
    # Create timestamp
    df['timestamp'] = pd.to_datetime(
        df['PH_DAY'].astype(str) + ' ' + df['PH_TIME'].astype(str)
    )
    
    # Create storm ID
    df['storm_id'] = df['SEASON'].astype(str) + '_' + df['PHNAME']
    
    # Create storm summary (start/end dates for each storm)
    summary = df.groupby('storm_id').agg({
        'timestamp': ['min', 'max'],
        'SEASON': 'first',
        'PHNAME': 'first'
    }).reset_index()
    
    summary.columns = ['storm_id', 'start_date', 'end_date', 'SEASON', 'PHNAME']
    
    print(f"  Loaded {len(df)} track points from {len(summary)} storms")
    
    return df, summary


def process_all_multistorm_data(
    storm_data_file: str = 'Storm_data/ph_storm_data.csv',
    output_file: str = 'Feature_Engineering_Data/group8/multistorm_features_group8.csv',
    province_list_file: str = 'Location_data/locations_latlng.csv'
) -> pd.DataFrame:
    """
    Process all storm data and generate GROUP 8 features.
    OPTIMIZED: Uses storm-level aggregation instead of timestep iteration.
    
    Args:
        storm_data_file: Path to storm data CSV
        output_file: Output CSV filename
        province_list_file: Path to province list CSV
    
    Returns:
        DataFrame with all features for all storms
    """
    print("=" * 60)
    print("MULTI-STORM FEATURE ENGINEERING (OPTIMIZED)")
    print("=" * 60)
    print()
    
    # Prepare data - but more efficiently
    print("Loading storm data...")
    df = pd.read_csv(storm_data_file)
    df['timestamp'] = pd.to_datetime(df['PH_DAY'].astype(str) + ' ' + df['PH_TIME'].astype(str))
    df['storm_id'] = df['SEASON'].astype(str) + '_' + df['PHNAME']
    
    # Create storm summary with date ranges
    print("Creating storm summary...")
    all_storms_summary = df.groupby('storm_id').agg({
        'timestamp': ['min', 'max'],
        'SEASON': 'first',
        'PHNAME': 'first'
    }).reset_index()
    all_storms_summary.columns = ['storm_id', 'start_date', 'end_date', 'SEASON', 'PHNAME']
    
    print(f"  Found {len(all_storms_summary)} storms")
    
    # Load provinces
    provinces_df = pd.read_csv(province_list_file)
    provinces_df = provinces_df[provinces_df['Province'].notna()]
    province_list = provinces_df['Province'].tolist()
    
    print(f"  Processing for {len(province_list)} provinces")
    print("=" * 60)
    
    all_results = []
    
    # Process each storm
    for idx, storm_row in all_storms_summary.iterrows():
        storm_id = storm_row['storm_id']
        year = int(storm_row['SEASON'])
        storm_name = storm_row['PHNAME']
        start_date = storm_row['start_date']
        end_date = storm_row['end_date']
        
        try:
            # Find concurrent storms (overlapping time periods)
            concurrent_storms = all_storms_summary[
                (all_storms_summary['storm_id'] != storm_id) &
                (
                    # Storm starts during this storm
                    ((all_storms_summary['start_date'] >= start_date) & 
                     (all_storms_summary['start_date'] <= end_date)) |
                    # Storm ends during this storm
                    ((all_storms_summary['end_date'] >= start_date) & 
                     (all_storms_summary['end_date'] <= end_date)) |
                    # This storm is completely within another storm
                    ((all_storms_summary['start_date'] <= start_date) & 
                     (all_storms_summary['end_date'] >= end_date))
                )
            ]
            
            # Concurrent features (same for all provinces)
            has_concurrent = 1 if len(concurrent_storms) > 0 else 0
            concurrent_count = len(concurrent_storms)
            
            # Get track for current storm (for distance calculation)
            current_track = df[df['storm_id'] == storm_id]
            
            # Get combined intensity and calculate distances to concurrent storms
            concurrent_intensity = 0.0
            min_distance = MISSING_VALUE
            
            if len(concurrent_storms) > 0:
                distances = []
                
                for _, conc_storm in concurrent_storms.iterrows():
                    conc_storm_id = conc_storm['storm_id']
                    conc_track = df[df['storm_id'] == conc_storm_id]
                    
                    # Calculate intensity
                    winds = []
                    for _, row in conc_track.iterrows():
                        wind = safe_float(row.get('USA_WIND', ''), default=0)
                        if wind == 0 or wind == MISSING_VALUE:
                            wind = safe_float(row.get('TOKYO_WIND', ''), default=0)
                        if wind != 0 and wind != MISSING_VALUE:
                            winds.append(wind * 1.852)  # Convert to km/h
                    if winds:
                        concurrent_intensity += np.mean(winds)
                    
                    # Calculate minimum distance during overlap period
                    # Find overlapping time window
                    overlap_start = max(start_date, conc_storm['start_date'])
                    overlap_end = min(end_date, conc_storm['end_date'])
                    
                    # Get positions during overlap
                    current_overlap = current_track[
                        (current_track['timestamp'] >= overlap_start) &
                        (current_track['timestamp'] <= overlap_end)
                    ]
                    conc_overlap = conc_track[
                        (conc_track['timestamp'] >= overlap_start) &
                        (conc_track['timestamp'] <= overlap_end)
                    ]
                    
                    # Calculate distances for all combinations during overlap
                    # (approximate - use nearest timestamp matching)
                    storm_distances = []
                    for _, curr_row in current_overlap.iterrows():
                        curr_time = curr_row['timestamp']
                        curr_lat = safe_float(curr_row.get('LAT', ''))
                        curr_lon = safe_float(curr_row.get('LON', ''))
                        
                        if curr_lat == MISSING_VALUE or curr_lon == MISSING_VALUE:
                            continue
                        
                        # Find closest timestamp in concurrent storm (within 6 hours)
                        time_diffs = abs(conc_overlap['timestamp'] - curr_time)
                        if len(time_diffs) > 0 and time_diffs.min().total_seconds() <= 6 * 3600:
                            closest_idx = time_diffs.idxmin()
                            conc_row = conc_overlap.loc[closest_idx]
                            conc_lat = safe_float(conc_row.get('LAT', ''))
                            conc_lon = safe_float(conc_row.get('LON', ''))
                            
                            if conc_lat != MISSING_VALUE and conc_lon != MISSING_VALUE:
                                dist = haversine_distance(curr_lat, curr_lon, conc_lat, conc_lon)
                                storm_distances.append(dist)
                    
                    if storm_distances:
                        distances.append(min(storm_distances))
                
                if distances:
                    min_distance = float(min(distances))
            
            # Recent history features
            previous_storms = all_storms_summary[
                all_storms_summary['end_date'] < start_date
            ].sort_values('end_date', ascending=False)
            
            if len(previous_storms) == 0:
                days_since_last = 365.0
                storms_past_30 = 0
            else:
                last_storm_end = previous_storms.iloc[0]['end_date']
                days_since_last = float(max(0, (start_date - last_storm_end).days))
                
                thirty_days_ago = start_date - timedelta(days=30)
                storms_past_30 = len(previous_storms[
                    previous_storms['end_date'] >= thirty_days_ago
                ])
            
            # For each province (features are same for all provinces in this simplified version)
            for province in province_list:
                features = {
                    'Year': year,
                    'Storm': storm_name,
                    'Province': province,
                    'has_concurrent_storm': has_concurrent,
                    'concurrent_storms_count': concurrent_count,
                    'nearest_concurrent_storm_distance': min_distance,
                    'concurrent_storms_combined_intensity': float(concurrent_intensity),
                    'days_since_last_storm': days_since_last,
                    'storms_past_30_days': storms_past_30
                }
                all_results.append(features)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(all_storms_summary)} storms...")
        
        except Exception as e:
            print(f"  Error processing {storm_id}: {str(e)}")
            continue
    
    print("=" * 60)
    
    # Create DataFrame
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # Reorder columns
        metadata_cols = ['Year', 'Storm', 'Province']
        feature_cols = [col for col in final_df.columns if col not in metadata_cols]
        final_df = final_df[metadata_cols + sorted(feature_cols)]
        
        # Create output directory
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
# DEPLOYMENT FUNCTION
# ============================================================================

def extract_multistorm_features_for_deployment(
    current_storm_date: datetime,
    current_storm_location: Tuple[float, float],
    active_storms: List[Dict],
    recent_storms_history: List[Dict]
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract GROUP 8 features from live storm data.
    
    Args:
        current_storm_date: Date/time of current storm
        current_storm_location: (lat, lon) of current storm
        active_storms: List of other currently active storms 
                      [{id, lat, lon, intensity_kmh}, ...]
        recent_storms_history: List of recent past storms
                              [{end_date}, ...]
    
    Returns:
        Dictionary with all 6 GROUP 8 features
    """
    # Concurrent storm features
    if len(active_storms) == 0:
        concurrent_features = {
            'has_concurrent_storm': 0,
            'concurrent_storms_count': 0,
            'nearest_concurrent_storm_distance': MISSING_VALUE,
            'concurrent_storms_combined_intensity': 0.0
        }
    else:
        distances = []
        total_intensity = 0.0
        
        for storm in active_storms:
            dist = haversine_distance(
                current_storm_location[0], current_storm_location[1],
                storm['lat'], storm['lon']
            )
            distances.append(dist)
            total_intensity += storm.get('intensity_kmh', 0)
        
        concurrent_features = {
            'has_concurrent_storm': 1,
            'concurrent_storms_count': len(active_storms),
            'nearest_concurrent_storm_distance': float(min(distances)),
            'concurrent_storms_combined_intensity': float(total_intensity)
        }
    
    # Recent history features
    if len(recent_storms_history) == 0:
        history_features = {
            'days_since_last_storm': 365.0,
            'storms_past_30_days': 0
        }
    else:
        # Sort by end date
        sorted_history = sorted(recent_storms_history, 
                               key=lambda x: x['end_date'], 
                               reverse=True)
        
        last_storm_end = sorted_history[0]['end_date']
        days_since = (current_storm_date - last_storm_end).days
        
        # Count storms in past 30 days
        thirty_days_ago = current_storm_date - timedelta(days=30)
        recent_count = sum(1 for s in sorted_history 
                          if s['end_date'] >= thirty_days_ago)
        
        history_features = {
            'days_since_last_storm': float(max(0, days_since)),
            'storms_past_30_days': int(recent_count)
        }
    
    return {**concurrent_features, **history_features}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-STORM & COMPOUND IMPACT FEATURE ENGINEERING")
    print("GROUP 8: 6 Features")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_multistorm_data(
        storm_data_file='Storm_data/ph_storm_data.csv',
        output_file='Feature_Engineering_Data/group8/multistorm_features_group8.csv',
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
        
        print("\nFeature Statistics:")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Year', 'Storm', 'Province']]
        print(features_df[feature_cols].describe())
        
        # Show concurrent storm statistics
        print("\nConcurrent Storm Analysis:")
        print(f"  Storms with concurrent activity: {features_df['has_concurrent_storm'].sum()}")
        print(f"  Max concurrent storms: {features_df['concurrent_storms_count'].max()}")
        
        print("\nRapid Succession Analysis:")
        rapid_succession = (features_df['days_since_last_storm'] < 7).sum()
        print(f"  Storm events within 7 days of previous: {rapid_succession}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

