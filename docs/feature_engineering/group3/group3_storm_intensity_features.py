"""
Storm Intensity Feature Engineering Pipeline
Processes Storm_data to generate GROUP 3 features for ML pipeline.

GROUP 3: STORM INTENSITY FEATURES (7 features total)
- Intensity at Critical Moments (4 features)
- Intensity Evolution Features (3 features)

⚠️ IMPORTANT: These features have 40-70% missing data!
Missing values are indicated with -999
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Constants
USA_TO_TOKYO_FACTOR = 0.88  # Convert 1-minute wind (USA) to 10-minute wind (TOKYO)
KMH_TO_KNOTS = 0.539957  # Convert km/h to knots
MISSING_VALUE = -999


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_float(value, default=MISSING_VALUE):
    """
    Safely convert value to float, return default if invalid.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        if pd.isna(value) or value == '' or value == ' ':
            return default
        return float(value)
    except:
        return default


def get_best_wind_value(tokyo_wind, usa_wind):
    """
    Get best available wind value, preferring TOKYO (10-min) over USA (1-min).
    
    Args:
        tokyo_wind: Tokyo wind value (10-minute sustained)
        usa_wind: USA wind value (1-minute sustained)
    
    Returns:
        Best available wind value in knots (10-minute equivalent) or MISSING_VALUE
    """
    tokyo = safe_float(tokyo_wind)
    usa = safe_float(usa_wind)
    
    if tokyo != MISSING_VALUE:
        return tokyo
    elif usa != MISSING_VALUE:
        # Convert USA 1-minute to Tokyo 10-minute equivalent
        return usa * USA_TO_TOKYO_FACTOR
    else:
        return MISSING_VALUE


# ============================================================================
# GROUP 3.1: INTENSITY AT CRITICAL MOMENTS (4 features)
# ============================================================================

def compute_intensity_at_critical_moments(
    track_df: pd.DataFrame,
    closest_approach_idx: int,
    summary_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Extract intensity data at key moments (handle missing data gracefully).
    Uses summary data as fallback when detailed track data is missing.
    
    Args:
        track_df: DataFrame with storm track data
        closest_approach_idx: Index of closest approach point
        summary_data: Optional dict with summary data (Peak_Windspeed_kmh, Peak_Pressure_hPa)
    
    Returns:
        Dictionary with 4 intensity features
    """
    features = {}
    
    # Get point at closest approach
    if closest_approach_idx >= 0 and closest_approach_idx < len(track_df):
        closest_point = track_df.iloc[closest_approach_idx]
        
        # Wind at closest approach
        features['wind_at_closest_approach_kt'] = get_best_wind_value(
            closest_point.get('TOKYO_WIND', ''),
            closest_point.get('USA_WIND', '')
        )
        
        # Pressure at closest approach
        features['pressure_at_closest_hpa'] = safe_float(
            closest_point.get('TOKYO_PRES', '')
        )
    else:
        features['wind_at_closest_approach_kt'] = MISSING_VALUE
        features['pressure_at_closest_hpa'] = MISSING_VALUE
    
    # FALLBACK: If wind at closest is missing, use max wind from summary
    if features['wind_at_closest_approach_kt'] == MISSING_VALUE and summary_data:
        if 'Peak_Windspeed_kmh' in summary_data and summary_data['Peak_Windspeed_kmh'] != MISSING_VALUE:
            features['wind_at_closest_approach_kt'] = summary_data['Peak_Windspeed_kmh'] * KMH_TO_KNOTS
    
    # FALLBACK: If pressure at closest is missing, use min pressure from summary
    if features['pressure_at_closest_hpa'] == MISSING_VALUE and summary_data:
        if 'Peak_Pressure_hPa' in summary_data and summary_data['Peak_Pressure_hPa'] != MISSING_VALUE:
            features['pressure_at_closest_hpa'] = summary_data['Peak_Pressure_hPa']
    
    # Maximum wind in track
    tokyo_winds = []
    usa_winds = []
    
    for _, row in track_df.iterrows():
        tokyo = safe_float(row.get('TOKYO_WIND', ''))
        usa = safe_float(row.get('USA_WIND', ''))
        
        if tokyo != MISSING_VALUE:
            tokyo_winds.append(tokyo)
        if usa != MISSING_VALUE:
            usa_winds.append(usa)
    
    if len(tokyo_winds) > 0:
        features['max_wind_in_track_kt'] = float(max(tokyo_winds))
    elif len(usa_winds) > 0:
        # Use USA wind but don't convert (we want the maximum)
        features['max_wind_in_track_kt'] = float(max(usa_winds))
    else:
        features['max_wind_in_track_kt'] = MISSING_VALUE
    
    # FALLBACK: Use summary data if track data has no winds
    if features['max_wind_in_track_kt'] == MISSING_VALUE and summary_data:
        if 'Peak_Windspeed_kmh' in summary_data and summary_data['Peak_Windspeed_kmh'] != MISSING_VALUE:
            features['max_wind_in_track_kt'] = summary_data['Peak_Windspeed_kmh'] * KMH_TO_KNOTS
    
    # Minimum pressure in track
    pressures = []
    for _, row in track_df.iterrows():
        pres = safe_float(row.get('TOKYO_PRES', ''))
        if pres != MISSING_VALUE:
            pressures.append(pres)
    
    if len(pressures) > 0:
        features['min_pressure_in_track_hpa'] = float(min(pressures))
    else:
        features['min_pressure_in_track_hpa'] = MISSING_VALUE
    
    # FALLBACK: Use summary data if track data has no pressures
    if features['min_pressure_in_track_hpa'] == MISSING_VALUE and summary_data:
        if 'Peak_Pressure_hPa' in summary_data and summary_data['Peak_Pressure_hPa'] != MISSING_VALUE:
            features['min_pressure_in_track_hpa'] = summary_data['Peak_Pressure_hPa']
    
    return features


# ============================================================================
# GROUP 3.2: INTENSITY EVOLUTION FEATURES (3 features)
# ============================================================================

def compute_intensity_evolution_features(
    track_df: pd.DataFrame,
    closest_approach_idx: int
) -> Dict[str, float]:
    """
    How storm intensity changed over time.
    
    Args:
        track_df: DataFrame with storm track data
        closest_approach_idx: Index of closest approach point
    
    Returns:
        Dictionary with 3 intensity evolution features
    """
    features = {
        'is_intensifying': MISSING_VALUE,
        'wind_change_approaching_kt': MISSING_VALUE,
        'intensification_rate_kt_per_day': MISSING_VALUE
    }
    
    # Collect winds with timestamps and position indices
    winds_with_info = []
    for position, (idx, row) in enumerate(track_df.iterrows()):
        wind = get_best_wind_value(
            row.get('TOKYO_WIND', ''),
            row.get('USA_WIND', '')
        )
        if wind != MISSING_VALUE:
            timestamp = pd.to_datetime(
                str(row['PH_DAY']) + ' ' + str(row['PH_TIME'])
            )
            winds_with_info.append({
                'position': position,  # Use position, not DataFrame index
                'wind': wind,
                'timestamp': timestamp
            })
    
    if len(winds_with_info) < 2:
        # Not enough data
        return features
    
    # Get winds before closest approach
    winds_before_closest = [
        w for w in winds_with_info 
        if w['position'] <= closest_approach_idx
    ]
    
    if len(winds_before_closest) >= 2:
        first_wind = winds_before_closest[0]
        last_wind = winds_before_closest[-1]
        
        # Is intensifying?
        wind_change = last_wind['wind'] - first_wind['wind']
        features['is_intensifying'] = 1.0 if wind_change > 0 else 0.0
        features['wind_change_approaching_kt'] = float(wind_change)
        
        # Intensification rate
        time_diff_seconds = (
            last_wind['timestamp'] - first_wind['timestamp']
        ).total_seconds()
        time_diff_days = time_diff_seconds / 86400
        
        if time_diff_days > 0:
            features['intensification_rate_kt_per_day'] = float(
                wind_change / time_diff_days
            )
        else:
            features['intensification_rate_kt_per_day'] = 0.0
    
    return features


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_intensity_features(
    storm_track_df: pd.DataFrame,
    province: str,
    distance_features_df: Optional[pd.DataFrame] = None,
    summary_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all 7 GROUP 3 storm intensity features.
    
    Args:
        storm_track_df: DataFrame with storm track data (all points)
        province: Province name
        distance_features_df: Optional DataFrame with distance features to find closest approach
        summary_data: Optional dict with summary data as fallback
    
    Returns:
        Dictionary with all 7 features
    """
    # Find closest approach index
    closest_approach_idx = len(storm_track_df) // 2  # Default to middle
    
    if distance_features_df is not None and len(distance_features_df) > 0:
        # Use distance features to find actual closest approach
        # This would require loading the raw distance data, not just features
        # For now, we'll estimate based on track progression
        pass
    
    # Initialize all features
    all_features = {}
    
    # 1. Intensity at Critical Moments (4 features) - with summary fallback
    all_features.update(
        compute_intensity_at_critical_moments(storm_track_df, closest_approach_idx, summary_data)
    )
    
    # 2. Intensity Evolution Features (3 features)
    all_features.update(
        compute_intensity_evolution_features(storm_track_df, closest_approach_idx)
    )
    
    return all_features


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_storm_for_all_provinces(
    storm_data_df: pd.DataFrame,
    year: int,
    storm_name: str,
    provinces: List[str],
    distance_features_df: Optional[pd.DataFrame] = None,
    summary_data: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Process a single storm and compute intensity features for all provinces.
    
    Args:
        storm_data_df: DataFrame with storm track data
        year: Year of the storm
        storm_name: Philippine name of the storm
        provinces: List of province names
        distance_features_df: Optional DataFrame with distance features
        summary_data: Optional dict with summary data as fallback
    
    Returns:
        DataFrame with one row per province containing all intensity features
    """
    results = []
    
    # For each province, compute features
    # Note: Storm intensity is the same regardless of province,
    # but we compute per province to maintain consistency with other groups
    for province in provinces:
        # Compute features
        features = compute_all_intensity_features(
            storm_data_df,
            province,
            distance_features_df,
            summary_data
        )
        
        # Add metadata
        features['Year'] = int(year)
        features['Storm'] = storm_name
        features['Province'] = province
        
        results.append(features)
    
    return pd.DataFrame(results)


def process_all_storm_intensity_data(
    storm_data_file: str = 'Storm_data/ph_storm_data.csv',
    output_file: str = 'Feature_Engineering_Data/group3/intensity_features_group3.csv',
    distance_features_file: Optional[str] = 'Feature_Engineering_Data/group1/distance_features_group1.csv',
    province_list_file: Optional[str] = 'Location_data/locations_latlng.csv',
    summary_file: Optional[str] = 'Storm_data/ph_storm_summary.csv'
) -> pd.DataFrame:
    """
    Process all storm data and generate GROUP 3 features.
    Uses summary file as fallback when detailed data is missing.
    
    Args:
        storm_data_file: Path to storm data CSV
        output_file: Output CSV filename
        distance_features_file: Path to distance features CSV (optional)
        province_list_file: Path to province list CSV (optional)
        summary_file: Path to storm summary CSV (optional, used as fallback)
    
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
    
    # Load summary data (fallback for missing values)
    summary_df = None
    if summary_file and Path(summary_file).exists():
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded {len(summary_df)} storms from summary file (fallback data)")
    
    # Load distance features if available (to get province list and possibly closest approach)
    distance_features_df = None
    provinces = None
    
    if distance_features_file and Path(distance_features_file).exists():
        print(f"Loading distance features from {distance_features_file}...")
        distance_features_df = pd.read_csv(distance_features_file)
        provinces = sorted(distance_features_df['Province'].unique())
        print(f"  Found {len(provinces)} provinces")
    
    # If no distance features, try to load province list
    if provinces is None and province_list_file and Path(province_list_file).exists():
        print(f"Loading province list from {province_list_file}...")
        province_df = pd.read_csv(province_list_file)
        provinces = sorted([p for p in province_df['Province'].unique() if pd.notna(p)])
        print(f"  Found {len(provinces)} provinces")
    
    if provinces is None:
        raise ValueError("Cannot determine province list. Provide distance_features_file or province_list_file")
    
    # Get unique storms
    storms = storm_data[['SEASON', 'PHNAME']].drop_duplicates()
    
    print(f"\nProcessing {len(storms)} storms...")
    print("=" * 60)
    
    all_results = []
    fallback_count = 0
    
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
            
            # Try to get summary data for this storm
            summary_data = None
            if summary_df is not None:
                summary_row = summary_df[
                    (summary_df['Year'] == year) &
                    (summary_df['PH_Name'] == storm_name)
                ]
                if len(summary_row) > 0:
                    summary_row = summary_row.iloc[0]
                    summary_data = {
                        'Peak_Windspeed_kmh': safe_float(summary_row.get('Peak_Windspeed_kmh', '')),
                        'Peak_Pressure_hPa': safe_float(summary_row.get('Peak_Pressure_hPa', ''))
                    }
                    # Check if we'll need fallback (no wind/pressure data in track)
                    has_wind = storm_track['TOKYO_WIND'].notna().any() or storm_track['USA_WIND'].notna().any()
                    has_pressure = storm_track['TOKYO_PRES'].notna().any()
                    if not has_wind or not has_pressure:
                        fallback_count += 1
            
            # Process this storm for all provinces
            storm_features = process_storm_for_all_provinces(
                storm_track,
                year,
                storm_name,
                provinces,
                distance_features_df,
                summary_data
            )
            
            all_results.append(storm_features)
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(storms)} storms...")
        
        except Exception as e:
            print(f"  Error processing {year}_{storm_name}: {str(e)}")
            continue
    
    print("=" * 60)
    if fallback_count > 0:
        print(f"\n[INFO] Used summary fallback data for {fallback_count} storms with missing detailed data")
    
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
        
        # Calculate data completeness
        print("\nData Completeness Analysis:")
        print("-" * 60)
        for col in feature_cols:
            missing_count = (final_df[col] == MISSING_VALUE).sum()
            missing_pct = (missing_count / len(final_df)) * 100
            print(f"  {col:40s}: {100-missing_pct:>5.1f}% complete")
        
        return final_df
    else:
        print("\n[ERROR] No data processed")
        return pd.DataFrame()


# ============================================================================
# DEPLOYMENT FUNCTION (for single storm)
# ============================================================================

def extract_intensity_features_for_deployment(
    track_data: pd.DataFrame,
    closest_approach_idx: Optional[int] = None,
    peak_windspeed_kmh: Optional[float] = None,
    peak_pressure_hpa: Optional[float] = None
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract all GROUP 3 features from storm track data.
    
    Args:
        track_data: DataFrame with columns [TOKYO_WIND, USA_WIND, TOKYO_PRES, PH_DAY, PH_TIME]
        closest_approach_idx: Index of closest approach (optional, defaults to middle)
        peak_windspeed_kmh: Optional fallback peak wind speed in km/h
        peak_pressure_hpa: Optional fallback peak pressure in hPa
    
    Returns:
        Dictionary with all 7 GROUP 3 features
    """
    if closest_approach_idx is None:
        closest_approach_idx = len(track_data) // 2
    
    # Prepare summary data if provided
    summary_data = None
    if peak_windspeed_kmh is not None or peak_pressure_hpa is not None:
        summary_data = {
            'Peak_Windspeed_kmh': peak_windspeed_kmh if peak_windspeed_kmh is not None else MISSING_VALUE,
            'Peak_Pressure_hPa': peak_pressure_hpa if peak_pressure_hpa is not None else MISSING_VALUE
        }
    
    # Compute all features
    features = {}
    features.update(compute_intensity_at_critical_moments(track_data, closest_approach_idx, summary_data))
    features.update(compute_intensity_evolution_features(track_data, closest_approach_idx))
    
    return features


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STORM INTENSITY FEATURE ENGINEERING PIPELINE")
    print("GROUP 3: 7 Features")
    print("WARNING: 40-70% missing data expected")
    print("WITH SUMMARY FALLBACK for missing data")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_storm_intensity_data(
        storm_data_file='Storm_data/ph_storm_data.csv',
        output_file='Feature_Engineering_Data/group3/intensity_features_group3.csv',
        distance_features_file='Feature_Engineering_Data/group1/distance_features_group1.csv',
        province_list_file='Location_data/locations_latlng.csv',
        summary_file='Storm_data/ph_storm_summary.csv'
    )
    
    if not features_df.empty:
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"\nTotal records: {len(features_df)}")
        print(f"Years covered: {features_df['Year'].min()} - {features_df['Year'].max()}")
        print(f"Unique storms: {features_df['Storm'].nunique()}")
        print(f"Provinces: {features_df['Province'].nunique()}")
        
        print("\nFeature statistics (excluding missing values):")
        feature_cols = [col for col in features_df.columns if col not in ['Year', 'Storm', 'Province']]
        
        for col in feature_cols:
            valid_data = features_df[features_df[col] != MISSING_VALUE][col]
            if len(valid_data) > 0:
                print(f"\n{col}:")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Min: {valid_data.min():.2f}")
                print(f"  Max: {valid_data.max():.2f}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

