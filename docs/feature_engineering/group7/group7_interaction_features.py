"""
Interaction Feature Engineering Pipeline
Combines existing feature groups to generate GROUP 7 features for ML pipeline.

GROUP 7: INTERACTION FEATURES (6 features total)
- Distance × Intensity Interactions (3 features)
- Distance × Weather Interactions (3 features)

Requires: GROUP 1 (Distance), GROUP 2 (Weather), GROUP 3 (Intensity - optional)
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
# GROUP 7.1: DISTANCE × INTENSITY INTERACTIONS (3 features)
# ============================================================================

def compute_distance_intensity_interactions(
    distance_features: Dict[str, float],
    weather_features: Dict[str, float],
    intensity_features: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Create interaction terms between distance and intensity.
    
    Uses weather-based intensity (max_wind_gust) as primary source,
    with optional storm intensity (JTWC) as supplement.
    
    Args:
        distance_features: Dict with distance features from GROUP 1
        weather_features: Dict with weather features from GROUP 2
        intensity_features: Optional dict with intensity features from GROUP 3
    
    Returns:
        Dictionary with 3 distance-intensity interaction features
    """
    min_dist = distance_features.get('min_distance_km', 500.0)
    max_wind = weather_features.get('max_wind_gust_kmh', 50.0)
    
    # Avoid division by zero
    proximity = 1 / (min_dist**2 + 1)
    
    features = {
        'proximity_intensity_product': float(proximity * max_wind),
        'min_distance_x_max_wind': float(min_dist * max_wind),
        'intensity_per_km': float(max_wind / (min_dist + 1))
    }
    
    return features


# ============================================================================
# GROUP 7.2: DISTANCE × WEATHER INTERACTIONS (3 features)
# ============================================================================

def compute_distance_weather_interactions(
    distance_features: Dict[str, float],
    weather_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Create interaction terms between distance and weather patterns.
    
    Note: Simplified version using aggregated features.
    Full version would use daily distance-weather alignment.
    
    Args:
        distance_features: Dict with distance features from GROUP 1
        weather_features: Dict with weather features from GROUP 2
    
    Returns:
        Dictionary with 3 distance-weather interaction features
    """
    min_dist = distance_features.get('min_distance_km', 500.0)
    total_rain = weather_features.get('total_precipitation_mm', 50.0)
    
    # Estimate close approach rainfall
    # Storms closer than 200km contribute more directly
    # This is a simplified proxy without daily alignment
    if min_dist < 200:
        # Very close - most rain is direct
        close_approach_rainfall = total_rain * 0.8
        distant_rainfall = total_rain * 0.2
    elif min_dist < 500:
        # Moderate distance - mixed
        close_approach_rainfall = total_rain * 0.5
        distant_rainfall = total_rain * 0.5
    else:
        # Distant - mostly outer bands
        close_approach_rainfall = total_rain * 0.2
        distant_rainfall = total_rain * 0.8
    
    # Rainfall concentration ratio
    rainfall_ratio = close_approach_rainfall / total_rain if total_rain > 0 else 0
    
    return {
        'close_approach_rainfall': float(close_approach_rainfall),
        'distant_rainfall': float(distant_rainfall),
        'rainfall_distance_ratio': float(rainfall_ratio)
    }


# ============================================================================
# MAIN FEATURE AGGREGATION FUNCTION
# ============================================================================

def compute_all_interaction_features(
    distance_features: Dict[str, float],
    weather_features: Dict[str, float],
    intensity_features: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute all 6 GROUP 7 interaction features.
    
    Args:
        distance_features: Dict with GROUP 1 features
        weather_features: Dict with GROUP 2 features
        intensity_features: Optional dict with GROUP 3 features
    
    Returns:
        Dictionary with all 6 interaction features
    """
    all_features = {}
    
    # 1. Distance × Intensity Interactions (3 features)
    all_features.update(compute_distance_intensity_interactions(
        distance_features, weather_features, intensity_features
    ))
    
    # 2. Distance × Weather Interactions (3 features)
    all_features.update(compute_distance_weather_interactions(
        distance_features, weather_features
    ))
    
    return all_features


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_all_interaction_data(
    distance_features_file: str = 'Feature_Engineering_Data/group1/distance_features_group1.csv',
    weather_features_file: str = 'Feature_Engineering_Data/group2/weather_features_group2.csv',
    intensity_features_file: Optional[str] = 'Feature_Engineering_Data/group3/intensity_features_group3.csv',
    output_file: str = 'Feature_Engineering_Data/group7/interaction_features_group7.csv'
) -> pd.DataFrame:
    """
    Process all storm-province pairs and generate GROUP 7 interaction features.
    
    Args:
        distance_features_file: Path to GROUP 1 features CSV
        weather_features_file: Path to GROUP 2 features CSV
        intensity_features_file: Path to GROUP 3 features CSV (optional)
        output_file: Output CSV filename
    
    Returns:
        DataFrame with all interaction features
    """
    print("=" * 60)
    print("Loading feature groups...")
    print("=" * 60)
    
    # Load GROUP 1: Distance features
    if not Path(distance_features_file).exists():
        raise FileNotFoundError(f"Distance features not found: {distance_features_file}")
    
    distance_df = pd.read_csv(distance_features_file)
    print(f"Loaded GROUP 1 (Distance): {len(distance_df)} records, {len(distance_df.columns)-3} features")
    
    # Load GROUP 2: Weather features
    if not Path(weather_features_file).exists():
        raise FileNotFoundError(f"Weather features not found: {weather_features_file}")
    
    weather_df = pd.read_csv(weather_features_file)
    print(f"Loaded GROUP 2 (Weather): {len(weather_df)} records, {len(weather_df.columns)-3} features")
    
    # Load GROUP 3: Intensity features (optional)
    intensity_df = None
    if intensity_features_file and Path(intensity_features_file).exists():
        intensity_df = pd.read_csv(intensity_features_file)
        print(f"Loaded GROUP 3 (Intensity): {len(intensity_df)} records, {len(intensity_df.columns)-3} features")
    else:
        print("GROUP 3 (Intensity) not available - using weather intensity as proxy")
    
    # Merge distance and weather features
    print("\nMerging feature groups...")
    combined_df = distance_df.merge(
        weather_df,
        on=['Year', 'Storm', 'Province'],
        how='inner',
        suffixes=('_dist', '_weather')
    )
    
    # Merge intensity if available
    if intensity_df is not None:
        combined_df = combined_df.merge(
            intensity_df,
            on=['Year', 'Storm', 'Province'],
            how='left',
            suffixes=('', '_intensity')
        )
    
    print(f"Combined dataset: {len(combined_df)} records")
    
    # Generate interaction features
    print("\nGenerating interaction features...")
    print("=" * 60)
    
    results = []
    
    for idx, row in combined_df.iterrows():
        # Extract distance features
        distance_feats = {
            col: row[col] for col in distance_df.columns 
            if col not in ['Year', 'Storm', 'Province']
        }
        
        # Extract weather features
        weather_feats = {
            col.replace('_weather', ''): row[col] 
            for col in combined_df.columns 
            if col.endswith('_weather')
        }
        # Add non-suffixed weather features
        for col in weather_df.columns:
            if col not in ['Year', 'Storm', 'Province'] and col in row:
                weather_feats[col] = row[col]
        
        # Extract intensity features if available
        intensity_feats = None
        if intensity_df is not None:
            intensity_feats = {
                col: row.get(col, MISSING_VALUE) 
                for col in intensity_df.columns 
                if col not in ['Year', 'Storm', 'Province']
            }
        
        # Compute interaction features
        interaction_feats = compute_all_interaction_features(
            distance_feats, weather_feats, intensity_feats
        )
        
        # Add metadata
        interaction_feats['Year'] = row['Year']
        interaction_feats['Storm'] = row['Storm']
        interaction_feats['Province'] = row['Province']
        
        results.append(interaction_feats)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(combined_df)} records...")
    
    print("=" * 60)
    
    # Create DataFrame
    final_df = pd.DataFrame(results)
    
    # Reorder columns
    metadata_cols = ['Year', 'Storm', 'Province']
    feature_cols = [col for col in final_df.columns if col not in metadata_cols]
    final_df = final_df[metadata_cols + sorted(feature_cols)]
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved {len(final_df)} records to {output_file}")
    print(f"[OK] Generated {len(feature_cols)} interaction features per record")
    
    return final_df


# ============================================================================
# DEPLOYMENT FUNCTION
# ============================================================================

def extract_interaction_features_for_deployment(
    distance_features: Dict[str, float],
    weather_features: Dict[str, float],
    intensity_features: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Streamlined function for deployment pipeline.
    Extract all GROUP 7 features from existing feature groups.
    
    Args:
        distance_features: Dictionary with GROUP 1 features
        weather_features: Dictionary with GROUP 2 features
        intensity_features: Optional dictionary with GROUP 3 features
    
    Returns:
        Dictionary with all 6 GROUP 7 interaction features
    """
    return compute_all_interaction_features(
        distance_features, weather_features, intensity_features
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INTERACTION FEATURE ENGINEERING PIPELINE")
    print("GROUP 7: 6 Features")
    print("=" * 60)
    print()
    
    # Process all data
    features_df = process_all_interaction_data(
        distance_features_file='Feature_Engineering_Data/group1/distance_features_group1.csv',
        weather_features_file='Feature_Engineering_Data/group2/weather_features_group2.csv',
        intensity_features_file='Feature_Engineering_Data/group3/intensity_features_group3.csv',
        output_file='Feature_Engineering_Data/group7/interaction_features_group7.csv'
    )
    
    if not features_df.empty:
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"\nTotal records: {len(features_df)}")
        print(f"Years covered: {features_df['Year'].min()} - {features_df['Year'].max()}")
        print(f"Unique storms: {features_df['Storm'].nunique()}")
        print(f"Provinces: {features_df['Province'].nunique()}")
        
        print("\nInteraction Feature Statistics:")
        feature_cols = [col for col in features_df.columns if col not in ['Year', 'Storm', 'Province']]
        print(features_df[feature_cols].describe())
        
        print("\n" + "=" * 60)
        print("Sample Records (first 5):")
        print("=" * 60)
        print(features_df.head(5).to_string())
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


