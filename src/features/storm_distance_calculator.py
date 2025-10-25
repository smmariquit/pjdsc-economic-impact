"""
Storm Distance and Bearing Calculator
Calculates distance and bearing from provinces to storm locations for feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import math


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
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371.0
    
    return c * r


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    initial_bearing = math.atan2(x, y)
    
    # Convert to degrees and normalize to 0-360
    bearing = (math.degrees(initial_bearing) + 360) % 360
    
    return bearing


def bearing_to_direction(bearing: float) -> str:
    """
    Convert bearing (degrees) to compass direction.
    
    Args:
        bearing: Bearing in degrees (0-360)
    
    Returns:
        Compass direction (e.g., 'N', 'NNE', 'NE', etc.)
    """
    directions = [
        'N', 'NNE', 'NE', 'ENE', 
        'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW', 
        'W', 'WNW', 'NW', 'NNW'
    ]
    
    # Each direction covers 22.5 degrees (360 / 16)
    index = int((bearing + 11.25) / 22.5) % 16
    
    return directions[index]


def calculate_storm_distances(
    storm_track: pd.DataFrame,
    province_locations: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate distances and bearings from all provinces to each storm position.
    
    This is the main function for the feature engineering pipeline.
    
    Args:
        storm_track: DataFrame with columns ['LAT', 'LON', 'Timestamp'] 
                     (or ['LAT', 'LON', 'PH_DAY', 'PH_TIME'])
        province_locations: DataFrame with columns ['Province', 'Lat', 'Lng']
    
    Returns:
        DataFrame with columns ['Timestamp', 'Province', 'Distance_KM', 
                                'Bearing_Degrees', 'Direction']
    """
    results = []
    
    # Create timestamp column if not exists
    if 'Timestamp' not in storm_track.columns:
        if 'PH_DAY' in storm_track.columns and 'PH_TIME' in storm_track.columns:
            storm_track['Timestamp'] = pd.to_datetime(
                storm_track['PH_DAY'].astype(str) + ' ' + storm_track['PH_TIME'].astype(str)
            )
        else:
            raise ValueError("Storm track must have 'Timestamp' or 'PH_DAY' and 'PH_TIME' columns")
    
    # For each storm position
    for _, storm_row in storm_track.iterrows():
        storm_lat = storm_row['LAT']
        storm_lon = storm_row['LON']
        timestamp = storm_row['Timestamp']
        
        # Calculate distance and bearing to each province
        for _, prov_row in province_locations.iterrows():
            province = prov_row['Province']
            prov_lat = prov_row['Lat']
            prov_lng = prov_row['Lng']
            
            # Calculate distance from province to storm
            distance = haversine_distance(prov_lat, prov_lng, storm_lat, storm_lon)
            
            # Calculate bearing from province to storm
            bearing = calculate_bearing(prov_lat, prov_lng, storm_lat, storm_lon)
            
            # Convert bearing to compass direction
            direction = bearing_to_direction(bearing)
            
            results.append({
                'Timestamp': timestamp,
                'Province': province,
                'Distance_KM': round(distance, 2),
                'Bearing_Degrees': round(bearing, 2),
                'Direction': direction
            })
    
    return pd.DataFrame(results)


def process_single_storm(
    storm_data_path: str,
    province_locations_path: str,
    storm_name: str,
    year: int,
    output_dir: str = 'storm_location_data'
) -> pd.DataFrame:
    """
    Process a single storm and save the output.
    
    Args:
        storm_data_path: Path to the main storm data CSV
        province_locations_path: Path to the province locations CSV
        storm_name: Philippine name of the storm (e.g., 'Agaton')
        year: Year of the storm
        output_dir: Directory to save output files
    
    Returns:
        DataFrame with calculated distances and bearings
    """
    # Load data
    storm_data = pd.read_csv(storm_data_path)
    province_locations = pd.read_csv(province_locations_path)
    
    # Filter for specific storm
    storm_track = storm_data[
        (storm_data['PHNAME'] == storm_name) & 
        (storm_data['SEASON'] == year)
    ].copy()
    
    if len(storm_track) == 0:
        raise ValueError(f"No data found for storm {storm_name} in year {year}")
    
    # Calculate distances and bearings
    result_df = calculate_storm_distances(storm_track, province_locations)
    
    # Create output directory
    output_path = Path(output_dir) / str(year)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_path / f"{year}_{storm_name}.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(result_df)} records to {output_file}")
    
    return result_df


def process_all_storms(
    storm_data_path: str = 'Storm_data/ph_storm_data.csv',
    province_locations_path: str = 'Location_data/locations_latlng.csv',
    output_dir: str = 'storm_location_data'
) -> None:
    """
    Process all storms in the dataset and generate output files.
    
    Args:
        storm_data_path: Path to the main storm data CSV
        province_locations_path: Path to the province locations CSV
        output_dir: Directory to save output files
    """
    # Load data
    storm_data = pd.read_csv(storm_data_path)
    province_locations = pd.read_csv(province_locations_path)
    
    # Get unique storm/year combinations
    storms = storm_data[['SEASON', 'PHNAME']].drop_duplicates()
    
    print(f"Processing {len(storms)} storms...")
    
    for idx, (_, row) in enumerate(storms.iterrows(), 1):
        year = int(row['SEASON'])
        storm_name = row['PHNAME']
        
        try:
            process_single_storm(
                storm_data_path,
                province_locations_path,
                storm_name,
                year,
                output_dir
            )
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(storms)} storms...")
                
        except Exception as e:
            print(f"Error processing {year}_{storm_name}: {str(e)}")
            continue
    
    print(f"\nComplete! Processed {len(storms)} storms.")


def calculate_distances_for_deployment(
    storm_lat: float,
    storm_lon: float,
    province_coords: Dict[str, Tuple[float, float]]
) -> List[Dict]:
    """
    Streamlined function for deployment pipeline.
    Calculate distances and bearings for a single storm position.
    
    Args:
        storm_lat: Latitude of storm position
        storm_lon: Longitude of storm position
        province_coords: Dictionary mapping province names to (lat, lng) tuples
    
    Returns:
        List of dictionaries with distance and bearing information
    """
    results = []
    
    for province, (prov_lat, prov_lng) in province_coords.items():
        distance = haversine_distance(prov_lat, prov_lng, storm_lat, storm_lon)
        bearing = calculate_bearing(prov_lat, prov_lng, storm_lat, storm_lon)
        direction = bearing_to_direction(bearing)
        
        results.append({
            'Province': province,
            'Distance_KM': round(distance, 2),
            'Bearing_Degrees': round(bearing, 2),
            'Direction': direction
        })
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Process a single storm
    print("=" * 60)
    print("Example 1: Processing Agaton 2010")
    print("=" * 60)
    
    result = process_single_storm(
        storm_data_path='Storm_data/ph_storm_data.csv',
        province_locations_path='Location_data/locations_latlng.csv',
        storm_name='Agaton',
        year=2010,
        output_dir='storm_location_data'
    )
    
    print("\nSample output (first 10 rows):")
    print(result.head(10))
    
    # Example 2: Use the deployment function
    print("\n" + "=" * 60)
    print("Example 2: Deployment function (single storm position)")
    print("=" * 60)
    
    # Load province coordinates
    province_df = pd.read_csv('Location_data/locations_latlng.csv')
    province_coords = {
        row['Province']: (row['Lat'], row['Lng']) 
        for _, row in province_df.iterrows() if pd.notna(row['Province'])
    }
    
    # Example storm position
    storm_lat, storm_lon = 3.2, 153.3
    
    distances = calculate_distances_for_deployment(
        storm_lat=storm_lat,
        storm_lon=storm_lon,
        province_coords=province_coords
    )
    
    print(f"\nDistances for storm at ({storm_lat}, {storm_lon}):")
    for d in distances[:5]:
        print(f"  {d['Province']}: {d['Distance_KM']} km, "
              f"{d['Bearing_Degrees']}° ({d['Direction']})")
    
    # Uncomment to process all storms
    # print("\n" + "=" * 60)
    # print("Processing ALL storms (this may take a while)...")
    # print("=" * 60)
    # process_all_storms()

