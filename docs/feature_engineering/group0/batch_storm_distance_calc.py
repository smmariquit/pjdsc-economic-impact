import pandas as pd
import numpy as np
import os
from pathlib import Path

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points.
    Returns bearing in degrees (0-360, where 0 is North).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    initial_bearing = np.arctan2(x, y)
    
    # Convert from radians to degrees
    initial_bearing = np.degrees(initial_bearing)
    
    # Normalize to 0-360
    bearing = (initial_bearing + 360) % 360
    
    return bearing

def bearing_to_cardinal(bearing):
    """
    Convert bearing in degrees to cardinal direction.
    """
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((bearing + 11.25) / 22.5) % 16
    return directions[idx]

def process_storm_data(storm_csv_path, locations_csv_path, output_base_dir='Feature_Engineering_Data/group0/storm_location_data'):
    """
    Process storm data and create CSV files for each storm with distance and direction
    calculations relative to each province.
    """
    # Read the CSV files
    print("Reading storm data...")
    storms_df = pd.read_csv(storm_csv_path)
    
    print("Reading locations data...")
    locations_df = pd.read_csv(locations_csv_path)
    
    # Create base output directory
    Path(output_base_dir).mkdir(exist_ok=True)
    
    # Group by storm (using SEASON and PHNAME)
    grouped = storms_df.groupby(['SEASON', 'PHNAME'])
    
    print(f"\nProcessing {len(grouped)} storms...")
    
    for (season, phname), storm_group in grouped:
        print(f"\nProcessing: {season} - {phname}")
        
        # Create year folder
        year_folder = Path(output_base_dir) / str(season)
        year_folder.mkdir(exist_ok=True)
        
        # Prepare output data
        output_rows = []
        
        # For each time point in the storm track
        for idx, storm_point in storm_group.iterrows():
            storm_lat = storm_point['LAT']
            storm_lon = storm_point['LON']
            timestamp = f"{storm_point['PH_DAY']} {storm_point['PH_TIME']}"
            
            # Calculate distance and bearing to each province
            for _, province in locations_df.iterrows():
                province_name = province['Province']
                province_lat = province['Lat']
                province_lng = province['Lng']
                
                # Calculate distance
                distance_km = haversine_distance(
                    storm_lat, storm_lon,
                    province_lat, province_lng
                )
                
                # Calculate bearing from province to storm
                bearing = calculate_bearing(
                    province_lat, province_lng,
                    storm_lat, storm_lon
                )
                
                # Convert to cardinal direction
                direction = bearing_to_cardinal(bearing)
                
                # Create output row
                output_rows.append({
                    'Timestamp': timestamp,
                    'Province': province_name,
                    'Distance_KM': round(distance_km, 2),
                    'Bearing_Degrees': round(bearing, 2),
                    'Direction': direction
                })
        
        # Create DataFrame and save to CSV
        output_df = pd.DataFrame(output_rows)
        
        # Sort by province first, then timestamp (so each province's data evolves over time)
        output_df = output_df.sort_values(['Province', 'Timestamp'])
        
        # Save to CSV
        output_filename = f"{season}_{phname}.csv"
        output_path = year_folder / output_filename
        output_df.to_csv(output_path, index=False)
        
        print(f"  [OK] Created: {output_path} ({len(storm_group)} time points x {len(locations_df)} provinces = {len(output_rows)} rows)")
    
    print(f"\n[OK] Processing complete! All files saved to '{output_base_dir}/' directory")
    print(f"  Total storms processed: {len(grouped)}")

if __name__ == "__main__":
    # File paths (from project root)
    storm_data_path = "Storm_data/ph_storm_data.csv"
    locations_data_path = "Location_data/locations_latlng.csv"
    output_dir = "Feature_Engineering_Data/group0/storm_location_data"
    
    # Process the data
    process_storm_data(storm_data_path, locations_data_path, output_dir)