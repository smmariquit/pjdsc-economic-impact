"""
Real-Time Weather Forecast Fetcher

Fetches weather forecast data from Open-Meteo API for a storm forecast period.
This provides GROUP 2 (weather exposure) features for real-time predictions.

USAGE:
    python fetch_forecast_weather.py --start 2025-10-19 --end 2025-10-24 --output forecast_weather.csv

DEPENDENCIES:
    pip install openmeteo-requests requests-cache retry-requests pandas
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def fetch_weather_forecast(
    start_date: str,
    end_date: str,
    locations_file: str = "Location_data/locations_latlng.csv",
    output_file: str = None,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo API for all provinces.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        locations_file: Path to province coordinates CSV
        output_file: Optional output CSV path
        batch_size: Number of locations to process per API call
    
    Returns:
        DataFrame with weather forecast for all provinces
    """
    print("\n" + "="*70)
    print("WEATHER FORECAST FETCH (Open-Meteo API)")
    print("="*70)
    
    # Setup Open-Meteo API client with cache and retry
    print("\n1. Setting up API client...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # API endpoint and variables
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Weather variables (must match training data!)
    daily_variables = [
        "precipitation_sum",          # Total daily precipitation (mm)
        "wind_speed_10m_max",          # Max wind speed at 10m (km/h)
        "wind_gusts_10m_max",          # Max wind gusts at 10m (km/h)
        "precipitation_hours",         # Hours with precipitation
    ]
    
    print(f"   Variables: {', '.join(daily_variables)}")
    
    # Load province locations
    print(f"\n2. Loading province locations from: {locations_file}")
    locations_df = pd.read_csv(locations_file)
    locations_df.columns = ['Province', 'Lat', 'Lng']  # Standardize column names
    locations_df = locations_df.dropna(subset=["Lat", "Lng"])
    print(f"   ✓ Loaded {len(locations_df)} provinces")
    
    # Validate dates
    print(f"\n3. Validating forecast period...")
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days + 1
        print(f"   Period: {start_date} to {end_date} ({days} days)")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    # Fetch forecast data in batches
    print(f"\n4. Fetching forecast data (batch size: {batch_size})...")
    
    total_locations = len(locations_df)
    all_data = []
    
    for batch_start in range(0, total_locations, batch_size):
        batch_end = min(batch_start + batch_size, total_locations)
        batch_locations = locations_df.iloc[batch_start:batch_end]
        
        latitudes = batch_locations["Lat"].astype(float).tolist()
        longitudes = batch_locations["Lng"].astype(float).tolist()
        
        print(f"   Batch {batch_start+1}-{batch_end} of {total_locations}...", end=" ")
        
        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "daily": daily_variables,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Manila"  # Philippine timezone
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            
            # Process each location in the batch
            for i, response in enumerate(responses):
                province = batch_locations.iloc[i]['Province']
                
                # Extract daily data
                daily = response.Daily()
                
                # Create date range
                dates = pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )
                
                # Extract variable data
                daily_data = {
                    var: daily.Variables(v_idx).ValuesAsNumpy() 
                    for v_idx, var in enumerate(daily_variables)
                }
                
                # Create rows for each date
                for date_idx, date in enumerate(dates):
                    row_data = {
                        'date': date,
                        'province': province,
                        'latitude': round(response.Latitude(), 6),
                        'longitude': round(response.Longitude(), 6),
                    }
                    
                    # Add weather variables
                    for var_name, var_values in daily_data.items():
                        row_data[var_name] = var_values[date_idx]
                    
                    all_data.append(row_data)
            
            print(f"✓ {len(dates)} days × {len(responses)} locations")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Failed at batch {batch_start+1}-{batch_end}")
            raise
    
    # Convert to DataFrame
    print(f"\n5. Creating DataFrame...")
    df = pd.DataFrame(all_data)
    
    # Sort by date and province
    df = df.sort_values(by=['date', 'province']).reset_index(drop=True)
    
    print(f"   ✓ Total records: {len(df):,}")
    print(f"   ✓ Provinces: {df['province'].nunique()}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("FORECAST FETCH COMPLETE")
    print("="*70)
    
    return df


def fetch_from_jtwc_forecast(forecast_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Convenience function: Extract dates from JTWC forecast and fetch weather.
    
    Args:
        forecast_file: Path to JTWC forecast bulletin
        output_file: Optional output CSV path
    
    Returns:
        DataFrame with weather forecast
    """
    from parse_jtwc_forecast import parse_jtwc_forecast
    
    # Parse forecast to get date range
    print(f"Parsing JTWC forecast: {forecast_file}")
    track = parse_jtwc_forecast(forecast_file)
    
    start_date = track['datetime'].min().strftime('%Y-%m-%d')
    end_date = track['datetime'].max().strftime('%Y-%m-%d')
    
    print(f"Forecast period: {start_date} to {end_date}")
    
    # Fetch weather forecast
    return fetch_weather_forecast(start_date, end_date, output_file=output_file)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch weather forecast from Open-Meteo API"
    )
    
    # Option 1: Specify dates directly
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    # Option 2: Extract from JTWC forecast
    parser.add_argument("--from-jtwc", type=str, help="JTWC forecast file to extract dates")
    
    # Common options
    parser.add_argument("--locations", type=str, default="Location_data/locations_latlng.csv",
                       help="Province locations CSV")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of locations per API call")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_jtwc:
        # Extract dates from JTWC forecast
        df = fetch_from_jtwc_forecast(args.from_jtwc, args.output)
    elif args.start and args.end:
        # Use specified dates
        df = fetch_weather_forecast(
            start_date=args.start,
            end_date=args.end,
            locations_file=args.locations,
            output_file=args.output,
            batch_size=args.batch_size
        )
    else:
        parser.error("Must specify either --start/--end OR --from-jtwc")
    
    # Display sample
    print("\n📊 Sample data (first 5 rows):")
    print(df.head())
    print("\n📈 Summary statistics:")
    print(df[['wind_speed_10m_max', 'wind_gusts_10m_max', 'precipitation_sum']].describe())

