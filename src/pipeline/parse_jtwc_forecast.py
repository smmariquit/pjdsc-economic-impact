"""
JTWC Forecast Parser

Parses Joint Typhoon Warning Center (JTWC) forecast bulletins
and extracts storm track data for model inference.

INPUT: storm_forecast.txt (JTWC format)
OUTPUT: Storm track DataFrame compatible with our pipeline
"""

import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def parse_latlon(lat_str: str, lon_str: str) -> Tuple[float, float]:
    """
    Parse latitude/longitude strings like '14.8N' and '119.7E'.
    
    Returns:
        (latitude, longitude) in decimal degrees
    """
    # Parse latitude
    lat_match = re.match(r'([\d.]+)([NS])', lat_str.strip())
    if not lat_match:
        raise ValueError(f"Invalid latitude: {lat_str}")
    
    lat_val = float(lat_match.group(1))
    lat_dir = lat_match.group(2)
    lat = lat_val if lat_dir == 'N' else -lat_val
    
    # Parse longitude
    lon_match = re.match(r'([\d.]+)([EW])', lon_str.strip())
    if not lon_match:
        raise ValueError(f"Invalid longitude: {lon_str}")
    
    lon_val = float(lon_match.group(1))
    lon_dir = lon_match.group(2)
    lon = lon_val if lon_dir == 'E' else -lon_val
    
    return lat, lon


from typing import Optional


def parse_timestamp(timestamp_str: str, year: Optional[int] = None) -> datetime:
    """
    Parse JTWC timestamp like '190600Z' (day 19, hour 06, minute 00, Zulu time).
    
    Args:
        timestamp_str: Format DDHHMM + Z (Z is optional)
        year: Year to use (defaults to current year)
    
    Returns:
        datetime object
    """
    match = re.match(r'(\d{2})(\d{2})(\d{2})Z?', timestamp_str.strip())
    if not match:
        raise ValueError(f"Invalid timestamp: {timestamp_str}")
    
    day = int(match.group(1))
    hour = int(match.group(2))
    minute = int(match.group(3))
    
    if year is None:
        year = datetime.now().year
    
    # Try current month first, fallback to next/previous if day invalid
    for month_offset in [0, 1, -1]:
        try:
            month = datetime.now().month + month_offset
            if month < 1:
                month = 12
                year -= 1
            elif month > 12:
                month = 1
                year += 1
            
            dt = datetime(year, month, day, hour, minute)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date for day {day}")


def parse_vector(vector_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse movement vector like '295 DEG/ 16 KTS'.
    
    Returns:
        (direction_degrees, speed_knots)
    """
    match = re.match(r'(\d+)\s*DEG/\s*(\d+)\s*KTS', vector_str.strip())
    if not match:
        return None, None
    
    direction = float(match.group(1))
    speed = float(match.group(2))
    
    return direction, speed


def parse_jtwc_forecast(forecast_input: str) -> pd.DataFrame:
    """
    Parse JTWC forecast bulletin into storm track DataFrame.
    
    Args:
        forecast_input: Either path to JTWC forecast text file OR bulletin text content
    
    Returns:
        DataFrame with columns:
            - LAT, LON: Position
            - PH_DAY, PH_TIME: Date/time
            - datetime: Parsed datetime
            - USA_WIND: Wind speed in knots
            - TOKYO_PRES: Pressure in hPa (if available)
            - STORM_SPEED, STORM_DIR: Movement vector
            - forecast_hour: 0 (current), 12, 24, 36, etc.
    """
    # Determine if input is file path or text content
    if '\n' in forecast_input or 'JTWC' in forecast_input.upper():
        # Looks like text content
        content = forecast_input
    else:
        # Treat as file path
        with open(forecast_input, 'r') as f:
            content = f.read()
    
    tracks = []
    
    # Extract year from timestamp or default to current
    # Look for year in specific contexts (avoid matching message IDs)
    year_match = re.search(r'(\d{2})OCT(\d{2})', content)  # Format like "19OCT25"
    if year_match:
        year = 2000 + int(year_match.group(2))  # 25 → 2025
    else:
        year = datetime.now().year
    
    # Extract storm name (robust to different JTWC subject formats)
    storm_name = "UNKNOWN"
    name_patterns = [
        r'\b(?:SUPER\s+TYPHOON|TYPHOON|TROPICAL\s+(?:STORM|DEPRESSION|CYCLONE))\s+\d{1,2}W\s*\(([^)]+)\)',
        r'\b\d{1,2}W\s*\(([^)]+)\)',
        r'SUBJ:.*?\(([^)]+)\)'
    ]
    for pat in name_patterns:
        m = re.search(pat, content, flags=re.IGNORECASE | re.DOTALL)
        if m:
            storm_name = m.group(1).strip().upper()
            break
    
    # Parse WARNING POSITION (current position)
    warning_match = re.search(
        r'WARNING POSITION:\s*(\d{6})Z.*?NEAR\s+([\d.]+[NS])\s+([\d.]+[EW]).*?'
        r'MAX SUSTAINED WINDS - (\d+) KT',
        content,
        re.DOTALL
    )
    
    if warning_match:
        timestamp = parse_timestamp(warning_match.group(1), year)
        lat, lon = parse_latlon(warning_match.group(2), warning_match.group(3))
        wind_kt = int(warning_match.group(4))
        
        tracks.append({
            'LAT': lat,
            'LON': lon,
            'datetime': timestamp,
            'USA_WIND': wind_kt,
            'forecast_hour': 0,
            'is_forecast': False
        })
    
    # Parse FORECASTS (12hr, 24hr, 36hr, etc.)
    forecast_pattern = re.compile(
        r'(\d+) HRS?, VALID AT:\s*'
        r'(\d{6})Z.*?'
        r'([\d.]+[NS])\s+([\d.]+[EW]).*?'
        r'MAX SUSTAINED WINDS - (\d+) KT',
        re.DOTALL
    )
    
    for match in forecast_pattern.finditer(content):
        forecast_hr = int(match.group(1))
        timestamp = parse_timestamp(match.group(2), year)
        lat, lon = parse_latlon(match.group(3), match.group(4))
        wind_kt = int(match.group(5))
        
        tracks.append({
            'LAT': lat,
            'LON': lon,
            'datetime': timestamp,
            'USA_WIND': wind_kt,
            'forecast_hour': forecast_hr,
            'is_forecast': True
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(tracks)
    # Ensure proper dtypes
    if 'LAT' in df.columns:
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    if 'LON' in df.columns:
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    if len(df) == 0:
        raise ValueError("No forecast positions found in file")
    
    # Sort by time
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Compute movement vectors from position changes
    df['STORM_SPEED'] = None
    df['STORM_DIR'] = None
    
    for i in range(1, len(df)):
        lat1 = float(df.loc[i-1, 'LAT']) if pd.notna(df.loc[i-1, 'LAT']) else 0.0
        lon1 = float(df.loc[i-1, 'LON']) if pd.notna(df.loc[i-1, 'LON']) else 0.0
        lat2 = float(df.loc[i, 'LAT']) if pd.notna(df.loc[i, 'LAT']) else lat1
        lon2 = float(df.loc[i, 'LON']) if pd.notna(df.loc[i, 'LON']) else lon1
        dt1 = df.loc[i-1, 'datetime']
        dt2 = df.loc[i, 'datetime']
        # Convert to python datetime if needed
        if hasattr(dt1, 'to_pydatetime'):
            dt1 = dt1.to_pydatetime()
        if hasattr(dt2, 'to_pydatetime'):
            dt2 = dt2.to_pydatetime()
        
        # Compute distance (km)
        from math import radians, sin, cos, sqrt, atan2, degrees
        
        R = 6371.0  # Earth radius in km
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_km = R * c
        
        # Compute time difference (hours)
        time_diff_hours = (dt2 - dt1).total_seconds() / 3600.0
        
        # Speed in knots (1 km/h = 0.539957 knots)
        speed_kmh = distance_km / time_diff_hours if time_diff_hours > 0 else 0
        speed_kt = speed_kmh * 0.539957
        
        # Direction (bearing)
        dlon = radians(lon2 - lon1)
        x = sin(dlon) * cos(lat2_rad)
        y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
        bearing = degrees(atan2(x, y))
        bearing = (bearing + 360) % 360
        
        df.loc[i, 'STORM_SPEED'] = speed_kt
        df.loc[i, 'STORM_DIR'] = bearing
    
    # Fill first row with second row values
    if len(df) > 1:
        df.loc[0, 'STORM_SPEED'] = df.loc[1, 'STORM_SPEED']
        df.loc[0, 'STORM_DIR'] = df.loc[1, 'STORM_DIR']
    
    # Extract central pressure from remarks if available
    pressure_match = re.search(r'MINIMUM CENTRAL PRESSURE.*?(\d{3,4})\s*MB', content)
    if pressure_match:
        central_pressure = int(pressure_match.group(1))
        df['TOKYO_PRES'] = central_pressure  # Assume constant for now
    else:
        df['TOKYO_PRES'] = None
    
    # Add metadata
    df['SEASON'] = year
    df['PHNAME'] = storm_name
    df['NAME'] = storm_name
    
    # Add date/time columns in expected format
    df['PH_DAY'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['PH_TIME'] = df['datetime'].dt.strftime('%H:%M:%S')
    
    # Reorder columns to match expected format
    columns_order = [
        'SEASON', 'PHNAME', 'NAME', 'LAT', 'LON', 'PH_DAY', 'PH_TIME',
        'datetime', 'STORM_SPEED', 'STORM_DIR', 'USA_WIND', 'TOKYO_PRES',
        'forecast_hour', 'is_forecast'
    ]
    
    df = df[columns_order]
    
    return df


def export_forecast_summary(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of the forecast."""
    summary = []
    summary.append("=" * 70)
    summary.append(f"FORECAST SUMMARY: {df['PHNAME'].iloc[0]} ({df['SEASON'].iloc[0]})")
    summary.append("=" * 70)
    summary.append("")
    
    current_rows = df[~df['is_forecast']]
    if len(current_rows) > 0:
        current = current_rows.iloc[0]
        summary.append("CURRENT POSITION:")
        summary.append(f"  Location: {current['LAT']:.1f}°N, {current['LON']:.1f}°E")
        summary.append(f"  Time: {current['datetime']}")
        summary.append(f"  Wind: {current['USA_WIND']:.0f} kt")
        summary.append(f"  Movement: {current['STORM_DIR']:.0f}° at {current['STORM_SPEED']:.0f} kt")
        summary.append("")
    
    summary.append("FORECAST TRACK:")
    summary.append(f"{'Hour':<8} {'Latitude':<10} {'Longitude':<10} {'Wind (kt)':<10} {'Time'}")
    summary.append("-" * 70)
    
    for _, row in df.iterrows():
        marker = "NOW" if not row['is_forecast'] else f"+{row['forecast_hour']}h"
        summary.append(
            f"{marker:<8} {row['LAT']:>8.1f}°N {row['LON']:>8.1f}°E "
            f"{row['USA_WIND']:>8.0f} kt  {row['datetime']}"
        )
    
    summary.append("=" * 70)
    
    return "\n".join(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse JTWC forecast bulletin")
    parser.add_argument("--input", type=str, default="storm_forecast.txt", help="Input forecast file")
    parser.add_argument("--output", type=str, help="Output CSV file (optional)")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    
    args = parser.parse_args()
    
    # Parse forecast
    print(f"\nParsing forecast from: {args.input}")
    df = parse_jtwc_forecast(args.input)
    
    print(f"✓ Extracted {len(df)} track points")
    print(f"✓ Storm: {df['PHNAME'].iloc[0]}")
    print(f"✓ Forecast period: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Print summary
    if args.summary or not args.output:
        print("\n" + export_forecast_summary(df))
    
    # Save to CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✓ Saved to: {args.output}")

