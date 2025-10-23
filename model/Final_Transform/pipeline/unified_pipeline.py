"""
UNIFIED FEATURE ENGINEERING PIPELINE
Integrates existing group scripts into single end-to-end pipeline.

RAW INPUT → FEATURE EXTRACTION → ML-READY OUTPUT

INPUT:
  - Storm track: Storm_data/ph_storm_data.csv
  - Weather: Weather_location_data/{year}/{year}_{storm}.csv
  - Population: Population_data/population_density_all_years.csv
  - Locations: Location_data/locations_latlng.csv
  - Historical storms: For group8 (past 90 days)

OUTPUT:
  - Feature matrix: [Year, Storm, Province] × 75+ features
"""

import sys
from pathlib import Path

# Add Feature_Engineering to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Feature_Engineering"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 in degrees."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


class StormFeaturePipeline:
    """
    End-to-end pipeline that calls existing feature engineering scripts.
    """
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        
        # Data directories
        self.storm_data_dir = self.base_dir / "Storm_data"
        self.weather_dir = self.base_dir / "Weather_location_data"
        self.pop_dir = self.base_dir / "Population_data"
        self.location_file = self.base_dir / "Location_data" / "locations_latlng.csv"
        
        # Load static data once
        self.locations = self._load_locations()
        self.population = self._load_population()
        
    def _load_locations(self) -> pd.DataFrame:
        """Load province centroids."""
        locs = pd.read_csv(self.location_file)
        locs.columns = ['Province', 'latitude', 'longitude']
        return locs
    
    def _load_population(self) -> pd.DataFrame:
        """Load all years population data."""
        return pd.read_csv(self.pop_dir / "population_density_all_years.csv")
    
    def _load_storm_track(self, year: int, storm_name: str) -> pd.DataFrame:
        """Extract single storm from master track file."""
        all_tracks = pd.read_csv(self.storm_data_dir / "ph_storm_data.csv")
        
        track = all_tracks[
            (all_tracks['SEASON'] == year) & 
            (all_tracks['PHNAME'] == storm_name)
        ].copy()
        
        if len(track) == 0:
            raise ValueError(f"Storm not found: {year} {storm_name}")
        
        # Add datetime column
        track['datetime'] = pd.to_datetime(
            track['PH_DAY'].astype(str) + ' ' + track['PH_TIME'].astype(str)
        )
        
        return track.sort_values('datetime').reset_index(drop=True)
    
    def _load_weather(self, year: int, storm_name: str) -> pd.DataFrame:
        """Load weather data for storm."""
        weather_file = self.weather_dir / str(year) / f"{year}_{storm_name}.csv"
        
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_file}")
        
        weather = pd.read_csv(weather_file)
        weather['date'] = pd.to_datetime(weather['date'])
        return weather
    
    def _get_historical_context(
        self,
        current_storm_start: datetime,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get historical storms for group8 features.
        This is STATIC context - storms that ended before current storm.
        """
        all_tracks = pd.read_csv(self.storm_data_dir / "ph_storm_data.csv")
        all_tracks['datetime'] = pd.to_datetime(
            all_tracks['PH_DAY'].astype(str) + ' ' + all_tracks['PH_TIME'].astype(str)
        )
        
        cutoff = current_storm_start - timedelta(days=lookback_days)
        historical = all_tracks[
            (all_tracks['datetime'] < current_storm_start) &
            (all_tracks['datetime'] >= cutoff)
        ].copy()
        
        return historical
    
    def extract_features_group1(
        self,
        track: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 1: Distance/proximity features (26 features).
        
        Uses existing: Feature_Engineering/group1/group1_distance_feature_engineering.py
        """
        from group1.group1_distance_feature_engineering import extract_features_for_deployment
        
        features_list = []
        for _, prov in self.locations.iterrows():
            # Compute distances and bearings for each track point
            distances = []
            bearings = []
            timestamps = []
            
            for _, track_point in track.iterrows():
                dist = haversine_distance(
                    prov['latitude'], prov['longitude'],
                    track_point['LAT'], track_point['LON']
                )
                bearing = calculate_bearing(
                    prov['latitude'], prov['longitude'],
                    track_point['LAT'], track_point['LON']
                )
                distances.append(dist)
                bearings.append(bearing)
                timestamps.append(str(track_point['datetime']))
            
            # Call deployment function
            features = extract_features_for_deployment(
                distances=distances,
                bearings=bearings,
                timestamps=timestamps,
                time_interval_hours=3.0
            )
            features['Year'] = year
            features['Storm'] = storm_name
            features['Province'] = prov['Province']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_group2(
        self,
        weather: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 2: Weather exposure features (20 features).
        
        Uses existing: Feature_Engineering/group2/group2_weather_feature_engineering.py
        """
        from group2.group2_weather_feature_engineering import extract_weather_features_for_deployment
        
        features_list = []
        for province in weather['province'].unique():
            prov_weather = weather[weather['province'] == province].copy().sort_values('date')
            
            # Extract lists
            dates = prov_weather['date'].astype(str).tolist()
            wind_gusts = prov_weather['wind_gusts_10m_max'].tolist()
            wind_speeds = prov_weather['wind_speed_10m_max'].tolist()
            precipitation_sums = prov_weather['precipitation_sum'].tolist()
            precipitation_hours = prov_weather['precipitation_hours'].tolist()
            
            features = extract_weather_features_for_deployment(
                dates=dates,
                wind_gusts=wind_gusts,
                wind_speeds=wind_speeds,
                precipitation_sums=precipitation_sums,
                precipitation_hours=precipitation_hours,
                closest_approach_date=None  # Can be computed from group1 if needed
            )
            features['Year'] = year
            features['Storm'] = storm_name
            features['Province'] = province
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_group3(
        self,
        track: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 3: Storm intensity features (7 features).
        
        Uses existing: Feature_Engineering/group3/group3_storm_intensity_features.py
        """
        from group3.group3_storm_intensity_features import extract_intensity_features_for_deployment
        
        # Intensity features are storm-level (same for all provinces)
        # Just need closest_approach_idx which varies per province, but for now use storm-level
        
        features_dict = extract_intensity_features_for_deployment(
            track_data=track,
            closest_approach_idx=None  # Will default to middle of track
        )
        
        # Broadcast to all provinces
        features_list = []
        for _, prov in self.locations.iterrows():
            feat = features_dict.copy()
            feat['Year'] = year
            feat['Storm'] = storm_name
            feat['Province'] = prov['Province']
            features_list.append(feat)
        
        return pd.DataFrame(features_list)
    
    def extract_features_group6(
        self,
        track: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 6: Storm motion features (10 features).
        
        Uses existing: Feature_Engineering/group6/group6_storm_motion_features.py
        """
        from group6.group6_storm_motion_features import extract_motion_features_for_deployment
        
        # Motion features need province location (for approach timing)
        features_list = []
        for _, prov in self.locations.iterrows():
            features = extract_motion_features_for_deployment(
                track_data=track,
                province_lat=prov['latitude'],
                province_lon=prov['longitude']
            )
            features['Year'] = year
            features['Storm'] = storm_name
            features['Province'] = prov['Province']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_group7(
        self,
        group1: pd.DataFrame,
        group2: pd.DataFrame,
        group3: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 7: Interaction features (6 features).
        
        Uses existing: Feature_Engineering/group7/group7_interaction_features.py
        """
        from group7.group7_interaction_features import extract_interaction_features_for_deployment
        
        # Merge all groups
        merged = group1.merge(group2, on=['Year', 'Storm', 'Province'], how='inner')
        merged = merged.merge(group3, on=['Year', 'Storm', 'Province'], how='inner')
        
        # Extract features row by row
        features_list = []
        for _, row in merged.iterrows():
            row_dict = row.to_dict()
            
            features = extract_interaction_features_for_deployment(
                distance_features=row_dict,
                weather_features=row_dict,
                intensity_features=row_dict
            )
            features['Year'] = year
            features['Storm'] = storm_name
            features['Province'] = row['Province']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_group8(
        self,
        track: pd.DataFrame,
        historical: pd.DataFrame,
        year: int,
        storm_name: str
    ) -> pd.DataFrame:
        """
        GROUP 8: Multi-storm features (6 features).
        
        Uses existing: Feature_Engineering/group8/group8_multistorm_features.py
        
        NOTE: This requires HISTORICAL CONTEXT (past storms).
        """
        from group8.group8_multistorm_features import extract_multistorm_features_for_deployment
        
        # Get current storm dates
        current_date = track['datetime'].iloc[len(track)//2]  # Use middle of storm
        current_location = (track['LAT'].iloc[len(track)//2], track['LON'].iloc[len(track)//2])
        
        # Find concurrent storms (active at same time)
        storm_start = track['datetime'].min()
        storm_end = track['datetime'].max()
        
        concurrent_storms = historical[
            (historical['datetime'] >= storm_start) & 
            (historical['datetime'] <= storm_end)
        ]
        # Group by storm ID to get unique concurrent storms
        active_storms = []
        for (season, name), group in concurrent_storms.groupby(['SEASON', 'PHNAME']):
            if season == year and name == storm_name:
                continue  # Skip current storm
            active_storms.append({
                'id': f"{season}_{name}",
                'lat': group['LAT'].mean(),
                'lon': group['LON'].mean(),
                'intensity_kmh': group.get('USA_WIND', group.get('TOKYO_WIND', 0)).max() * 1.852
            })
        
        # Find recent past storms
        recent_storms = historical[historical['datetime'] < storm_start]
        recent_history = []
        for (season, name), group in recent_storms.groupby(['SEASON', 'PHNAME']):
            recent_history.append({
                'end_date': group['datetime'].max()
            })
        
        # Compute features (same for all provinces)
        features_dict = extract_multistorm_features_for_deployment(
            current_storm_date=current_date,
            current_storm_location=current_location,
            active_storms=active_storms,
            recent_storms_history=recent_history
        )
        
        # Broadcast to all provinces
        features_list = []
        for _, prov in self.locations.iterrows():
            feat = features_dict.copy()
            feat['Year'] = year
            feat['Storm'] = storm_name
            feat['Province'] = prov['Province']
            features_list.append(feat)
        
        return pd.DataFrame(features_list)
    
    def process_storm(
        self,
        year: int,
        storm_name: str,
        include_group8: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        MAIN PIPELINE: Process one storm end-to-end.
        
        Args:
            year: Storm year
            storm_name: Philippine storm name
            include_group8: Whether to compute multi-storm features (requires history)
            verbose: Print progress
        
        Returns:
            DataFrame with all features for all provinces
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {year} {storm_name}")
            print(f"{'='*70}\n")
        
        # Load raw data
        if verbose:
            print("1. Loading raw data...")
        track = self._load_storm_track(year, storm_name)
        weather = self._load_weather(year, storm_name)
        
        if verbose:
            print(f"   Track: {len(track)} points from {track['datetime'].min()} to {track['datetime'].max()}")
            print(f"   Weather: {len(weather)} records for {weather['province'].nunique()} provinces")
        
        # Extract features by group
        if verbose:
            print("\n2. Extracting features...")
        
        print("   GROUP 1: Distance features...")
        group1 = self.extract_features_group1(track, year, storm_name)
        
        print("   GROUP 2: Weather features...")
        group2 = self.extract_features_group2(weather, year, storm_name)
        
        print("   GROUP 3: Intensity features...")
        group3 = self.extract_features_group3(track, year, storm_name)
        
        print("   GROUP 6: Motion features...")
        group6 = self.extract_features_group6(track, year, storm_name)
        
        print("   GROUP 7: Interaction features...")
        group7 = self.extract_features_group7(group1, group2, group3, year, storm_name)
        
        if include_group8:
            print("   GROUP 8: Multi-storm features (with historical context)...")
            historical = self._get_historical_context(track['datetime'].min())
            group8 = self.extract_features_group8(track, historical, year, storm_name)
            print(f"      Found {historical.groupby(['SEASON', 'PHNAME']).ngroups} historical storms")
        else:
            print("   GROUP 8: Skipped (no historical context)")
            group8 = pd.DataFrame({'Year': year, 'Storm': storm_name, 'Province': self.locations['Province']})
        
        # Merge all groups
        if verbose:
            print("\n3. Merging feature groups...")
        
        features = group1.copy()
        for i, group_df in enumerate([group2, group3, group6, group7, group8], start=2):
            features = features.merge(
                group_df,
                on=['Year', 'Storm', 'Province'],
                how='outer'
            )
        
        # Add population
        if verbose:
            print("4. Adding population data...")
        
        pop_year = self.population[self.population['Year'] == year]
        features = features.merge(
            pop_year[['Province', 'Population', 'PopulationDensity']],
            on='Province',
            how='left'
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ COMPLETE: {len(features)} provinces × {len(features.columns)} features")
            print(f"{'='*70}\n")
        
        return features


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract all features for a single storm"
    )
    parser.add_argument("--year", type=int, required=True, help="e.g., 2024")
    parser.add_argument("--storm", type=str, required=True, help="e.g., Kristine")
    parser.add_argument("--no-group8", action="store_true", help="Skip multi-storm features")
    parser.add_argument("--output", type=str, help="Output CSV file")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = StormFeaturePipeline()
    
    features = pipeline.process_storm(
        year=args.year,
        storm_name=args.storm,
        include_group8=not args.no_group8,
        verbose=True
    )
    
    # Save or display
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path, index=False)
        print(f"\n💾 Saved to: {out_path}")
    else:
        print("\n📊 Preview (first 3 provinces, first 10 columns):")
        print(features.iloc[:3, :10])
        
        print(f"\n📋 Feature Summary:")
        print(f"   Total features: {len(features.columns)}")
        print(f"   Provinces: {features['Province'].nunique()}")
        print(f"   Missing values: {features.isnull().sum().sum()}")

