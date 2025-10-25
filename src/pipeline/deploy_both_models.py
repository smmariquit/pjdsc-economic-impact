"""
DUAL MODEL DEPLOYMENT: Affected Persons + House Damage

Performs inference for BOTH prediction models:
1. Affected Persons (humanitarian impact)
2. House Damage (infrastructure impact)

USAGE:
    # Live mode (fetch from JTWC)
    python pipeline/deploy_both_models.py --storm-id wp3025
    
    # File mode (use sample file)
    python pipeline/deploy_both_models.py --forecast storm_forecast.txt
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import requests

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse_jtwc_forecast import parse_jtwc_forecast
from fetch_forecast_weather import fetch_weather_forecast
from unified_pipeline import StormFeaturePipeline


def fetch_jtwc_live(storm_id: str) -> str:
    """Fetch live JTWC bulletin from web (returns text content)."""
    url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
    
    print(f"📡 Fetching live JTWC bulletin: {storm_id}")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Extract storm name
        storm_name = "UNKNOWN"
        for line in response.text.split('\n'):
            if "(" in line and ")" in line and "WARNING" in line:
                start = line.find("(") + 1
                end = line.find(")")
                storm_name = line[start:end]
                break
        
        print(f"✓ Downloaded: {storm_name}")
        print(f"✓ Processing in memory (no temp file)\n")
        
        return response.text
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"❌ Storm bulletin not found: {storm_id}\n"
                f"   Check active storms at: https://www.metoc.navy.mil/jtwc/jtwc.html"
            )
        else:
            raise RuntimeError(f"❌ HTTP error: {e}")
    
    except requests.exceptions.Timeout:
        raise RuntimeError("❌ Request timed out after 10 seconds.")
    
    except Exception as e:
        raise RuntimeError(f"❌ Error fetching bulletin: {e}")


def add_group_prefixes(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add groupN__ prefixes to feature names to match training format."""
    rename_map = {}
    
    # Group 1: Distance features
    group1_features = [
        'min_distance_km', 'mean_distance_km', 'max_distance_km', 'distance_range_km', 'distance_std_km',
        'hours_under_50km', 'hours_under_100km', 'hours_under_200km', 'hours_under_300km', 'hours_under_500km',
        'distance_at_current', 'distance_at_12hr', 'distance_at_24hr', 'distance_at_48hr', 'distance_at_72hr',
        'integrated_proximity', 'weighted_exposure_hours', 'proximity_peak',
        'approach_speed_kmh', 'departure_speed_kmh', 'time_approaching_hours', 'time_departing_hours',
        'bearing_at_closest_deg', 'bearing_variability_deg', 'did_cross_province', 'approach_angle_deg'
    ]
    
    # Group 2: Weather features
    group2_features = [
        'max_wind_gust_kmh', 'max_wind_speed_kmh', 'total_precipitation_mm', 'total_precipitation_hours',
        'days_with_rain', 'consecutive_rain_days', 'max_daily_precip_mm', 'max_hourly_precip_mm',
        'mean_daily_precipitation_mm', 'precip_variability', 'precipitation_concentration_index',
        'days_with_heavy_rain', 'days_with_very_heavy_rain', 'days_with_strong_wind', 'days_with_damaging_wind',
        'wind_gust_persistence_score', 'wind_rain_product', 'compound_hazard_score', 'compound_hazard_days',
        'rain_during_closest_approach'
    ]
    
    # Group 3: Intensity features
    group3_features = [
        'max_wind_in_track_kt', 'min_pressure_in_track_hpa', 'wind_at_closest_approach_kt',
        'pressure_at_closest_hpa', 'wind_change_approaching_kt', 'intensification_rate_kt_per_day',
        'is_intensifying'
    ]
    
    # Group 6: Motion features
    group6_features = [
        'mean_forward_speed', 'max_forward_speed', 'min_forward_speed', 'speed_at_closest_approach',
        'mean_direction', 'direction_variability', 'track_sinuosity', 'is_slow_moving',
        'is_fast_moving', 'is_recurving'
    ]
    
    # Group 7: Interaction features
    group7_features = [
        'min_distance_x_max_wind', 'proximity_intensity_product', 'intensity_per_km',
        'rainfall_distance_ratio', 'close_approach_rainfall', 'distant_rainfall'
    ]
    
    # Group 8: Multi-storm features
    group8_features = [
        'has_concurrent_storm', 'concurrent_storms_count', 'nearest_concurrent_storm_distance',
        'concurrent_storms_combined_intensity', 'days_since_last_storm', 'storms_past_30_days'
    ]
    
    for feat in group1_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group1__{feat}"
    
    for feat in group2_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group2__{feat}"
    
    for feat in group3_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group3__{feat}"
    
    for feat in group6_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group6__{feat}"
    
    for feat in group7_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group7__{feat}"
    
    for feat in group8_features:
        if feat in features_df.columns:
            rename_map[feat] = f"group8__{feat}"
    
    return features_df.rename(columns=rename_map)


def dual_model_inference(
    forecast_input: str,
    is_text_content: bool = False,
    artifacts_dir: str = "artifacts",
    artifacts_houses_dir: str = "artifacts_houses",
    output_dir: str = "output",
):
    """
    Complete dual-model forecast inference (persons + houses).
    
    Args:
        forecast_input: Either path to JTWC forecast bulletin OR bulletin text content
        is_text_content: If True, forecast_input is text content (not file path)
        artifacts_dir: Directory with persons model artifacts
        artifacts_houses_dir: Directory with houses model artifacts
        output_dir: Output directory for results
    
    Returns:
        Dict with results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DUAL MODEL DEPLOYMENT: PERSONS + HOUSES")
    print("="*70)
    print(f"\n📁 Output directory: {output_path.absolute()}\n")
    
    # Step 1: Parse JTWC forecast
    print("="*70)
    print("STEP 1: Parse JTWC Forecast")
    print("="*70)
    
    forecast_track = parse_jtwc_forecast(forecast_input)
    
    year = forecast_track['SEASON'].iloc[0]
    storm_name = forecast_track['PHNAME'].iloc[0]
    start_date = forecast_track['datetime'].min().strftime('%Y-%m-%d')
    end_date = forecast_track['datetime'].max().strftime('%Y-%m-%d')
    
    print(f"✓ Storm: {storm_name} ({year})")
    print(f"✓ Track points: {len(forecast_track)}")
    print(f"✓ Period: {start_date} to {end_date}\n")
    
    # Step 2: Fetch weather forecast
    print("="*70)
    print("STEP 2: Fetch Weather Forecast (Open-Meteo API)")
    print("="*70)
    
    weather_cache_file = output_path / f"weather_forecast_{storm_name}_{year}.csv"
    
    if weather_cache_file.exists():
        print(f"📁 Loading cached weather data: {weather_cache_file}")
        weather_df = pd.read_csv(weather_cache_file)
        print(f"✓ Loaded {len(weather_df)} weather records\n")
    else:
        weather_df = fetch_weather_forecast(forecast_track)
        weather_df.to_csv(weather_cache_file, index=False)
        print(f"✓ Fetched {len(weather_df)} weather records")
        print(f"✓ Cached to: {weather_cache_file}\n")
    
    # Step 3: Prepare data for feature extraction
    print("="*70)
    print("STEP 3: Prepare Data for Feature Extraction")
    print("="*70)
    
    import shutil
    
    intermediate_dir = output_path / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    
    # Save intermediate files
    track_file = intermediate_dir / f"track_{storm_name}_{year}.csv"
    weather_file = intermediate_dir / f"{year}_{storm_name}.csv"
    
    forecast_track.to_csv(track_file, index=False)
    weather_df.to_csv(weather_file, index=False)
    
    print(f"✓ Saved track to: {track_file}")
    print(f"✓ Saved weather to: {weather_file}\n")
    
    # Step 4: Extract features
    print("="*70)
    print("STEP 4: Extract Features (ALL GROUPS)")
    print("="*70)
    
    # Temporarily add forecast data to expected locations
    original_weather_dir = Path("Weather_location_data")
    temp_weather_year = original_weather_dir / str(year)
    temp_weather_year.mkdir(parents=True, exist_ok=True)
    
    # Copy weather file to expected location
    expected_weather_file = temp_weather_year / f"{year}_{storm_name}.csv"
    shutil.copy(weather_file, expected_weather_file)
    
    # Also add track to Storm_data temporarily
    storm_data_file = Path("Storm_data/ph_storm_data.csv")
    original_storm_data = pd.read_csv(storm_data_file)
    
    # Append forecast track
    combined_storm_data = pd.concat([original_storm_data, forecast_track], ignore_index=True)
    combined_storm_data.to_csv(storm_data_file, index=False)
    
    try:
        # Run unified pipeline
        print("🔧 Running unified pipeline...\n")
        
        pipeline = StormFeaturePipeline(base_dir=Path.cwd())
        features_df = pipeline.process_storm(year, storm_name)
        
        print(f"\n✓ Feature extraction complete!")
        print(f"   Features: {features_df.shape}\n")
        
    finally:
        # Clean up temporary files
        print("🧹 Cleaned up temporary files\n")
        
        # Restore original storm data
        original_storm_data.to_csv(storm_data_file, index=False)
        
        # Remove temporary weather file
        if expected_weather_file.exists():
            expected_weather_file.unlink()
    
    # Add group prefixes
    features_df = add_group_prefixes(features_df)
    
    # Step 5: Load models and feature lists
    print("="*70)
    print("STEP 5: Load Models")
    print("="*70)
    
    # Persons models
    clf_persons = joblib.load(Path(artifacts_dir) / "stage1_classifier.joblib")
    reg_persons = joblib.load(Path(artifacts_dir) / "stage2_regressor.joblib")
    
    with open(Path(artifacts_dir) / "feature_columns.json") as f:
        feature_data = json.load(f)
        required_features_persons = feature_data.get("feature_columns", feature_data) if isinstance(feature_data, dict) else feature_data
    
    # Houses models
    clf_houses = joblib.load(Path(artifacts_houses_dir) / "stage1_classifier_houses.joblib")
    reg_houses = joblib.load(Path(artifacts_houses_dir) / "stage2_regressor_houses.joblib")
    
    with open(Path(artifacts_houses_dir) / "feature_columns_houses.json") as f:
        feature_data_houses = json.load(f)
        required_features_houses = feature_data_houses.get("feature_columns", feature_data_houses) if isinstance(feature_data_houses, dict) else feature_data_houses
    
    print(f"✓ Loaded persons models")
    print(f"   Required features: {len(required_features_persons)}")
    print(f"✓ Loaded houses models")
    print(f"   Required features: {len(required_features_houses)}\n")
    
    # Add missing vulnerability features if needed
    for feat in ['hist_storms', 'hist_avg_affected', 'hist_max_affected']:
        if feat not in features_df.columns:
            features_df[feat] = 0.0
    
    for feat in ['hist_storms_houses', 'hist_avg_houses', 'hist_max_houses']:
        if feat not in features_df.columns:
            features_df[feat] = 0.0
    
    # Step 6: Generate predictions
    print("="*70)
    print("STEP 6: Generate Predictions (BOTH MODELS)")
    print("="*70)
    
    # PERSONS predictions
    print("🎯 Predicting affected persons...")
    prob_impact = clf_persons.predict_proba(features_df[required_features_persons])[:, 1]
    affected_log = reg_persons.predict(features_df[required_features_persons])
    predicted_affected = np.expm1(affected_log)
    
    # HOUSES predictions
    print("🏠 Predicting house damage...")
    prob_houses = clf_houses.predict_proba(features_df[required_features_houses])[:, 1]
    houses_log = reg_houses.predict(features_df[required_features_houses])
    predicted_houses = np.expm1(houses_log)
    
    print("✓ Predictions generated\n")
    
    # Step 7: Format results
    print("="*70)
    print("STEP 7: Format Results")
    print("="*70)
    
    results = pd.DataFrame({
        'Province': features_df['Province'],
        'Year': year,
        'Storm': storm_name,
        
        # Persons predictions
        'Impact_Probability_%': (prob_impact * 100).round(2),
        'Predicted_Affected_Persons': predicted_affected.round(0).astype(int),
        
        # Houses predictions
        'House_Damage_Probability_%': (prob_houses * 100).round(2),
        'Predicted_Houses_Damaged': predicted_houses.round(0).astype(int),
        
        # Risk categorization
        'Risk_Level_Persons': pd.cut(prob_impact, 
                                      bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
                                      labels=['LOW', 'MODERATE', 'HIGH', 'EXTREME']),
        'Risk_Level_Houses': pd.cut(prob_houses,
                                     bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
                                     labels=['LOW', 'MODERATE', 'HIGH', 'EXTREME'])
    })
    
    results = results.sort_values('Impact_Probability_%', ascending=False)
    
    # Save results
    csv_file = output_path / f"predictions_DUAL_{storm_name}_{year}.csv"
    results.to_csv(csv_file, index=False)
    print(f"✓ Saved predictions to: {csv_file}\n")
    
    # Create summary
    high_risk_persons = (prob_impact > 0.7).sum()
    high_risk_houses = (prob_houses > 0.7).sum()
    total_affected = predicted_affected.sum()
    total_houses = predicted_houses.sum()
    
    print("="*70)
    print("📊 PREDICTION SUMMARY")
    print("="*70)
    print(f"\nStorm: {storm_name} ({year})")
    print(f"Forecast period: {start_date} to {end_date}")
    print(f"Provinces analyzed: {len(results)}")
    print()
    print("AFFECTED PERSONS:")
    print(f"  High risk provinces (>70%): {high_risk_persons}")
    print(f"  Total predicted affected: {total_affected:,.0f} persons")
    print()
    print("HOUSE DAMAGE:")
    print(f"  High risk provinces (>70%): {high_risk_houses}")
    print(f"  Total predicted houses damaged: {total_houses:,.0f} houses")
    print()
    
    # Top 10 by persons
    print("🚨 TOP 10 PROVINCES AT RISK (Affected Persons)")
    print("="*70)
    print(f"{'Rank':<6} {'Province':<25} {'Impact %':<12} {'Predicted':<15} {'Houses':<12}")
    print("="*70)
    
    for idx, row in results.head(10).iterrows():
        print(f"{idx+1:<6} {row['Province']:<25} {row['Impact_Probability_%']:>10.1f}% {row['Predicted_Affected_Persons']:>13,} {row['Predicted_Houses_Damaged']:>10,}")
    
    print()
    
    # Save summary
    summary_file = output_path / f"summary_DUAL_{storm_name}_{year}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DUAL MODEL PREDICTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Storm: {storm_name} ({year})\n")
        f.write(f"Forecast period: {start_date} to {end_date}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models: Affected Persons + House Damage\n\n")
        f.write(f"AFFECTED PERSONS:\n")
        f.write(f"  High risk provinces (>70%): {high_risk_persons}\n")
        f.write(f"  Total predicted affected: {total_affected:,.0f} persons\n\n")
        f.write(f"HOUSE DAMAGE:\n")
        f.write(f"  High risk provinces (>70%): {high_risk_houses}\n")
        f.write(f"  Total predicted houses damaged: {total_houses:,.0f} houses\n\n")
        f.write("="*70 + "\n")
        f.write("TOP 20 PROVINCES AT RISK (by Impact Probability)\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Rank':<6} {'Province':<25} {'Impact %':<12} {'Affected':<15} {'Houses':<12}\n")
        f.write("-"*70 + "\n")
        
        for i, row in results.head(20).iterrows():
            f.write(f"{i+1:<6} {row['Province']:<25} {row['Impact_Probability_%']:>10.1f}% {row['Predicted_Affected_Persons']:>13,} {row['Predicted_Houses_Damaged']:>10,}\n")
    
    print(f"✓ Saved summary to: {summary_file}\n")
    
    print("="*70)
    print("✅ DUAL MODEL INFERENCE COMPLETE")
    print("="*70)
    print(f"\n📁 All outputs saved to: {output_path.absolute()}\n")
    
    return {
        'storm_name': storm_name,
        'year': year,
        'results': results,
        'summary': {
            'high_risk_persons': int(high_risk_persons),
            'high_risk_houses': int(high_risk_houses),
            'total_affected': float(total_affected),
            'total_houses': float(total_houses)
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dual model deployment (persons + houses)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Live mode (fetch from JTWC)
  python pipeline/deploy_both_models.py --storm-id wp3025
  
  # File mode (use sample file)
  python pipeline/deploy_both_models.py --forecast storm_forecast.txt
        """
    )
    
    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--storm-id", 
        type=str,
        help="Fetch live JTWC bulletin by storm ID (e.g., wp3025)"
    )
    input_group.add_argument(
        "--forecast", 
        type=str,
        help="Use local JTWC forecast bulletin file"
    )
    
    # Output options
    parser.add_argument("--artifacts", type=str, default="artifacts",
                       help="Persons model artifacts directory")
    parser.add_argument("--artifacts-houses", type=str, default="artifacts_houses",
                       help="Houses model artifacts directory")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for results (default: output)")
    
    args = parser.parse_args()
    
    # Determine forecast input
    if args.storm_id:
        # Live mode: fetch from JTWC
        try:
            forecast_input = fetch_jtwc_live(args.storm_id)
            is_text = True
        except RuntimeError as e:
            print(str(e))
            exit(1)
    else:
        # File mode: use provided file
        forecast_input = args.forecast
        is_text = False
    
    # Run dual model inference
    dual_model_inference(
        forecast_input=forecast_input,
        is_text_content=is_text,
        artifacts_dir=args.artifacts,
        artifacts_houses_dir=args.artifacts_houses,
        output_dir=args.output_dir
    )

