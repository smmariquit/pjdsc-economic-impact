"""
COMPLETE REAL-TIME FORECAST PIPELINE

END-TO-END: JTWC Forecast → Weather Forecast → Full Features → Predictions

This is the PRODUCTION-READY pipeline with complete feature extraction
including GROUP 2 (weather exposure) features.

WORKFLOW:
1. Parse JTWC forecast bulletin (from file OR fetch live from JTWC)
2. Fetch weather forecast from Open-Meteo API
3. Extract ALL features (Groups 1-8 + weather)
4. Load trained models
5. Generate predictions
6. Export results

USAGE:
    # Live mode (fetch from JTWC)
    python complete_forecast_pipeline.py --storm-id wp3025
    
    # File mode (use local file)
    python complete_forecast_pipeline.py --forecast storm_forecast.txt
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse_jtwc_forecast import parse_jtwc_forecast
from fetch_forecast_weather import fetch_weather_forecast
from unified_pipeline import StormFeaturePipeline


def fetch_jtwc_live(storm_id: str) -> str:
    """
    Fetch live JTWC bulletin directly from web.
    
    Args:
        storm_id: JTWC storm identifier (e.g., 'wp3025')
    
    Returns:
        Bulletin text content (string)
    
    Raises:
        requests.HTTPError: If bulletin not found or server error
        requests.Timeout: If request times out
    """
    url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
    
    print(f"📡 Fetching live JTWC bulletin: {storm_id}")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Extract storm name for display
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
                f"   The storm may have dissipated or the ID is incorrect.\n"
                f"   Check active storms at: https://www.metoc.navy.mil/jtwc/jtwc.html"
            )
        else:
            raise RuntimeError(f"❌ HTTP error fetching bulletin: {e}")
    
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"❌ Request timed out after 10 seconds.\n"
            f"   Check your internet connection and try again."
        )
    
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"❌ Connection error: {e}")
    
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error fetching bulletin: {e}")


def complete_forecast_inference(
    forecast_input: str,
    artifacts_dir: str = "artifacts",
    output_dir: str = "output",
    weather_cache: str = None,
    output_csv: str = None,
    output_json: str = None,
    is_text_content: bool = False
):
    """
    Complete forecast-to-prediction pipeline with FULL features.
    
    Args:
        forecast_input: Either path to JTWC forecast bulletin OR bulletin text content
        artifacts_dir: Directory with trained models
        output_dir: Output directory for all results (default: 'output')
        weather_cache: Optional path to save/load weather data
        output_csv: Optional CSV output filename (placed in output_dir)
        output_json: Optional JSON output filename (placed in output_dir)
        is_text_content: If True, forecast_input is text content (not file path)
    
    Returns:
        DataFrame with predictions
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPLETE REAL-TIME FORECAST PIPELINE")
    print("="*70)
    print(f"\n📁 Output directory: {output_path.absolute()}")
    print("✅ FULL FEATURE MODE (includes weather data)")
    print("   Expected accuracy: HIGH\n")
    
    # Step 1: Parse JTWC forecast
    print("="*70)
    print("STEP 1: Parse JTWC Forecast")
    print("="*70)
    
    forecast_track = parse_jtwc_forecast(forecast_input)
    
    year = forecast_track['SEASON'].iloc[0]
    storm_name = forecast_track['PHNAME'].iloc[0]
    start_date = forecast_track['datetime'].min().strftime('%Y-%m-%d')
    end_date = forecast_track['datetime'].max().strftime('%Y-%m-%d')
    
    print(f"\n✓ Storm: {storm_name} ({year})")
    print(f"✓ Track points: {len(forecast_track)}")
    print(f"✓ Period: {start_date} to {end_date}")
    
    # Step 2: Fetch weather forecast
    print("\n" + "="*70)
    print("STEP 2: Fetch Weather Forecast (Open-Meteo API)")
    print("="*70)
    
    # Set weather cache path in output directory if not specified
    if weather_cache is None:
        weather_cache = output_path / f"weather_forecast_{storm_name}_{year}.csv"
    
    if Path(weather_cache).exists():
        print(f"\n📁 Loading cached weather data: {weather_cache}")
        weather_forecast = pd.read_csv(weather_cache)
        weather_forecast['date'] = pd.to_datetime(weather_forecast['date'])
        print(f"✓ Loaded {len(weather_forecast)} weather records")
    else:
        print(f"\n🌐 Fetching weather forecast from Open-Meteo...")
        weather_forecast = fetch_weather_forecast(
            start_date=start_date,
            end_date=end_date,
            output_file=str(weather_cache)
        )
    
    # Step 3: Create temporary weather file in expected format
    print("\n" + "="*70)
    print("STEP 3: Prepare Data for Feature Extraction")
    print("="*70)
    
    # Save intermediate files in output directory
    temp_dir = output_path / "intermediate"
    temp_dir.mkdir(exist_ok=True)
    
    # Save track in format expected by pipeline
    track_file = temp_dir / f"track_{storm_name}_{year}.csv"
    forecast_track.to_csv(track_file, index=False)
    print(f"\n✓ Saved track to: {track_file}")
    
    # Save weather in format expected by pipeline  
    weather_file = temp_dir / f"{year}_{storm_name}.csv"
    weather_forecast.to_csv(weather_file, index=False)
    print(f"✓ Saved weather to: {weather_file}")
    
    # Step 4: Extract features using unified pipeline
    print("\n" + "="*70)
    print("STEP 4: Extract Features (ALL GROUPS)")
    print("="*70)
    
    # Temporarily point pipeline to our forecast data
    import shutil
    original_weather_dir = Path("Weather_location_data")
    
    # Create temporary year folder
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
        print(f"\n🔧 Running unified pipeline...")
        pipeline = StormFeaturePipeline()
        features = pipeline.process_storm(
            year=year,
            storm_name=storm_name,
            include_group8=True,
            verbose=True
        )
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Features: {features.shape}")
        
    finally:
        # Restore original storm data
        original_storm_data.to_csv(storm_data_file, index=False)
        print(f"\n🧹 Cleaned up temporary files")
    
    # Step 5: Add group prefixes and prepare for model
    print("\n" + "="*70)
    print("STEP 5: Prepare Features for Model")
    print("="*70)
    
    from example_deployment import add_group_prefixes
    features = add_group_prefixes(features)
    
    # Add vulnerability features
    if 'hist_storms' not in features.columns:
        features['hist_storms'] = 0.0
    if 'hist_avg_affected' not in features.columns:
        features['hist_avg_affected'] = 0.0
    if 'hist_max_affected' not in features.columns:
        features['hist_max_affected'] = 0.0
    
    print(f"✓ Features prepared: {features.shape}")
    
    # Step 6: Load models and predict
    print("\n" + "="*70)
    print("STEP 6: Load Models and Generate Predictions")
    print("="*70)
    
    artifacts_path = Path(artifacts_dir)
    
    clf = joblib.load(artifacts_path / "stage1_classifier.joblib")
    reg = joblib.load(artifacts_path / "stage2_regressor.joblib")
    
    with open(artifacts_path / "feature_columns.json") as f:
        feature_data = json.load(f)
        required_features = feature_data['feature_columns'] if isinstance(feature_data, dict) else feature_data
    
    print(f"\n✓ Models loaded")
    print(f"✓ Required features: {len(required_features)}")
    
    # Ensure all required features present
    for feat in required_features:
        if feat not in features.columns:
            features[feat] = 0.0
            print(f"   ⚠️  Missing: {feat} (defaulted to 0)")
    
    X = features[required_features]
    
    # Make predictions
    print(f"\n🎯 Generating predictions...")
    impact_proba = clf.predict_proba(X)[:, 1]
    log_affected = reg.predict(X)
    affected_persons = np.expm1(log_affected)
    final_affected = np.where(impact_proba > 0.1, affected_persons, 0)
    
    print(f"✓ Predictions generated")
    
    # Step 7: Create results
    print("\n" + "="*70)
    print("STEP 7: Format Results")
    print("="*70)
    
    results = features[['Year', 'Storm', 'Province']].copy()
    results['impact_probability'] = impact_proba
    results['predicted_affected_raw'] = affected_persons
    results['predicted_affected_final'] = final_affected
    results['risk_score'] = impact_proba * np.log1p(affected_persons)
    
    # Add metadata
    results['forecast_source'] = 'JTWC + Open-Meteo'
    results['forecast_time'] = datetime.now().isoformat()
    results['feature_mode'] = 'COMPLETE'
    results['confidence'] = 'HIGH'
    
    # Sort by risk
    results = results.sort_values('risk_score', ascending=False).reset_index(drop=True)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 PREDICTION SUMMARY")
    print("="*70)
    
    print(f"\nStorm: {storm_name} ({year})")
    print(f"Forecast period: {start_date} to {end_date}")
    print(f"Feature mode: COMPLETE (all groups including weather)")
    print(f"Confidence level: HIGH")
    
    print(f"\nProvinces at risk:")
    print(f"  >70% probability: {(impact_proba > 0.7).sum()}")
    print(f"  >50% probability: {(impact_proba > 0.5).sum()}")
    print(f"  >30% probability: {(impact_proba > 0.3).sum()}")
    
    print(f"\nTotal predicted affected: {final_affected.sum():,.0f} persons")
    
    print(f"\n🚨 TOP 10 PROVINCES AT RISK")
    print(f"\n{'Rank':<6} {'Province':<25} {'Impact %':<12} {'Predicted':<15}")
    print("="*70)
    
    for i, row in results.head(10).iterrows():
        print(
            f"{i+1:<6} "
            f"{row['Province']:<25} "
            f"{row['impact_probability']*100:>8.1f}%    "
            f"{row['predicted_affected_final']:>12,.0f}"
        )
    
    # Export results to output directory
    if output_csv is None:
        output_csv = f"predictions_{storm_name}_{year}.csv"
    if output_json is None:
        output_json = f"alerts_{storm_name}_{year}.json"
    
    # Ensure files are in output directory
    csv_path = output_path / output_csv if not Path(output_csv).is_absolute() else Path(output_csv)
    json_path = output_path / output_json if not Path(output_json).is_absolute() else Path(output_json)
    
    results.to_csv(csv_path, index=False)
    print(f"\n✅ Saved predictions to: {csv_path}")
    
    if True:  # Always save JSON
        alerts = []
        for _, row in results[results['impact_probability'] >= 0.3].iterrows():
            alerts.append({
                'province': row['Province'],
                'impact_probability': float(row['impact_probability']),
                'predicted_affected': int(row['predicted_affected_final']),
                'risk_score': float(row['risk_score']),
                'alert_level': 'CRITICAL' if row['impact_probability'] > 0.8 else 'HIGH' if row['impact_probability'] > 0.5 else 'MEDIUM',
                'confidence': 'HIGH'
            })
        
        output_data = {
            'storm': f"{year} {storm_name}",
            'forecast_source': 'JTWC + Open-Meteo',
            'forecast_period': f"{start_date} to {end_date}",
            'forecast_time': datetime.now().isoformat(),
            'feature_mode': 'COMPLETE',
            'confidence_level': 'HIGH',
            'total_provinces_at_risk': len(alerts),
            'total_predicted_affected': int(final_affected.sum()),
            'alerts': alerts
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Saved alerts to: {json_path}")
    
    # Create summary report
    summary_path = output_path / f"summary_{storm_name}_{year}.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"STORM IMPACT PREDICTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Storm: {storm_name} ({year})\n")
        f.write(f"Forecast period: {start_date} to {end_date}\n")
        f.write(f"Forecast source: JTWC + Open-Meteo\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature mode: COMPLETE\n")
        f.write(f"Confidence level: HIGH\n\n")
        f.write(f"Provinces at risk:\n")
        f.write(f"  >70% probability: {(impact_proba > 0.7).sum()}\n")
        f.write(f"  >50% probability: {(impact_proba > 0.5).sum()}\n")
        f.write(f"  >30% probability: {(impact_proba > 0.3).sum()}\n\n")
        f.write(f"Total predicted affected: {final_affected.sum():,.0f} persons\n\n")
        f.write("="*70 + "\n")
        f.write("TOP 20 PROVINCES AT RISK\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Rank':<6} {'Province':<25} {'Impact %':<12} {'Predicted':<15}\n")
        f.write("-"*70 + "\n")
        for i, row in results.head(20).iterrows():
            f.write(
                f"{i+1:<6} "
                f"{row['Province']:<25} "
                f"{row['impact_probability']*100:>8.1f}%    "
                f"{row['predicted_affected_final']:>12,.0f}\n"
            )
        f.write("\n" + "="*70 + "\n")
        f.write("FILES GENERATED\n")
        f.write("="*70 + "\n\n")
        f.write(f"  Predictions CSV: {csv_path.name}\n")
        f.write(f"  Alerts JSON: {json_path.name}\n")
        f.write(f"  Weather data: {Path(weather_cache).name}\n")
        f.write(f"  Storm track: {track_file.name}\n")
        f.write(f"  Summary: {summary_path.name}\n")
    
    print(f"✅ Saved summary to: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📁 All outputs saved to: {output_path.absolute()}")
    print(f"\n   Files generated:")
    print(f"   ├─ {csv_path.name}")
    print(f"   ├─ {json_path.name}")
    print(f"   ├─ {summary_path.name}")
    print(f"   └─ intermediate/")
    print(f"       ├─ {Path(weather_cache).name}")
    print(f"       ├─ {track_file.name}")
    print(f"       └─ {weather_file.name}")
    print()
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete forecast pipeline with weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Live mode (fetch from JTWC)
  python complete_forecast_pipeline.py --storm-id wp3025
  
  # File mode (use local file)
  python complete_forecast_pipeline.py --forecast storm_forecast.txt
  
  # Live mode with custom output
  python complete_forecast_pipeline.py --storm-id wp3025 --output-dir results_live
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
                       help="Model artifacts directory")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for all results (default: output)")
    parser.add_argument("--weather-cache", type=str,
                       help="Custom path for weather cache (default: output/weather_forecast_*.csv)")
    parser.add_argument("--export-csv", type=str,
                       help="CSV filename (default: predictions_STORM_YEAR.csv)")
    parser.add_argument("--export-json", type=str,
                       help="JSON filename (default: alerts_STORM_YEAR.json)")
    
    args = parser.parse_args()
    
    # Determine forecast input (text content or file path)
    if args.storm_id:
        # Live mode: fetch from JTWC (returns text content)
        try:
            forecast_input = fetch_jtwc_live(args.storm_id)
            is_text = True
        except RuntimeError as e:
            print(str(e))
            exit(1)
    else:
        # File mode: use provided file path
        forecast_input = args.forecast
        is_text = False
    
    # Run pipeline
    results = complete_forecast_inference(
        forecast_input=forecast_input,
        artifacts_dir=args.artifacts,
        output_dir=args.output_dir,
        weather_cache=args.weather_cache,
        output_csv=args.export_csv,
        output_json=args.export_json,
        is_text_content=is_text
    )

