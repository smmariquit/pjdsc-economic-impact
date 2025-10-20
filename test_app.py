"""
Test script to validate the Philippines Storm Impact Analyzer application.
This script tests all major components without starting the GUI.
"""

import sys
import traceback
from utils import (
    calculate_distance, 
    calculate_storm_impact, 
    create_philippines_map,
    create_impact_charts,
    validate_data_format
)
import pandas as pd
import folium
import plotly.graph_objects as go


def test_data_loading():
    """Test loading sample data files."""
    print("🔍 Testing data loading...")
    try:
        provinces = pd.read_csv("data/sample_provinces.csv")
        forecast = pd.read_csv("data/sample_forecast.csv")
        
        assert len(provinces) > 0, "No provinces loaded"
        assert len(forecast) > 0, "No forecast data loaded"
        assert 'province' in provinces.columns, "Missing province column"
        assert 'lat' in provinces.columns, "Missing lat column"
        assert 'intensity' in forecast.columns, "Missing intensity column"
        
        print(f"✅ Data loaded: {len(provinces)} provinces, {len(forecast)} forecast points")
        return provinces, forecast
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None


def test_utility_functions(provinces, forecast):
    """Test all utility functions."""
    print("\n🔍 Testing utility functions...")
    
    try:
        # Test distance calculation
        dist = calculate_distance(14.5995, 120.9842, 13.7565, 121.0584)
        assert 90 < dist < 100, f"Distance calculation seems wrong: {dist}"
        print(f"✅ Distance calculation: {dist:.1f} km")
        
        # Test data validation
        is_valid_p, errors_p = validate_data_format(provinces, 'provinces')
        is_valid_f, errors_f = validate_data_format(forecast, 'forecast')
        assert is_valid_p, f"Province validation failed: {errors_p}"
        assert is_valid_f, f"Forecast validation failed: {errors_f}"
        print("✅ Data validation passed")
        
        # Test impact calculation
        impact_data = calculate_storm_impact(forecast, provinces.head(20))
        assert len(impact_data) == 20, "Impact calculation returned wrong number of rows"
        assert 'impact_score' in impact_data.columns, "Missing impact_score column"
        assert 'economic_impact_usd' in impact_data.columns, "Missing economic impact column"
        print(f"✅ Impact calculation: {len(impact_data)} provinces processed")
        
        return impact_data
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        traceback.print_exc()
        return None


def test_visualizations(provinces, forecast, impact_data):
    """Test visualization functions."""
    print("\n🔍 Testing visualizations...")
    
    try:
        # Test map creation
        philippines_map = create_philippines_map(provinces.head(10), forecast.head(10), impact_data.head(10))
        assert isinstance(philippines_map, folium.Map), "Map creation failed"
        print("✅ Philippines map created successfully")
        
        # Test chart creation
        bar_chart, scatter_chart = create_impact_charts(impact_data)
        assert isinstance(bar_chart, go.Figure), "Bar chart creation failed"
        assert isinstance(scatter_chart, go.Figure), "Scatter chart creation failed"
        print("✅ Impact charts created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔍 Testing edge cases...")
    
    try:
        # Test empty data
        empty_df = pd.DataFrame()
        is_valid, errors = validate_data_format(empty_df, 'provinces')
        assert not is_valid, "Empty data should not be valid"
        print("✅ Empty data validation works")
        
        # Test invalid coordinates
        bad_coords = pd.DataFrame({
            'province': ['Test'],
            'lat': [200],  # Invalid latitude
            'lon': [200]   # Invalid longitude
        })
        is_valid, errors = validate_data_format(bad_coords, 'provinces')
        assert not is_valid, "Invalid coordinates should not be valid"
        assert len(errors) > 0, "Should have validation errors"
        print("✅ Invalid coordinate detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🌪️ Philippines Storm Impact Analyzer - Test Suite")
    print("=" * 60)
    
    # Test data loading
    provinces, forecast = test_data_loading()
    if provinces is None or forecast is None:
        print("❌ Cannot continue testing without data")
        sys.exit(1)
    
    # Test utility functions
    impact_data = test_utility_functions(provinces, forecast)
    if impact_data is None:
        print("❌ Cannot continue testing without impact calculations")
        sys.exit(1)
    
    # Test visualizations
    viz_success = test_visualizations(provinces, forecast, impact_data)
    if not viz_success:
        print("❌ Visualization tests failed")
        sys.exit(1)
    
    # Test edge cases
    edge_success = test_edge_cases()
    if not edge_success:
        print("❌ Edge case tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! The application is ready to use.")
    print("\nTo run the Streamlit app:")
    print("streamlit run app.py")
    print("\nThe app will be available at: http://localhost:8501")


if __name__ == "__main__":
    main()