"""
MINIMAL MODEL APP - Simple Interface
Uses the 6-feature RandomForest model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Minimal Storm Impact Predictor",
    page_icon="🌪️",
    layout="wide"
)

MODEL_PATH = Path("model/Final_Transform")

@st.cache_resource
def load_minimal_models():
    """Load the minimal sklearn models"""
    try:
        artifacts_dir = MODEL_PATH / "artifacts_minimal"
        
        clf = joblib.load(artifacts_dir / "classifier_minimal.joblib")
        reg = joblib.load(artifacts_dir / "regressor_minimal.joblib")
        
        with open(artifacts_dir / "features_minimal.json", 'r') as f:
            features = json.load(f)['features']
        
        return {'classifier': clf, 'regressor': reg, 'features': features}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    st.title("🌪️ Minimal Storm Impact Predictor")
    st.markdown("**6-Feature RandomForest Model** - Fast & Interpretable")
    
    # Load models
    models = load_minimal_models()
    
    if not models:
        st.error("❌ Could not load models. Run train_minimal.py first!")
        return
    
    st.success(f"✅ Models loaded | Features: {', '.join(models['features'])}")
    
    st.markdown("---")
    
    # Manual input form
    st.subheader("📝 Enter Storm & Province Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_distance_km = st.number_input(
            "Minimum Distance (km)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            help="How close did the storm get to the province?"
        )
        
        max_wind_gust_kmh = st.number_input(
            "Max Wind Gust (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            help="Peak wind gust speed"
        )
        
        total_precipitation_mm = st.number_input(
            "Total Precipitation (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=200.0,
            help="Total rainfall during storm"
        )
    
    with col2:
        max_wind_in_track_kt = st.number_input(
            "Max Wind in Track (knots)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            help="Maximum wind speed in storm track"
        )
        
        hours_under_100km = st.number_input(
            "Hours Under 100km",
            min_value=0.0,
            max_value=200.0,
            value=24.0,
            help="How long was the storm within 100km?"
        )
        
        population = st.number_input(
            "Province Population",
            min_value=0,
            max_value=20000000,
            value=1000000,
            help="Population of the province"
        )
    
    if st.button("🔮 Predict Impact", type="primary"):
        # Create feature vector
        X = pd.DataFrame({
            'min_distance_km': [min_distance_km],
            'max_wind_gust_kmh': [max_wind_gust_kmh],
            'total_precipitation_mm': [total_precipitation_mm],
            'max_wind_in_track_kt': [max_wind_in_track_kt],
            'hours_under_100km': [hours_under_100km],
            'Population': [population]
        })
        
        # Predict
        prob = models['classifier'].predict_proba(X)[0, 1]
        
        if prob > 0.5:
            pred_log = models['regressor'].predict(X)[0]
            pred_count = np.expm1(pred_log)
        else:
            pred_count = 0
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Impact Probability",
                f"{prob*100:.1f}%",
                delta="High Risk" if prob > 0.5 else "Low Risk"
            )
        
        with col2:
            st.metric(
                "Predicted Affected",
                f"{pred_count:,.0f} people"
            )
        
        with col3:
            risk_level = "🔴 CRITICAL" if prob > 0.7 else "🟡 MODERATE" if prob > 0.3 else "🟢 LOW"
            st.metric("Risk Level", risk_level)
        
        # Explain prediction
        st.markdown("---")
        st.subheader("📈 Feature Importance")
        
        # Feature importance from model
        importances = models['classifier'].feature_importances_
        feature_df = pd.DataFrame({
            'Feature': models['features'],
            'Importance': importances,
            'Your Value': X.iloc[0].values
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(feature_df, use_container_width=True)
        
        # Interpretation
        st.markdown("### 💡 Interpretation")
        
        if min_distance_km < 50:
            st.warning("⚠️ Storm very close (<50km) - high risk!")
        
        if max_wind_gust_kmh > 150:
            st.warning("⚠️ Very strong winds (>150 km/h) - significant damage risk!")
        
        if total_precipitation_mm > 300:
            st.warning("⚠️ Heavy rainfall (>300mm) - flooding risk!")
        
        if hours_under_100km > 48:
            st.warning("⚠️ Long exposure (>48 hours) - cumulative damage!")

if __name__ == "__main__":
    main()
