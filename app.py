import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import folium
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime
import io

# Add model paths
MODEL_PATH = Path("model/Final_Transform")
sys.path.insert(0, str(MODEL_PATH / "pipeline"))

# Import model pipeline components
try:
    from parse_jtwc_forecast import parse_jtwc_forecast
    from fetch_forecast_weather import fetch_weather_forecast
    from unified_pipeline import StormFeaturePipeline
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    st.error("⚠️ Model pipeline not found. Please check model/Final_Transform/pipeline/")


def load_ml_models():
    """Load trained ML models for inference."""
    try:
        # Load persons models
        clf_persons = joblib.load(MODEL_PATH / "artifacts" / "stage1_classifier.joblib")
        reg_persons = joblib.load(MODEL_PATH / "artifacts" / "stage2_regressor.joblib")
        
        # Load houses models
        clf_houses = joblib.load(MODEL_PATH / "artifacts_houses" / "stage1_classifier_houses.joblib")
        reg_houses = joblib.load(MODEL_PATH / "artifacts_houses" / "stage2_regressor_houses.joblib")
        
        # Load feature columns
        with open(MODEL_PATH / "artifacts" / "feature_columns.json", 'r') as f:
            feature_cols = json.load(f)['feature_columns']
        
        return {
            'persons': {'classifier': clf_persons, 'regressor': reg_persons},
            'houses': {'classifier': clf_houses, 'regressor': reg_houses},
            'features': feature_cols
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def display_ml_results(results_df, storm_name, year, track_df):
    """Display ML prediction results with visualizations."""
    
    # Summary metrics
    st.subheader(f"📊 Prediction Summary: {storm_name} ({year})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk = len(results_df[results_df['Impact_Probability_Persons'] > 70])
        st.metric("High Risk Provinces (>70%)", high_risk)
    
    with col2:
        total_affected = results_df['Predicted_Affected_Persons'].sum()
        st.metric("Total Predicted Affected", f"{total_affected:,.0f} persons")
    
    with col3:
        total_houses = results_df['Predicted_Houses_Damaged'].sum()
        st.metric("Total Houses Damaged", f"{total_houses:,.0f}")
    
    with col4:
        avg_prob = results_df['Impact_Probability_Persons'].mean()
        st.metric("Avg Impact Probability", f"{avg_prob:.1f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Top Provinces", "🗺️ Risk Map", "📊 Charts", "💾 Export"])
    
    with tab1:
        st.subheader("Top 20 Provinces at Risk")
        
        display_cols = ['Province', 'Risk_Category', 'Impact_Probability_Persons', 
                       'Predicted_Affected_Persons', 'Predicted_Houses_Damaged']
        
        top_20 = results_df.head(20)[display_cols].copy()
        top_20.columns = ['Province', 'Risk', 'Impact %', 'Persons', 'Houses']
        
        # Style the dataframe
        st.dataframe(
            top_20.style.background_gradient(subset=['Impact %'], cmap='Reds'),
            use_container_width=True,
            height=600
        )
    
    with tab2:
        st.subheader("Interactive Risk Map")
        
        # Load province coordinates
        try:
            loc_df = pd.read_csv(MODEL_PATH / "Location_data" / "locations_latlng.csv")
            
            # Merge with results
            map_data = results_df.merge(loc_df, left_on='Province', right_on='Province', how='left')
            
            # Create map
            m = folium.Map(location=[12.8797, 121.7740], zoom_start=6)
            
            # Add storm track
            for _, row in track_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LON']],
                    radius=5,
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7,
                    popup=f"Intensity: {row.get('INTENSITY', 'N/A')} kt"
                ).add_to(m)
            
            # Add province markers
            for _, row in map_data.iterrows():
                if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                    prob = row['Impact_Probability_Persons']
                    
                    if prob > 70:
                        color = 'darkred'
                        radius = 12
                    elif prob > 50:
                        color = 'orange'
                        radius = 10
                    elif prob > 30:
                        color = 'yellow'
                        radius = 8
                    else:
                        color = 'green'
                        radius = 6
                    
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.6,
                        popup=folium.Popup(
                            f"<b>{row['Province']}</b><br>"
                            f"Risk: {row['Risk_Category']}<br>"
                            f"Impact: {prob:.1f}%<br>"
                            f"Persons: {row['Predicted_Affected_Persons']:,.0f}<br>"
                            f"Houses: {row['Predicted_Houses_Damaged']:,.0f}",
                            max_width=250
                        )
                    ).add_to(m)
            
            st_folium(m, width=1200, height=600)
            
        except Exception as e:
            st.error(f"Map error: {str(e)}")
    
    with tab3:
        st.subheader("Impact Analysis Charts")
        
        # Chart 1: Top provinces bar chart
        fig1 = px.bar(
            results_df.head(15),
            x='Province',
            y='Predicted_Affected_Persons',
            color='Risk_Category',
            title='Top 15 Provinces: Predicted Affected Persons',
            color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
        )
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Risk distribution
        risk_dist = results_df['Risk_Category'].value_counts()
        fig2 = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Distribution of Risk Levels',
            color=risk_dist.index,
            color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Scatter plot
        fig3 = px.scatter(
            results_df,
            x='Predicted_Affected_Persons',
            y='Predicted_Houses_Damaged',
            color='Impact_Probability_Persons',
            hover_data=['Province'],
            title='Persons vs Houses Impact',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.subheader("Export Predictions")
        
        # Prepare export
        export_df = results_df.copy()
        export_df['Storm'] = storm_name
        export_df['Year'] = year
        export_df['Generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv_data,
            file_name=f"predictions_{storm_name}_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.dataframe(export_df, use_container_width=True)


def main():
    """Main Streamlit application for Philippines Typhoon Impact Prediction."""
    
    st.set_page_config(
        page_title="🌪️ Philippines Typhoon Impact Predictor (ML)",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌪️ Philippines Typhoon Impact Predictor")
    st.markdown("""
    **AI-Powered Storm Impact Prediction System**  
    Predicts humanitarian and infrastructure impacts using real-time JTWC forecasts and trained ML models.
    
    📊 **Models**: Dual two-stage cascades (Persons Affected + Houses Damaged)  
    🎯 **Accuracy**: 99.8% F1 Score on test data  
    🕐 **Lead Time**: 24-120 hours before landfall
    """)
    
    # Sidebar for prediction mode selection
    st.sidebar.header("🎯 Prediction Mode")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["🤖 ML Prediction (Real Storm)", "📊 Simple Impact Estimation", "📈 Historical Storm Analysis"],
        help="Choose prediction method"
    )
    
    # =========================================================================
    # MODE 1: ML PREDICTION WITH REAL MODELS
    # =========================================================================
    if mode == "🤖 ML Prediction (Real Storm)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🌪️ Storm Input")
        
        input_method = st.sidebar.radio(
            "Storm Data Source",
            ["📝 JTWC Bulletin (Text)", "🔗 Live JTWC (Storm ID)", "📁 Sample Forecast"]
        )
        
        storm_bulletin = None
        storm_id = None
        
        if input_method == "📝 JTWC Bulletin (Text)":
            st.sidebar.info("Paste JTWC bulletin text (from https://www.metoc.navy.mil/jtwc/)")
            storm_bulletin = st.sidebar.text_area(
                "JTWC Bulletin",
                height=200,
                placeholder="Paste full JTWC warning text here..."
            )
        
        elif input_method == "🔗 Live JTWC (Storm ID)":
            storm_id = st.sidebar.text_input(
                "JTWC Storm ID",
                value="wp3025",
                help="Format: wp[number][year] (e.g., wp3025 = Western Pacific storm 30, 2025)"
            )
        
        elif input_method == "📁 Sample Forecast":
            sample_file = MODEL_PATH / "storm_forecast.txt"
            if sample_file.exists():
                with open(sample_file, 'r') as f:
                    storm_bulletin = f.read()
                st.sidebar.success("✅ Sample forecast (FENGSHEN 2025) loaded!")
            else:
                st.sidebar.error("❌ Sample file not found")
        
        # Run prediction button
        if st.sidebar.button("🚀 Run ML Prediction", type="primary", use_container_width=True):
            if not MODEL_AVAILABLE:
                st.error("❌ Model pipeline not available. Please check installation.")
                return
            
            with st.spinner("🔄 Loading ML models..."):
                models = load_ml_models()
            
            if models is None:
                st.error("❌ Failed to load models")
                return
            
            try:
                # STEP 1: Parse storm forecast
                with st.spinner("� Parsing storm forecast..."):
                    if storm_bulletin:
                        # Save bulletin to temp file
                        temp_file = Path("temp_bulletin.txt")
                        with open(temp_file, 'w') as f:
                            f.write(storm_bulletin)
                        track_df, storm_name, year = parse_jtwc_forecast(str(temp_file))
                        temp_file.unlink()  # Delete temp file
                    elif storm_id:
                        # Fetch live
                        import requests
                        url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        
                        temp_file = Path("temp_bulletin.txt")
                        with open(temp_file, 'w') as f:
                            f.write(response.text)
                        track_df, storm_name, year = parse_jtwc_forecast(str(temp_file))
                        temp_file.unlink()
                    else:
                        st.error("No storm data provided")
                        return
                
                st.success(f"✅ Storm: {storm_name} ({year})")
                
                # STEP 2: Fetch weather forecasts
                with st.spinner("🌦️ Fetching weather forecasts for provinces..."):
                    weather_df = fetch_weather_forecast(
                        track_df,
                        str(MODEL_PATH / "Location_data" / "locations_latlng.csv")
                    )
                
                st.success(f"✅ Weather data fetched for {len(weather_df['Province'].unique())} provinces")
                
                # STEP 3: Engineer features
                with st.spinner("🔧 Engineering features (75+ features per province)..."):
                    pipeline = StormFeaturePipeline(
                        track_data=track_df,
                        weather_data=weather_df,
                        year=year,
                        storm_name=storm_name
                    )
                    features_df = pipeline.extract_all_features()
                
                st.success(f"✅ Features extracted: {features_df.shape[1]} features × {features_df.shape[0]} provinces")
                
                # STEP 4: Make predictions
                with st.spinner("🤖 Running ML predictions (dual models)..."):
                    # Align features
                    X = features_df[models['features']]
                    
                    # Persons prediction
                    prob_persons = models['persons']['classifier'].predict_proba(X)[:, 1]
                    pred_persons_log = models['persons']['regressor'].predict(X)
                    pred_persons = np.expm1(pred_persons_log)  # Inverse log transform
                    pred_persons = np.where(prob_persons > 0.5, pred_persons, 0)
                    
                    # Houses prediction
                    prob_houses = models['houses']['classifier'].predict_proba(X)[:, 1]
                    pred_houses_log = models['houses']['regressor'].predict(X)
                    pred_houses = np.expm1(pred_houses_log)
                    pred_houses = np.where(prob_houses > 0.5, pred_houses, 0)
                
                # STEP 5: Create results dataframe
                results_df = pd.DataFrame({
                    'Province': features_df.index,
                    'Impact_Probability_Persons': prob_persons * 100,
                    'Predicted_Affected_Persons': pred_persons.astype(int),
                    'Impact_Probability_Houses': prob_houses * 100,
                    'Predicted_Houses_Damaged': pred_houses.astype(int)
                })
                
                # Add risk category
                results_df['Risk_Category'] = pd.cut(
                    results_df['Impact_Probability_Persons'],
                    bins=[0, 30, 50, 70, 100],
                    labels=['Low', 'Moderate', 'High', 'Very High']
                )
                
                results_df = results_df.sort_values('Impact_Probability_Persons', ascending=False)
                
                st.success("✅ Predictions complete!")
                
                # Display results
                display_ml_results(results_df, storm_name, year, track_df)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # =========================================================================
    # MODE 2: SIMPLE IMPACT ESTIMATION (LEGACY)
    # =========================================================================
    elif mode == "📊 Simple Impact Estimation":
        from utils import (
            calculate_storm_impact, 
            create_philippines_map, 
            create_impact_charts,
            validate_data_format
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("�📊 Data Input")
        
        # Sample data option
        use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
        
        provinces_data = None
        storm_data = None
    
    if use_sample_data:
        # Load sample data
        try:
            provinces_data = pd.read_csv("data/sample_provinces.csv")
            storm_data = pd.read_csv("data/sample_forecast.csv")
            st.sidebar.success("✅ Sample data loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error("❌ Sample data files not found. Please upload your own data.")
            use_sample_data = False
    
    if not use_sample_data:
        # File upload sections
        st.sidebar.subheader("Province Data")
        provinces_file = st.sidebar.file_uploader(
            "Upload Province CSV", 
            type=['csv'],
            help="CSV with columns: province, lat, lon, population (optional), gdp_per_capita (optional)"
        )
        
        st.sidebar.subheader("Storm Forecast Data")
        storm_file = st.sidebar.file_uploader(
            "Upload Storm Forecast CSV", 
            type=['csv'],
            help="CSV with columns: lat, lon, intensity, timestamp (optional)"
        )
        
        # Process uploaded files
        if provinces_file is not None:
            try:
                provinces_data = pd.read_csv(provinces_file)
                is_valid, errors = validate_data_format(provinces_data, 'provinces')
                if is_valid:
                    st.sidebar.success("✅ Province data validated!")
                else:
                    st.sidebar.error("❌ Province data validation errors:")
                    for error in errors:
                        st.sidebar.error(f"• {error}")
                    provinces_data = None
            except Exception as e:
                st.sidebar.error(f"❌ Error loading province data: {str(e)}")
        
        if storm_file is not None:
            try:
                storm_data = pd.read_csv(storm_file)
                is_valid, errors = validate_data_format(storm_data, 'forecast')
                if is_valid:
                    st.sidebar.success("✅ Storm data validated!")
                else:
                    st.sidebar.error("❌ Storm data validation errors:")
                    for error in errors:
                        st.sidebar.error(f"• {error}")
                    storm_data = None
            except Exception as e:
                st.sidebar.error(f"❌ Error loading storm data: {str(e)}")
    
    # Analysis parameters
    st.sidebar.subheader("⚙️ Analysis Parameters")
    max_distance = st.sidebar.slider(
        "Maximum Impact Distance (km)", 
        min_value=50, 
        max_value=500, 
        value=200, 
        step=25,
        help="Maximum distance from storm center where economic impact is considered"
    )
    
    # Main content area
    if provinces_data is not None and storm_data is not None:
        # Perform impact analysis
        with st.spinner("🔄 Calculating storm impact..."):
            impact_data = calculate_storm_impact(storm_data, provinces_data, max_distance)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_provinces = len(provinces_data)
            st.metric("Total Provinces", total_provinces)
        
        with col2:
            affected_provinces = len(impact_data[impact_data['affected'] == True])
            st.metric("Affected Provinces", affected_provinces)
        
        with col3:
            total_economic_impact = impact_data['economic_impact_usd'].sum()
            st.metric("Total Economic Impact", f"${total_economic_impact:,.0f}")
        
        with col4:
            avg_impact_score = impact_data[impact_data['affected'] == True]['impact_score'].mean()
            if not pd.isna(avg_impact_score):
                st.metric("Avg Impact Score", f"{avg_impact_score:.3f}")
            else:
                st.metric("Avg Impact Score", "N/A")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Impact Analysis", "📋 Data Tables", "💾 Export Results"])
        
        with tab1:
            st.subheader("Interactive Philippines Map")
            
            # Create and display map
            philippines_map = create_philippines_map(provinces_data, storm_data, impact_data)
            map_data = st_folium(philippines_map, width=1200, height=600)
            
            st.markdown("""
            **Map Legend:**
            - 🔴 **Red circles**: High impact provinces (>50% impact score)
            - 🟠 **Orange circles**: Medium impact provinces (20-50% impact score)  
            - 🟡 **Yellow circles**: Low impact provinces (10-20% impact score)
            - 🟢 **Green circles**: No significant impact (<10% impact score)
            - 🔴 **Red line**: Storm track and intensity points
            """)
        
        with tab2:
            st.subheader("Economic Impact Analysis")
            
            # Create impact charts
            bar_chart, scatter_chart = create_impact_charts(impact_data)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(bar_chart, use_container_width=True)
            
            with col2:
                st.plotly_chart(scatter_chart, use_container_width=True)
            
            # Impact distribution
            st.subheader("Impact Distribution")
            impact_ranges = pd.cut(
                impact_data['impact_score'], 
                bins=[0, 0.1, 0.2, 0.5, 1.0], 
                labels=['No Impact', 'Low Impact', 'Medium Impact', 'High Impact']
            )
            impact_dist = impact_ranges.value_counts()
            
            fig_pie = px.pie(
                values=impact_dist.values, 
                names=impact_dist.index, 
                title="Distribution of Impact Levels"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            st.subheader("Data Tables")
            
            # Show different data views
            data_view = st.selectbox(
                "Select Data View", 
                ["Most Affected Provinces", "All Provinces", "Storm Forecast Points", "Summary by Impact Level"]
            )
            
            if data_view == "Most Affected Provinces":
                affected_data = impact_data[impact_data['affected'] == True].sort_values(
                    'economic_impact_usd', ascending=False
                )
                st.dataframe(affected_data, use_container_width=True)
            
            elif data_view == "All Provinces":
                display_data = impact_data.sort_values('impact_score', ascending=False)
                st.dataframe(display_data, use_container_width=True)
            
            elif data_view == "Storm Forecast Points":
                st.dataframe(storm_data, use_container_width=True)
            
            elif data_view == "Summary by Impact Level":
                summary_data = []
                for level in ['High Impact', 'Medium Impact', 'Low Impact', 'No Impact']:
                    if level == 'High Impact':
                        mask = impact_data['impact_score'] > 0.5
                    elif level == 'Medium Impact':
                        mask = (impact_data['impact_score'] > 0.2) & (impact_data['impact_score'] <= 0.5)
                    elif level == 'Low Impact':
                        mask = (impact_data['impact_score'] > 0.1) & (impact_data['impact_score'] <= 0.2)
                    else:
                        mask = impact_data['impact_score'] <= 0.1
                    
                    subset = impact_data[mask]
                    summary_data.append({
                        'Impact Level': level,
                        'Number of Provinces': len(subset),
                        'Total Population': subset['population'].sum(),
                        'Total Economic Impact ($)': subset['economic_impact_usd'].sum(),
                        'Avg Impact Score': subset['impact_score'].mean()
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        with tab4:
            st.subheader("Export Analysis Results")
            
            # Prepare export data
            export_data = impact_data.copy()
            export_data['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            export_data['max_impact_distance_km'] = max_distance
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            export_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Impact Analysis (CSV)",
                data=csv_data,
                file_name=f"storm_impact_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Show preview of export data
            st.subheader("Export Data Preview")
            st.dataframe(export_data.head(10), use_container_width=True)
    
    else:
        # Show instructions when no data is loaded
        st.info("""
        👆 Please upload your data files or enable sample data in the sidebar to begin analysis.
        
        ### Required Data Formats:
        
        **Province Data (CSV):**
        - `province`: Name of the province
        - `lat`: Latitude (decimal degrees)
        - `lon`: Longitude (decimal degrees)  
        - `population`: Population count (optional)
        - `gdp_per_capita`: GDP per capita in USD (optional)
        
        **Storm Forecast Data (CSV):**
        - `lat`: Latitude of storm point (decimal degrees)
        - `lon`: Longitude of storm point (decimal degrees)
        - `intensity`: Storm intensity (0-200 scale)
        - `timestamp`: Time of forecast (optional)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Philippines Storm Impact Analyzer | Built with Streamlit | 
    For educational and research purposes
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()