"""
Philippines Typhoon Impact Predictor - ML-Powered Streamlit App
Uses trained machine learning models to predict humanitarian and infrastructure impacts
"""

import requests
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import folium
import folium.map
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# USD to PHP conversion rate (approximate)
USD_TO_PHP = 56.0  # 1 USD = 56 PHP (as of 2025)

# Get project root directory (two levels up from src/app/main.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import Philippine location data from new location
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.locations import PH_REGIONS, REGION_NAMES

# Set data paths to new structure
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"

# Import model pipeline components from new location
try:
    from pipeline.parse_jtwc_forecast import parse_jtwc_forecast
    from pipeline.fetch_forecast_weather import fetch_weather_forecast
    from pipeline.unified_pipeline import StormFeaturePipeline
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)


def try_fetch_jtwc_bulletin(max_storm: int = 33,
                             max_warning: int = 29,
                             attempt_limit: int = 200,
                             timeout: int = 3):
    """Attempt to discover a live JTWC bulletin by iterating ids.

    Pattern: wp<storm#><warning#>web.txt, e.g., wp3110web.txt
    Returns (storm_id, text) on success, else (None, None).
    """
    import requests as _requests

    base_url = "https://www.metoc.navy.mil/jtwc/products"

    attempts = 0
    # Search newest-first to find the most recent likely bulletin quickly
    for storm_num in range(max_storm, 0, -1):
        for warn_num in range(max_warning, 0, -1):
            if attempts >= attempt_limit:
                return None, None
            storm_id = f"wp{storm_num:02d}{warn_num:02d}"
            url = f"{base_url}/{storm_id}web.txt"
            try:
                print("Trying URL:", url)
                # Fast HEAD probe to avoid downloading full body when missing
                head = _requests.head(url, timeout=timeout, allow_redirects=True)
                if head.status_code != 200:
                    continue
                # Stream GET and read only a small chunk to search marker
                resp = _requests.get(url, timeout=timeout, stream=True)
                content_chunks = []
                bytes_read = 0
                max_bytes = 65536  # 64 KB cap
                for chunk in resp.iter_content(chunk_size=4096):
                    if not chunk:
                        break
                    content_chunks.append(chunk)
                    bytes_read += len(chunk)
                    if bytes_read >= max_bytes:
                        break
                text = b''.join(content_chunks).decode(errors='ignore')
                if "WARNING POSITION" in text.upper():
                    # If we capped content, we might not have entire file; fetch full text now
                    if bytes_read < max_bytes:
                        full_text = text
                    else:
                        full_resp = _requests.get(url, timeout=timeout)
                        full_text = full_resp.text
                    return storm_id, full_text
            except _requests.exceptions.RequestException:
                # Ignore and continue
                pass
            finally:
                attempts += 1

    return None, None


@st.cache_data(show_spinner=False, ttl=600)
def cached_try_fetch_jtwc_bulletin():
    """Cached wrapper to avoid repeated scans within a short window."""
    return try_fetch_jtwc_bulletin()


def load_ml_models():
    """Load trained ML models for inference - ALL THREE IMPACT TYPES."""
    try:
        # Load MINIMAL models (6 features, sklearn RandomForest)
        artifacts_dir = MODELS_PATH / "minimal"
        
        models = {}
        
        # Load people models
        models['people'] = {
            'classifier': joblib.load(artifacts_dir / "classifier_people.joblib"),
            'regressor': joblib.load(artifacts_dir / "regressor_people.joblib")
        }
        
        # Load houses models
        models['houses'] = {
            'classifier': joblib.load(artifacts_dir / "classifier_houses.joblib"),
            'regressor': joblib.load(artifacts_dir / "regressor_houses.joblib")
        }
        
        # Load cost models
        models['cost'] = {
            'classifier': joblib.load(artifacts_dir / "classifier_cost.joblib"),
            'regressor': joblib.load(artifacts_dir / "regressor_cost.joblib")
        }
        
        # Load feature list (same for all models)
        with open(artifacts_dir / "features_minimal.json", 'r') as f:
            features = json.load(f)['features']
        
        models['features'] = features
        
        return models
    except Exception as e:
        st.error(f"Error loading minimal models: {str(e)}")
        st.info("💡 Run: `python -m src.training.train_minimal_all` first!")
        return None


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


@st.cache_data
def load_historical_impact_data():
    """Load historical impact data to compute province-level statistics."""
    try:
        impact_df = pd.read_csv(DATA_PATH / "raw" / "Impact_data" / "people_affected_all_years.csv")
        
        # Compute historical statistics by province
        hist_stats = impact_df.groupby('Province').agg({
            'Storm': 'count',  # Number of storms that affected this province
            'Affected': ['mean', 'max']  # Average and max persons affected
        }).reset_index()
        
        hist_stats.columns = ['Province', 'hist_storms', 'hist_avg_affected', 'hist_max_affected']
        
        # Fill NaN with 0 (provinces with no historical impact)
        hist_stats = hist_stats.fillna(0)
        
        return hist_stats
    except Exception as e:
        st.warning(f"Could not load historical data: {e}. Using default values.")
        return None


@st.cache_data
def load_historical_storm_data():
    """Load historical storm summary data from ph_storm_summary.csv."""
    try:
        storm_df = pd.read_csv(DATA_PATH / "raw" / "Storm_data" / "ph_storm_summary.csv")
        return storm_df
    except Exception as e:
        st.warning(f"Could not load storm summary data: {e}")
        return None


def add_historical_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add historical impact features based on province history."""
    hist_stats = load_historical_impact_data()
    
    if hist_stats is not None and 'Province' in features_df.columns:
        # Merge historical stats
        features_df = features_df.merge(hist_stats, on='Province', how='left')
        
        # Fill any remaining NaN with median values
        features_df['hist_storms'] = features_df['hist_storms'].fillna(features_df['hist_storms'].median())
        features_df['hist_avg_affected'] = features_df['hist_avg_affected'].fillna(features_df['hist_avg_affected'].median())
        features_df['hist_max_affected'] = features_df['hist_max_affected'].fillna(features_df['hist_max_affected'].median())
    else:
        # Use conservative defaults if no historical data available
        features_df['hist_storms'] = 5  # Median value
        features_df['hist_avg_affected'] = 1000  # Median value
        features_df['hist_max_affected'] = 5000  # Median value
    
    return features_df


def generate_lgu_insights(results_df, storm_name, year, selected_province=None):
    """Generate LGU recommendations using OpenAI API based on prediction results."""
    try:
        # Configure OpenAI API
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Prepare data summary
        high_risk_provinces = results_df[results_df['Impact_Probability_Persons'] > 30]
        total_affected = results_df['Predicted_Affected_Persons'].sum()
        total_houses = results_df['Predicted_Houses_Damaged'].sum()
        total_cost_usd = results_df['Predicted_Economic_Cost_USD'].sum()
        total_cost_php = total_cost_usd * USD_TO_PHP
        
        # Create province-specific context if selected
        province_context = ""
        if selected_province:
            province_data = results_df[results_df['Province'] == selected_province]
            if not province_data.empty:
                p = province_data.iloc[0]
                cost_php = p['Predicted_Economic_Cost_USD'] * USD_TO_PHP
                province_context = f"""
FOCUS PROVINCE: {selected_province}
- Risk Level: {p['Risk_Category']}
- Impact Probability: {p['Impact_Probability_Persons']:.1f}%
- Predicted People Affected: {int(p['Predicted_Affected_Persons']):,}
- Predicted Houses Damaged: {int(p['Predicted_Houses_Damaged']):,}
- Economic Cost: ₱{cost_php/1e6:.2f}M PHP
"""
        
        # Base context for all queries
        base_context = f"""TYPHOON: {storm_name} ({year})

PREDICTED IMPACT SUMMARY:
- High Risk Provinces (>30% probability): {len(high_risk_provinces)} provinces
- Total People Affected: {int(total_affected):,}
- Total Houses Damaged: {int(total_houses):,}
- Total Economic Cost: ₱{total_cost_php/1e6:.1f}M PHP

TOP 5 MOST AFFECTED PROVINCES:
{chr(10).join([f"- {row['Province']}: {row['Impact_Probability_Persons']:.1f}% probability, {int(row['Predicted_Affected_Persons']):,} people, {int(row['Predicted_Houses_Damaged']):,} houses" for _, row in results_df.head(5).iterrows()])}
{province_context}"""

        # Generate four separate responses for each aspect
        # Make system prompt emphasize specificity to the province
        if selected_province:
            system_prompt = f"You are a disaster management expert advising the Local Government Unit of {selected_province}. Provide clear, direct, SPECIFIC answers for {selected_province} only - not generic nationwide advice. Use actual numbers from the data provided. Write in plain text with simple numbered lists. Do NOT use markdown formatting, asterisks, or special characters."
        else:
            system_prompt = "You are a disaster management expert advising Local Government Units in the Philippines. Provide clear, direct answers without markdown formatting, asterisks, or special characters. Write in plain text with simple numbered lists."
        
        insights = {}
        
        # Question 1: Pre-Impact Preparations
        if selected_province:
            prompt1 = f"""{base_context}

Based on the predicted {int(results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Affected_Persons']):,} people affected and {int(results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Houses_Damaged']):,} houses damaged in {selected_province}, what SPECIFIC actions should the {selected_province} LGU take in the next 24-48 hours? Give concrete numbers for evacuations, shelters needed, and communication steps tailored to {selected_province}."""
        else:
            prompt1 = f"""{base_context}

What specific actions should LGUs take in the next 24-48 hours before the typhoon hits? Focus on evacuation, resource positioning, and communication."""
        
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt1}
            ],
            temperature=0.7,
            max_tokens=400
        )
        insights['preparation'] = response1.choices[0].message.content
        
        # Question 2: Response Operations
        if selected_province:
            prompt2 = f"""{base_context}

During the typhoon impact in {selected_province}, what should emergency response teams prioritize? Be SPECIFIC to {selected_province}'s predicted impact levels. Give concrete deployment numbers and priority areas within {selected_province}."""
        else:
            prompt2 = f"""{base_context}

What should emergency response teams prioritize during the typhoon impact? Cover response priorities, deployment, and safety protocols."""
        
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.7,
            max_tokens=400
        )
        insights['response'] = response2.choices[0].message.content
        
        # Question 3: Post-Impact Recovery
        if selected_province:
            prompt3 = f"""{base_context}

In the first 72 hours after the typhoon hits {selected_province}, what are the critical recovery actions? Use the specific numbers: {int(results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Affected_Persons']):,} people and {int(results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Houses_Damaged']):,} houses. Give CONCRETE steps for {selected_province} only."""
        else:
            prompt3 = f"""{base_context}

What are the critical recovery actions in the first 72 hours after the typhoon? Include assessment, relief distribution, and infrastructure restoration."""
        
        response3 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt3}
            ],
            temperature=0.7,
            max_tokens=400
        )
        insights['recovery'] = response3.choices[0].message.content
        
        # Question 4: Resource Requirements
        if selected_province:
            cost_php = results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Economic_Cost_USD'] * USD_TO_PHP
            prompt4 = f"""{base_context}

For {selected_province} specifically, what resources, supplies, and personnel are needed? The predicted economic cost is ₱{cost_php/1e6:.2f}M. Give SPECIFIC quantities (food packs, water, medical supplies, personnel) needed for {int(results_df[results_df['Province'] == selected_province].iloc[0]['Predicted_Affected_Persons']):,} affected people in {selected_province}."""
        else:
            prompt4 = f"""{base_context}

What specific resources, supplies, and personnel does the LGU need to prepare? Include quantities and budget considerations."""
        
        response4 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt4}
            ],
            temperature=0.7,
            max_tokens=400
        )
        insights['resources'] = response4.choices[0].message.content
        
        return insights
        
    except Exception as e:
        st.error(f"Error generating LGU insights: {str(e)}")
        return None


def display_ml_results(results_df, storm_name, year, track_df):
    """Display ML prediction results with visualizations."""
    
    # Summary metrics
    st.subheader(f"📊 Prediction Summary: {storm_name} ({year})")
    
    # Check if user has selected a province
    selected_province = st.session_state.get('selected_province', None)
    
    # COMMENTED OUT: LGU Action Plan section (GPT-generated recommendations)
    # st.markdown("---")
    # st.subheader("💡 LGU Action Plan")
    # 
    # with st.spinner("🧠 Generating disaster management recommendations..."):
    #     lgu_insights = generate_lgu_insights(results_df, storm_name, year, selected_province)
    
    if False:  # lgu_insights:
        # Create four columns for the four aspects
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 Pre-Impact (24-48hrs)")
            st.markdown(
                f"""
                <div style="height: 300px; overflow-y: auto; padding: 15px; color: #000;
                            border: 1px solid #d1ecf1; border-radius: 5px; background-color: #d1ecf1;">
                    {lgu_insights['preparation']}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("### 🏥 Post-Impact (First 72hrs)")
            st.markdown(
                f"""
                <div style="height: 300px; overflow-y: auto; padding: 15px; color: #000;
                            border: 1px solid #d4edda; border-radius: 5px; background-color: #d4edda;">
                    {lgu_insights['recovery']}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("### ⚡ During Impact")
            st.markdown(
                f"""
                <div style="height: 300px; overflow-y: auto; padding: 15px; color: #000000;
                            border: 1px solid #fff3cd; border-radius: 5px; background-color: #fff3cd;">
                    {lgu_insights['response']}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("### 📦 Resources Needed")
            st.markdown(
                f"""
                <div style="height: 300px; overflow-y: auto; padding: 15px; color: #000000;
                            border: 1px solid #f8d7da; border-radius: 5px; background-color: #f8d7da;">
                    {lgu_insights['resources']}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Add download as PDF option
        st.markdown("---")
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            from io import BytesIO
            
            # Create PDF in memory
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor='darkblue',
                spaceAfter=20
            )
            story.append(Paragraph(f"LGU Action Plan: {storm_name} ({year})", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Add each section
            section_style = ParagraphStyle(
                'SectionTitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='darkred',
                spaceAfter=10
            )
            
            sections = [
                ("Pre-Impact Preparations (24-48 hours)", lgu_insights['preparation']),
                ("Response Operations (During Impact)", lgu_insights['response']),
                ("Post-Impact Recovery (First 72 hours)", lgu_insights['recovery']),
                ("Resource Requirements", lgu_insights['resources'])
            ]
            
            for title, content in sections:
                story.append(Paragraph(title, section_style))
                for line in content.split('\n'):
                    if line.strip():
                        story.append(Paragraph(line.strip(), styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            doc.build(story)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Complete Action Plan (PDF)",
                data=pdf_buffer,
                file_name=f"LGU_Action_Plan_{storm_name}_{year}.pdf",
                mime="application/pdf",
                help="Download this action plan as PDF for offline reference"
            )
        except ImportError:
            # Fallback to text if reportlab not available
            full_text = f"""LGU Action Plan: {storm_name} ({year})

PRE-IMPACT PREPARATIONS (24-48 HOURS):
{lgu_insights['preparation']}

RESPONSE OPERATIONS (DURING IMPACT):
{lgu_insights['response']}

POST-IMPACT RECOVERY (FIRST 72 HOURS):
{lgu_insights['recovery']}

RESOURCE REQUIREMENTS:
{lgu_insights['resources']}
"""
            st.download_button(
                label="📥 Download Complete Action Plan (TXT)",
                data=full_text,
                file_name=f"LGU_Action_Plan_{storm_name}_{year}.txt",
                mime="text/plain",
                help="Download this action plan for offline reference"
            )
    # else:
    #     st.warning("⚠️ Could not generate LGU insights. Please check your OPENAI_API_KEY in .env file.")
    # 
    # st.markdown("---")
    
    # Get user province data if available
    user_province_data = None
    if selected_province:
        user_data = results_df[results_df['Province'] == selected_province]
        if not user_data.empty:
            user_province_data = user_data.iloc[0]
    
    # Display overall summary first
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk = len(results_df[results_df['Impact_Probability_Persons'] > 30])
        st.metric("High Risk Provinces (>30%)", high_risk)
    
    with col2:
        total_affected = results_df['Predicted_Affected_Persons'].sum()
        st.metric("Total People Affected", f"{total_affected:,.0f}")
    
    with col3:
        total_houses = results_df['Predicted_Houses_Damaged'].sum()
        st.metric("Total Houses Damaged", f"{total_houses:,.0f}")
    
    with col4:
        total_cost_usd = results_df['Predicted_Economic_Cost_USD'].sum()
        total_cost_php = total_cost_usd * USD_TO_PHP
        st.metric("Total Economic Cost", f"₱{total_cost_php/1e6:.1f}M")
    
    # Show detailed insights for selected province
    if user_province_data is not None:
        st.markdown("---")
        st.subheader(f"🎯 Impact on Your Province: **{selected_province}**")
        
        # Create columns for province-specific metrics
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        
        with pcol1:
            risk_cat = user_province_data['Risk_Category']
            risk_emoji = {
                'Low': '🟢',
                'Moderate': '🟡',
                'High': '🟠',
                'Very High': '🔴'
            }.get(risk_cat, '⚪')
            st.metric("Risk Level", f"{risk_emoji} {risk_cat}")
        
        with pcol2:
            prob = user_province_data['Impact_Probability_Persons']
            st.metric("Impact Probability", f"{prob:.1f}%")
        
        with pcol3:
            people = user_province_data['Predicted_Affected_Persons']
            st.metric("People Affected", f"{people:,.0f}")
        
        with pcol4:
            houses = user_province_data['Predicted_Houses_Damaged']
            st.metric("Houses Damaged", f"{houses:,.0f}")
        
        with pcol5:
            cost_usd = user_province_data['Predicted_Economic_Cost_USD']
            cost_php = cost_usd * USD_TO_PHP
            st.metric("Economic Cost", f"₱{cost_php/1e6:.2f}M")
        
        # Show province ranking
        province_rank = results_df[results_df['Province'] == selected_province].index[0] + 1
        total_provinces = len(results_df)
        st.write(f"📊 **Province Ranking**: #{province_rank} out of {total_provinces} provinces (ranked by impact probability)")
    
    st.markdown("---")
    
    # Dropdown 1: Top Provinces at Risk
    with st.expander("🔥 Top Provinces at Risk", expanded=True):
        display_cols = ['Province', 'Risk_Category', 'Impact_Probability_Persons', 
                       'Predicted_Affected_Persons', 'Predicted_Houses_Damaged', 'Predicted_Economic_Cost_USD']
        
        top_20 = results_df.head(20)[display_cols].copy()
        top_20['Predicted_Economic_Cost_PHP'] = top_20['Predicted_Economic_Cost_USD'].apply(lambda x: f"₱{x*USD_TO_PHP/1e6:.2f}M")
        top_20 = top_20.drop('Predicted_Economic_Cost_USD', axis=1)
        top_20.columns = ['Province', 'Risk', 'Impact %', 'People', 'Houses', 'Cost (PHP)']
        
        # Style the dataframe
        styled_df = top_20.style.background_gradient(subset=['Impact %'], cmap='Reds')
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=600
        )
    
    # Dropdown 2: Impact Analysis Charts
    with st.expander("📊 Impact Analysis Charts", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Chart 1: Top provinces bar chart - People
            fig1 = px.bar(
                results_df.head(15),
                x='Province',
                y='Predicted_Affected_Persons',
                color='Risk_Category',
                title='Top 15 Provinces: People Affected',
                color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
            )
            fig1.update_xaxes(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Chart 2: Houses damaged
            fig2 = px.bar(
                results_df.head(15),
                x='Province',
                y='Predicted_Houses_Damaged',
                color='Risk_Category',
                title='Top 15 Provinces: Houses Damaged',
                color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Economic cost
        results_df_chart = results_df.head(15).copy()
        results_df_chart['Cost_PHP'] = results_df_chart['Predicted_Economic_Cost_USD'] * USD_TO_PHP
        fig3 = px.bar(
            results_df_chart,
            x='Province',
            y='Cost_PHP',
            color='Risk_Category',
            title='Top 15 Provinces: Economic Cost (PHP)',
            color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
        )
        fig3.update_xaxes(tickangle=45)
        fig3.update_yaxes(title='Cost (PHP)')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Chart 4: Risk distribution
        risk_dist = results_df['Risk_Category'].value_counts()
        fig4 = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Distribution of Risk Levels',
            color=risk_dist.index,
            color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
        )
        st.plotly_chart(fig4, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="BARLO: Typhoon Impact Predictor",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for results persistence
    # Persistent memory that survives across reruns of the ap
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'storm_name' not in st.session_state:
        st.session_state.storm_name = None
    if 'year' not in st.session_state:
        st.session_state.year = None
    if 'track_df' not in st.session_state:
        st.session_state.track_df = None
    if 'prediction_time' not in st.session_state:
        st.session_state.prediction_time = None
    
    st.title("Typhoon Impact Predictor")
    st.markdown("""
    **AI-Powered Storm Impact Prediction System**  
    Predicts humanitarian and infrastructure impacts using real-time JTWC forecasts and trained ML models.
    """)
    
    # Location selector
    st.subheader("Enter your location")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_region = st.selectbox(
            "Select Your Region",
            options=REGION_NAMES,
            index=0,
            help="Choose your region in the Philippines"
        )
    
    with col2:
        # Get provinces for selected region
        provinces_in_region = PH_REGIONS[selected_region]
        selected_province = st.selectbox(
            "Select Your Province",
            options=provinces_in_region,
            index=0,
            help="Choose your province"
        )
    
    # Store selected province in session state for prediction summary
    if 'selected_province' not in st.session_state or st.session_state.selected_province != selected_province:
        st.session_state.selected_province = selected_province

        
    # Display map with selected province highlighted
    try:
        loc_df = pd.read_csv(DATA_PATH / "raw" / "Location_data" / "locations_latlng.csv")
        
        # Get coordinates for selected province
        province_data = loc_df[loc_df['Province'] == selected_province]
        
        if not province_data.empty:
            province_lat = province_data.iloc[0]['Lat']
            province_lng = province_data.iloc[0]['Lng']
                     
            # Create map centered on selected province with appropriate zoom
            # If predictions exist, zoom out to show storm track, otherwise zoom in on province
            if st.session_state.results_df is not None and st.session_state.track_df is not None:
                # Zoom out to show both province and storm track
                province_map = folium.Map(
                    location=[province_lat, province_lng],
                    zoom_start=7,
                    tiles='OpenStreetMap',
                    prefer_canvas=True
                )
            else:
                # Zoom in on selected province
                province_map = folium.Map(
                    location=[province_lat, province_lng],
                    zoom_start=9,
                    tiles='OpenStreetMap',
                    prefer_canvas=True
                )
            
            # Add marker for selected province (highlighted)
            folium.Marker(
                location=[province_lat, province_lng],
                popup=folium.Popup(
                    f"<b>YOUR LOCATION</b><br>{selected_province}<br>{selected_region}",
                    max_width=200
                ),
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(province_map)
            
            # Add circle to highlight the selected province
            folium.Circle(
                location=[province_lat, province_lng],
                radius=30000,  # 30km radius
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3,
                weight=3
            ).add_to(province_map)
            
            # OVERLAY: If ML predictions exist, add them to the map
            if st.session_state.results_df is not None and st.session_state.track_df is not None:
                with st.spinner("🗺️ Updating map with storm track and impact predictions..."):
                    results_df = st.session_state.results_df
                    track_df = st.session_state.track_df
                    
                    # Merge results with coordinates
                    map_data = results_df.merge(loc_df, left_on='Province', right_on='Province', how='left')
                
                # Add storm track
                track_points = []
                for idx, row in track_df.iterrows():
                    track_points.append([row['LAT'], row['LON']])
                    
                    # Calculate uncertainty radius
                    hours_ahead = idx * 6
                    if hours_ahead <= 24:
                        uncertainty_km = 50 + (hours_ahead / 24) * 50
                    elif hours_ahead <= 48:
                        uncertainty_km = 100 + ((hours_ahead - 24) / 24) * 100
                    elif hours_ahead <= 72:
                        uncertainty_km = 200 + ((hours_ahead - 48) / 24) * 100
                    else:
                        uncertainty_km = 300 + ((hours_ahead - 72) / 48) * 100
                    
                    uncertainty_km = min(uncertainty_km, 400)
                    
                    # Add uncertainty cone
                    folium.Circle(
                        location=[row['LAT'], row['LON']],
                        radius=uncertainty_km * 1000,
                        color='orange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.1,
                        weight=1,
                        opacity=0.3
                    ).add_to(province_map)

                    folium.map.Marker(
                        location=[row['LAT'], row['LON']],
                        icon=folium.DivIcon(
                            html=f'<div style="color: black">{hours_ahead}h</div>'
                        )
                    ).add_to(province_map)
                    
                    # Add forecast point marker
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=5,
                        color='darkorange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.9,
                        weight=2,
                        popup=folium.Popup(
                            f"<b>Forecast Point {idx + 1}</b><br>"
                            f"Time: +{hours_ahead}h<br>"
                            f"Intensity: {row.get('INTENSITY', 'N/A')} kt<br>"
                            f"Uncertainty: ±{uncertainty_km:.0f} km",
                            max_width=200
                        )
                    ).add_to(province_map)
                
                # Draw storm track line
                if len(track_points) > 1:
                    folium.PolyLine(
                        locations=track_points,
                        color='orange',
                        weight=3,
                        opacity=0.8,
                        popup="Storm Track"
                    ).add_to(province_map)
                
                # Add province risk markers
                for _, row in map_data.iterrows():
                    if pd.notna(row.get('Lat')) and pd.notna(row.get('Lng')):
                        prob = row['Impact_Probability_Persons']
                        
                        # Skip selected province (already highlighted)
                        if row['Province'] == selected_province:
                            continue
                        
                        # Determine color and size based on probability
                        # 30% is already high-risk (red)
                        if prob > 50:
                            color = 'darkred'
                            radius = 10
                        elif prob > 30:
                            color = 'red'
                            radius = 9
                        elif prob > 20:
                            color = 'orange'
                            radius = 7
                        elif prob > 10:
                            color = 'yellow'
                            radius = 5
                        else:
                            color = 'lightgray'
                            radius = 4
                        
                        # Add province marker
                        folium.CircleMarker(
                            location=[row['Lat'], row['Lng']],
                            radius=radius,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.6,
                            popup=folium.Popup(
                                f"<b>{row['Province']}</b><br>"
                                f"Risk: {row['Risk_Category']}<br>"
                                f"Impact Prob: {prob:.1f}%<br>"
                                f"People: {row['Predicted_Affected_Persons']:,.0f}",
                                max_width=250
                            )
                            ).add_to(province_map)
            
            # Display the map with proper sizing
            try:
                st_folium(province_map, width=1200, height=600, returned_objects=[])
            except Exception as e:
                st.error(f"Map rendering error: {str(e)}")
                st.info("Map data is available but there was an issue displaying it. Try refreshing the page.")
            
            # Historical Storm Insights for Selected Province
            st.subheader(f"📊 Historical Storm Impact: {selected_province} (2010-2024)")
            
            # Load impact data for the province
            try:
                impact_df = pd.read_csv(DATA_PATH / "raw" / "Impact_data" / "people_affected_all_years.csv")
                province_impacts = impact_df[impact_df['Province'] == selected_province].copy()
                
                if not province_impacts.empty:
                    # Load storm summary to get category information
                    storm_df = load_historical_storm_data()
                    
                    # Merge to get storm details
                    if storm_df is not None:
                        province_impacts = province_impacts.merge(
                            storm_df[['Year', 'PH_Name', 'Peak_Category', 'Peak_Windspeed_kmh']],
                            left_on=['Year', 'Storm'],
                            right_on=['Year', 'PH_Name'],
                            how='left'
                        )
                    
                    # Calculate statistics
                    total_storms = len(province_impacts)
                    total_affected = province_impacts['Affected'].sum()
                    avg_affected = province_impacts['Affected'].mean()
                    max_affected = province_impacts['Affected'].max()
                    max_idx = province_impacts['Affected'].idxmax()
                    worst_storm = str(province_impacts.loc[max_idx, 'Storm'])
                    worst_year_raw = province_impacts.loc[max_idx, 'Year']
                    _year_numeric = pd.to_numeric(worst_year_raw, errors='coerce')
                    worst_year = int(_year_numeric) if not pd.isna(_year_numeric) else 0
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Storms Impacted", total_storms)
                    
                    with col2:
                        st.metric("Total People Affected", f"{total_affected:,.0f}")
                    
                    with col3:
                        st.metric("Avg People Affected per Storm", f"{avg_affected:,.0f}")
                    
                    with col4:
                        st.metric("Worst Storm", f"{worst_storm} ({worst_year})")
                    
                    # Note: Using expanders instead of tabs below
                    
                    # Impact Over Time Expander
                    with st.expander("📈 Impact Over Time", expanded=False):
                        # Timeline of impacts
                        yearly_impact = province_impacts.groupby('Year').agg({
                            'Affected': 'sum',
                            'Storm': 'count'
                        }).reset_index()
                        yearly_impact.columns = ['Year', 'Total_Affected', 'Storm_Count']
                        
                        # Line graph for people affected over time
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=yearly_impact['Year'],
                            y=yearly_impact['Total_Affected'],
                            mode='lines+markers',
                            name='People Affected',
                            line=dict(color='indianred', width=3),
                            marker=dict(size=8, color='darkred'),
                            hovertemplate='Year: %{x}<br>Affected: %{y:,.0f}<extra></extra>'
                        ))
                        
                        fig1.update_layout(
                            title=f'People Affected in {selected_province} by Year',
                            xaxis_title='Year',
                            yaxis_title='People Affected',
                            hovermode='x unified',
                            showlegend=False
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Line graph for number of storms per year
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=yearly_impact['Year'],
                            y=yearly_impact['Storm_Count'],
                            mode='lines+markers',
                            name='Number of Storms',
                            line=dict(color='orange', width=3),
                            marker=dict(size=8, color='darkorange'),
                            hovertemplate='Year: %{x}<br>Storms: %{y}<extra></extra>'
                        ))
                        
                        fig2.update_layout(
                            title=f'Number of Storms Impacting {selected_province} per Year',
                            xaxis_title='Year',
                            yaxis_title='Number of Storms',
                            hovermode='x unified',
                            showlegend=False
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with st.expander("🌪️ Worst Storms", expanded=False):
                        # Top 10 worst storms
                        st.write(f"**Top 10 Most Damaging Storms in {selected_province}:**")
                        
                        worst_storms = province_impacts.nlargest(10, 'Affected')[
                            ['Year', 'Storm', 'Peak_Category', 'Affected', 'Peak_Windspeed_kmh']
                        ].copy()
                        
                        # Format for display
                        st.dataframe(
                            worst_storms,
                            hide_index=True,
                            column_config={
                                'Year': 'Year',
                                'Storm': 'Storm Name',
                                'Peak_Category': 'Category',
                                'Affected': st.column_config.NumberColumn(
                                    'People Affected',
                                    format='%d'
                                ),
                                'Peak_Windspeed_kmh': st.column_config.NumberColumn(
                                    'Wind Speed',
                                    format='%d km/h'
                                )
                            },
                            use_container_width=True
                        )
                        
                        # Chart of worst storms
                        fig3 = px.bar(
                            worst_storms.head(10),
                            x='Storm',
                            y='Affected',
                            color='Peak_Category',
                            title=f'Top 10 Most Damaging Storms',
                            labels={'Affected': 'People Affected'}
                        )
                        fig3.update_xaxes(tickangle=45)
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with st.expander(" Statistics", expanded=False):
                        # Statistical summary
                        st.write(f"**Impact Statistics for {selected_province}:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Storms (2010-2024)", total_storms)
                            st.metric("Total People Affected", f"{total_affected:,.0f}")
                            st.metric("Average Affected per Storm", f"{avg_affected:,.0f}")
                            st.metric("Maximum in Single Storm", f"{max_affected:,.0f}")
                        
                        with col2:
                            # Category breakdown
                            if 'Peak_Category' in province_impacts.columns:
                                category_dist = province_impacts['Peak_Category'].value_counts()
                                
                                st.write("**Storm Categories:**")
                                for cat, count in category_dist.items():
                                    if cat is not None and str(cat) != 'nan':
                                        st.write(f"- {cat}: {count}")
                            
                            # Calculate risk level
                            avg_storms_per_year = total_storms / 15  # 2010-2024 = 15 years
                            st.metric("Average Storms per Year", f"{avg_storms_per_year:.1f}")
                
                else:
                    st.info(f"ℹ️ No historical storm impact data found for {selected_province} in our records (2010-2024).")
                    st.write("This could mean:")
                    st.write("- The province has been fortunate to avoid major impacts")
                    st.write("- Impact data was not recorded for this location")
                    st.write("- The province name may differ in historical records")
            
            except Exception as e:
                st.warning(f"⚠️ Could not load historical impact data: {str(e)}")
        
        else:
            st.warning(f"⚠️ Coordinates not found for {selected_province}")
    except Exception as e:
        st.warning(f"Could not load map: {str(e)}")
    
    # Display ML prediction results if available (OUTSIDE map/historical data section)
    if st.session_state.results_df is not None:
        st.markdown("---")
        display_ml_results(
            st.session_state.results_df,
            st.session_state.storm_name,
            st.session_state.year,
            st.session_state.track_df
        )
    
    st.markdown("---")
    
    if not MODEL_AVAILABLE:
        st.error(f"⚠️ Model pipeline not found. Import error: {IMPORT_ERROR}")
        st.info("""
        **Troubleshooting:**
        - Make sure you're running from the project root directory
        - Check that `model/Final_Transform/pipeline/` exists
        - Verify Python files: `parse_jtwc_forecast.py`, `fetch_forecast_weather.py`, `unified_pipeline.py`
        """)
        return
    
    # Sidebar storm input
    st.sidebar.header("🌪️ Storm Forecast")
    
    # Initialize session state for trigger
    if 'trigger_prediction' not in st.session_state:
        st.session_state.trigger_prediction = False
    
    # Load storm list
    try:
        storm_list_file = DATA_PATH / "samples" / "storm_list.json"
        if storm_list_file.exists():
            with open(storm_list_file, 'r') as f:
                storm_config = json.load(f)
                recent_storms = storm_config['recent_storms']
        else:
            recent_storms = []
    except Exception as e:
        st.sidebar.error(f"Could not load storm list: {e}")
        recent_storms = []
    
    # Initialize variables
    storm_bulletin = None
    storm_id = None
    selected_storm_name = None
    selected_storm_date = None
    run_prediction = False
    is_live_source = False
    # Section 2: Live Forecast
    st.sidebar.subheader("🌐 Live Forecast")
    
    if st.sidebar.button("📡 Obtain Live Forecast", type="secondary", use_container_width=True):
        # Try to load a live forecast file (you can create one for current active storms)
        live_forecast_file = DATA_PATH / "samples" / "live_forecast.txt"
        
        if live_forecast_file.exists():
            with open(live_forecast_file, 'r') as f:
                storm_bulletin = f.read()
            
            # Parse to get storm name
            try:
                temp_file = Path("temp_bulletin.txt")
                with open(temp_file, 'w') as f:
                    f.write(storm_bulletin)
                temp_track = parse_jtwc_forecast(str(temp_file))
                temp_file.unlink()
                
                live_storm_name = temp_track['PHNAME'].iloc[0] if 'PHNAME' in temp_track.columns else "MARIA"
                live_storm_date = datetime.now().strftime("%b %d, %Y")
                
                selected_storm_name = live_storm_name
                selected_storm_date = live_storm_date
                
                st.sidebar.success(f"🔴 **LIVE: Typhoon {live_storm_name}**\n\n{live_storm_date}")
                run_prediction = True
                is_live_source = True
            except Exception as e:
                st.sidebar.error(f"Error parsing live forecast: {e}")
        elif not live_forecast_file.exists():
            # No local live forecast file; try discovering a live JTWC product by iterating ids
            with st.spinner("🔎 Searching for latest JTWC live bulletin..."):
                storm_id_found, text = cached_try_fetch_jtwc_bulletin()
            if storm_id_found and text:
                storm_bulletin = text
                selected_storm_name = None  # inferred later from parsed track
                selected_storm_date = datetime.now().strftime("%b %d, %Y")
                st.sidebar.success(f"Found live JTWC bulletin: {storm_id_found}web.txt")
                run_prediction = True
                is_live_source = True
            else:
                st.sidebar.info("No current storm in the Philippines.")
                storm_bulletin = None
        else:
            # No live forecast file, use the most recent historical storm
            if recent_storms:
                # Get the most recent storm (first in list)
                latest_storm = recent_storms[0]
                
                forecast_file = DATA_PATH / latest_storm['file']
                if forecast_file.exists():
                    with open(forecast_file, 'r') as f:
                        storm_bulletin = f.read()
                    selected_storm_name = latest_storm['ph_name']
                    selected_storm_date = latest_storm['date']
                    
                    st.sidebar.success(f"📡 **Latest Storm: {latest_storm['ph_name']}**\n\n{latest_storm['category']} | {latest_storm['date']}")
                    run_prediction = True
                else:
                    st.sidebar.error("❌ Could not load latest storm data")
            else:
                st.sidebar.warning("⚠️ No storm data available")
    
    # Section 1: Historical Storms
    st.sidebar.subheader("📋 Historical Storms")
    
    if recent_storms:
        # Sort storms by date (most recent first)
        sorted_storms = sorted(recent_storms, key=lambda x: x['date'], reverse=True)
        
        # Create dropdown options with storm names and dates
        storm_options = [f"{s['ph_name']} ({s['date']})" for s in sorted_storms]
        selected_storm_display = st.sidebar.selectbox(
            "Select a storm to analyze",
            options=storm_options,
            index=0,
            help="Choose from recent Philippine storms (sorted by date, newest first)"
        )
        
        # Extract the selected storm data
        selected_index = storm_options.index(selected_storm_display)
        storm_data = sorted_storms[selected_index]
        
        # Analyze Impact button (only load bulletin on click to avoid overriding LIVE)
        forecast_file = DATA_PATH / storm_data['file']
        if st.sidebar.button("🚀 Analyze Impact", type="primary", use_container_width=True):
            if forecast_file.exists():
                with open(forecast_file, 'r') as f:
                    storm_bulletin = f.read()
                selected_storm_name = storm_data['ph_name']
                selected_storm_date = storm_data['date']
                run_prediction = True
            else:
                st.sidebar.error("❌ Could not load selected storm bulletin")
    else:
        st.sidebar.warning("⚠️ No historical storms available")
    
    st.sidebar.markdown("---")

    # Section 3: Advanced Options (collapsible)
    with st.sidebar.expander("⚙️ Advanced Options"):
        st.write("**Manual Input Methods:**")
        
        manual_method = st.radio(
            "Choose input method",
            ["JTWC Storm ID", "Paste Bulletin Text"],
            label_visibility="collapsed"
        )
        
        if manual_method == "JTWC Storm ID":
            storm_id = st.text_input(
                "Enter Storm ID",
                value="wp3025",
                help="Format: wp[number][year] (e.g., wp3025)"
            )
            
            if st.button("Fetch Storm Data"):
                run_prediction = True
                selected_storm_name = storm_id.upper()
        
        elif manual_method == "Paste Bulletin Text":
            st.info("Paste JTWC bulletin from https://www.jtwc.navy.mil/")
            storm_bulletin = st.text_area(
                "JTWC Bulletin",
                height=200,
                placeholder="Paste full JTWC warning text here..."
            )
            
            if st.button("Analyze Bulletin"):
                if storm_bulletin:
                    run_prediction = True
                    selected_storm_name = "CUSTOM"
                else:
                    st.warning("Please paste bulletin text first")
    
    # Run prediction if triggered
    if run_prediction:
        # Create a single progress container in the sidebar
        progress_container = st.sidebar.empty()
        
        with progress_container:
            with st.spinner("🔄 Loading ML models..."):
                models = load_ml_models()
        
        if models is None:
            st.sidebar.error("❌ Failed to load models")
            return
        
        try:
            # STEP 1: Parse storm forecast
            with progress_container:
                with st.spinner("📡 Step 1/4: Analyzing storm track and intensity..."):
                    if storm_bulletin:
                        # Save bulletin to temp file
                        temp_file = Path("temp_bulletin.txt")
                        with open(temp_file, 'w') as f:
                            f.write(storm_bulletin)
                        track_df = parse_jtwc_forecast(str(temp_file))
                        temp_file.unlink()  # Delete temp file
                    elif storm_id:
                        # Fetch live bulletin by storm_id from JTWC with robust error handling
                        import requests as _requests
                        url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
                        try:
                            _resp = _requests.get(url, timeout=12)
                            _resp.raise_for_status()
                        except _requests.exceptions.RequestException as e:
                            st.sidebar.error(f"Failed to fetch JTWC bulletin for {storm_id}: {e}")
                            return
                        temp_file = Path("temp_bulletin.txt")
                        with open(temp_file, 'w') as f:
                            f.write(_resp.text)
                        track_df = parse_jtwc_forecast(str(temp_file))
                        temp_file.unlink()
                    else:
                        st.sidebar.error("❌ No storm data provided")
                        return

                    # Region filter: Only proceed if track is within our AOI (avoid Africa etc.)
                    # Define a conservative Western Pacific/Philippines bounding box
                    # Latitude: -5 to 35 N, Longitude: 105 to 155 E
                    LAT_MIN, LAT_MAX = -5.0, 35.0
                    LON_MIN, LON_MAX = 105.0, 155.0

                    if not {'LAT', 'LON'}.issubset(set(track_df.columns)):
                        st.sidebar.error("Parsed track missing LAT/LON columns; cannot apply region filter.")
                        return

                    lat_min = float(track_df['LAT'].min())
                    lat_max = float(track_df['LAT'].max())
                    lon_min = float(track_df['LON'].min())
                    lon_max = float(track_df['LON'].max())

                    overlaps_lat = (lat_max >= LAT_MIN) and (lat_min <= LAT_MAX)
                    overlaps_lon = (lon_max >= LON_MIN) and (lon_min <= LON_MAX)
                    if not (overlaps_lat and overlaps_lon):
                        if is_live_source:
                            st.sidebar.info("No current storm in the Philippines.")
                        else:
                            st.sidebar.warning(
                                "⛔ Storm track is outside the Western Pacific/Philippines region. Skipping processing.")
                        return
                    
                    # Extract storm info from track dataframe
                    year = int(track_df['SEASON'].iloc[0]) if 'SEASON' in track_df.columns else datetime.now().year
                    
                    # Use the name from sidebar first, then try parsing from bulletin
                    if selected_storm_name and selected_storm_name not in ["CUSTOM", ""]:
                        storm_name = selected_storm_name
                    elif 'PHNAME' in track_df.columns and pd.notna(track_df['PHNAME'].iloc[0]):
                        storm_name = track_df['PHNAME'].iloc[0]
                    elif 'NAME' in track_df.columns and pd.notna(track_df['NAME'].iloc[0]):
                        storm_name = track_df['NAME'].iloc[0]
                    else:
                        storm_name = f"Storm-{year}-{datetime.now().strftime('%m%d')}"
                    
                    # CRITICAL: Add storm name and year to track_df NOW (before any processing)
                    track_df['PHNAME'] = storm_name
                    track_df['SEASON'] = year
            
            st.sidebar.success(f"✅ Storm: {storm_name} ({year})")
            
            # STEP 2: Fetch weather forecasts
            with progress_container:
                with st.spinner("🌦️ Step 2/4: Gathering weather data for all provinces..."):
                    # Extract date range from track
                    start_date = track_df['datetime'].min().strftime('%Y-%m-%d')
                    end_date = track_df['datetime'].max().strftime('%Y-%m-%d')
                    
                    # Call weather forecast with correct parameters
                    weather_df = fetch_weather_forecast(
                        start_date=start_date,
                        end_date=end_date,
                        locations_file=str(DATA_PATH / "raw" / "Location_data" / "locations_latlng.csv")
                    )
                    
                    # Standardize column name for consistency
                    if 'province' in weather_df.columns:
                        weather_df.rename(columns={'province': 'Province'}, inplace=True)
            
            # STEP 3: Engineer features
            with progress_container:
                with st.spinner("🔧 Step 3/4: Engineering storm impact features..."):
                    # Save temporary files in expected locations
                    import shutil
                    
                    # Create temp directories
                    temp_weather_dir = DATA_PATH / "raw" / "Weather_location_data" / str(year)
                    temp_weather_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save weather data (convert Province back to lowercase for pipeline compatibility)
                    weather_file = temp_weather_dir / f"{year}_{storm_name}.csv"
                    weather_to_save = weather_df.copy()
                    if 'Province' in weather_to_save.columns:
                        weather_to_save.rename(columns={'Province': 'province'}, inplace=True)
                    weather_to_save.to_csv(weather_file, index=False)
                    
                    # Append track to storm data temporarily
                    storm_data_file = DATA_PATH / "raw" / "Storm_data" / "ph_storm_data.csv"
                    original_storm_data = pd.read_csv(storm_data_file)
                    
                    # track_df already has PHNAME and SEASON set earlier
                    combined_storm_data = pd.concat([original_storm_data, track_df], ignore_index=True)
                    
                    # Backup and save
                    backup_file = storm_data_file.with_suffix('.csv.backup')
                    shutil.copy(storm_data_file, backup_file)
                    combined_storm_data.to_csv(storm_data_file, index=False)
                    
                    try:
                        # Run feature pipeline
                        pipeline = StormFeaturePipeline(base_dir=DATA_PATH / "raw")
                        features_df = pipeline.process_storm(year, storm_name, verbose=False)
                    finally:
                        # Restore original storm data
                        shutil.copy(backup_file, storm_data_file)
                        backup_file.unlink()
                        # Clean up temp weather file
                        if weather_file.exists():
                            weather_file.unlink()
            
            # STEP 4: Make predictions
            with progress_container:
                with st.spinner("🤖 Step 4/4: Computing impact predictions..."):
                    # SAFETY CHECK: Distance-based filter
                    if 'min_distance_km' in features_df.columns:
                        closest_distance = features_df['min_distance_km'].min()
                        
                        if closest_distance > 500:
                            st.sidebar.warning(f"⚠️ Storm is over {closest_distance:.0f} km from Philippines")
                            
                            # Create zero-impact results
                            results_df = pd.DataFrame({
                                'Province': features_df['Province'].astype(str),
                                'Impact_Probability_Persons': 0.0,
                                'Predicted_Affected_Persons': 0,
                                'Impact_Probability_Houses': 0.0,
                                'Predicted_Houses_Damaged': 0,
                                'Impact_Probability_Cost': 0.0,
                                'Predicted_Economic_Cost_USD': 0
                            })
                            results_df['Risk_Category'] = 'Low'
                            
                            # Store in session state
                            st.session_state.results_df = results_df
                            st.session_state.storm_name = storm_name
                            st.session_state.year = year
                            st.session_state.track_df = track_df
                            st.session_state.prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            progress_container.empty()
                            st.sidebar.success("✅ Analysis complete - No impact predicted")
                            return  # Skip ML prediction
                    
                    # The minimal models only need 6 features (no prefixes or complex engineering)
                    required_features = models['features']
                    
                    # Check which features we have
                    available_features = [f for f in required_features if f in features_df.columns]
                    missing_features = [f for f in required_features if f not in features_df.columns]
                    
                    if missing_features:
                        # Use sensible defaults based on training data medians
                        defaults = {
                            'min_distance_km': 150.0,
                            'max_wind_gust_kmh': 80.0,
                            'total_precipitation_mm': 100.0,
                            'max_wind_in_track_kt': 60.0,
                            'hours_under_100km': 12.0,
                            'Population': 1000000
                        }
                        for feat in missing_features:
                            features_df[feat] = defaults.get(feat, 0)
                    
                    # Select only the 6 required features in correct order
                    X = features_df[required_features].copy()
                    
                    # Handle any remaining NaN values
                    X = X.fillna(X.median())
                    
                    # Make predictions for ALL THREE impact types
                    predictions = {}
                    
                    for impact_type in ['people', 'houses', 'cost']:
                        # Get probability
                        prob = models[impact_type]['classifier'].predict_proba(X)[:, 1]
                        
                        # Always predict values (not just for high probability)
                        # Use a lower threshold (10%) for when to apply predictions
                        pred_values = np.zeros(len(X))
                        prediction_mask = prob > 0.10  # Lower threshold - predict if >10% probability
                        
                        if prediction_mask.sum() > 0:
                            pred_log = models[impact_type]['regressor'].predict(X[prediction_mask])
                            pred_values[prediction_mask] = np.expm1(pred_log)
                        
                        # For very low probabilities, scale down the prediction
                        pred_values = pred_values * (prob / prob.max())  # Scale by relative probability
                        
                        predictions[impact_type] = {
                            'probability': prob,
                            'values': pred_values
                        }
            
            # STEP 5: Create results dataframe
            results_df = pd.DataFrame({
                'Province': features_df['Province'].astype(str),
                'Impact_Probability_Persons': predictions['people']['probability'] * 100,
                'Predicted_Affected_Persons': predictions['people']['values'].astype(int),
                'Impact_Probability_Houses': predictions['houses']['probability'] * 100,
                'Predicted_Houses_Damaged': predictions['houses']['values'].astype(int),
                'Impact_Probability_Cost': predictions['cost']['probability'] * 100,
                'Predicted_Economic_Cost_USD': predictions['cost']['values'].astype(int)
            })
            
            # Add risk category (based on people probability)
            # Lowered thresholds: 30% is already High Risk
            results_df['Risk_Category'] = pd.cut(
                results_df['Impact_Probability_Persons'],
                bins=[0, 10, 20, 30, 100],
                labels=['Low', 'Moderate', 'High', 'Very High']
            ).astype(str)  # Convert to string to avoid plotly categorical errors
            
            results_df = results_df.sort_values('Impact_Probability_Persons', ascending=False)
            
            # Store results in session state with timestamp
            st.session_state.results_df = results_df
            st.session_state.storm_name = storm_name
            st.session_state.year = year
            st.session_state.track_df = track_df
            st.session_state.prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Clear progress indicator and show success
            progress_container.empty()
            st.sidebar.success("✅ Predictions complete!")
            
            # Trigger rerun to display results immediately
            st.rerun()
            
        except Exception as e:
            progress_container.empty()
            st.sidebar.error(f"❌ Prediction error: {str(e)}")
            import traceback
            st.sidebar.code(traceback.format_exc())

    # Clear results button (only show if results exist)
    if st.session_state.results_df is not None:
        if st.sidebar.button("🗑️ Clear Results", help="Clear current predictions to run a new storm"):
            st.session_state.results_df = None
            st.session_state.storm_name = None
            st.session_state.year = None
            st.session_state.track_df = None
            st.rerun()


if __name__ == "__main__":
    main()




