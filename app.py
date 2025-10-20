import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    calculate_storm_impact, 
    create_philippines_map, 
    create_impact_charts,
    validate_data_format
)
import io


def main():
    """Main Streamlit application for Philippines Storm Impact Analysis."""
    
    st.set_page_config(
        page_title="Philippines Storm Impact Analyzer",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌪️ Philippines Storm Impact Analyzer")
    st.markdown("""
    This application analyzes the economic impact of storms on Philippine provinces.
    Upload your data files or use the sample data to get started.
    """)
    
    # Sidebar for data upload and controls
    st.sidebar.header("📊 Data Input")
    
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