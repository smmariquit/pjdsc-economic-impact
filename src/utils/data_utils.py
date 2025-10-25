import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, List
import math


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
    
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r


def calculate_storm_impact(storm_data: pd.DataFrame, provinces_data: pd.DataFrame, 
                          max_impact_distance: float = 200) -> pd.DataFrame:
    """
    Calculate economic impact of storm on provinces based on proximity and intensity.
    
    Args:
        storm_data: DataFrame with columns ['lat', 'lon', 'intensity', 'timestamp']
        provinces_data: DataFrame with columns ['province', 'lat', 'lon', 'population', 'gdp_per_capita']
        max_impact_distance: Maximum distance (km) for storm impact consideration
    
    Returns:
        DataFrame with provinces and their calculated impact metrics
    """
    impact_results = []
    
    for _, province in provinces_data.iterrows():
        max_impact = 0
        closest_distance = float('inf')
        
        # Calculate impact from each storm data point
        for _, storm_point in storm_data.iterrows():
            distance = calculate_distance(
                province['lat'], province['lon'],
                storm_point['lat'], storm_point['lon']
            )
            
            if distance <= max_impact_distance:
                # Impact decreases exponentially with distance
                distance_factor = np.exp(-distance / 100)  # 100km decay constant
                
                # Impact increases with storm intensity
                intensity_factor = storm_point['intensity'] / 100  # Normalize intensity
                
                # Calculate impact score (0-1)
                impact_score = distance_factor * intensity_factor
                
                if impact_score > max_impact:
                    max_impact = impact_score
                    closest_distance = distance
        
        # Calculate economic impact
        population = province.get('population', 1000000)  # Default if missing
        gdp_per_capita = province.get('gdp_per_capita', 3000)  # Default if missing
        
        # Economic impact as percentage of local GDP
        economic_impact_pct = max_impact * 0.3  # Max 30% GDP impact
        economic_impact_usd = population * gdp_per_capita * economic_impact_pct
        
        impact_results.append({
            'province': province['province'],
            'lat': province['lat'],
            'lon': province['lon'],
            'population': population,
            'gdp_per_capita': gdp_per_capita,
            'impact_score': max_impact,
            'closest_distance_km': closest_distance if closest_distance != float('inf') else None,
            'economic_impact_pct': economic_impact_pct,
            'economic_impact_usd': economic_impact_usd,
            'affected': max_impact > 0.1  # Threshold for being considered "affected"
        })
    
    return pd.DataFrame(impact_results)


def create_philippines_map(provinces_data: pd.DataFrame, storm_data: pd.DataFrame = None, 
                          impact_data: pd.DataFrame = None) -> folium.Map:
    """
    Create an interactive map of the Philippines with provinces and storm data.
    
    Args:
        provinces_data: DataFrame with province information
        storm_data: Optional DataFrame with storm forecast points
        impact_data: Optional DataFrame with calculated impact data
    
    Returns:
        Folium map object
    """
    # Center on Philippines
    center_lat = 12.8797
    center_lon = 121.7740
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add provinces
    if impact_data is not None:
        # Color provinces by impact level
        for _, row in impact_data.iterrows():
            if row['impact_score'] > 0:
                color = 'red' if row['impact_score'] > 0.5 else 'orange' if row['impact_score'] > 0.2 else 'yellow'
                radius = 8 + row['impact_score'] * 12  # Scale marker size by impact
            else:
                color = 'green'
                radius = 6
            
            distance_text = f"{row['closest_distance_km']:.1f}" if pd.notna(row['closest_distance_km']) else 'N/A'
            popup_text = f"""
            <b>{row['province']}</b><br>
            Population: {row['population']:,}<br>
            GDP per capita: ${row['gdp_per_capita']:,}<br>
            Impact Score: {row['impact_score']:.3f}<br>
            Economic Impact: ${row['economic_impact_usd']:,.0f}<br>
            Distance to Storm: {distance_text} km
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    else:
        # Default province markers
        for _, row in provinces_data.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,
                popup=f"<b>{row['province']}</b>",
                color='blue',
                weight=2,
                fillColor='lightblue',
                fillOpacity=0.7
            ).add_to(m)
    
    # Add storm track if provided
    if storm_data is not None and len(storm_data) > 0:
        # Storm path
        if len(storm_data) > 1:
            storm_coords = [[row['lat'], row['lon']] for _, row in storm_data.iterrows()]
            folium.PolyLine(
                locations=storm_coords,
                color='red',
                weight=3,
                opacity=0.8,
                popup="Storm Track"
            ).add_to(m)
        
        # Storm points with intensity
        for _, row in storm_data.iterrows():
            intensity = row.get('intensity', 50)
            radius = 5 + (intensity / 100) * 15  # Scale by intensity
            
            popup_text = f"""
            <b>Storm Point</b><br>
            Intensity: {intensity}<br>
            Coordinates: ({row['lat']:.2f}, {row['lon']:.2f})
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=200),
                color='darkred',
                weight=2,
                fillColor='red',
                fillOpacity=0.8
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Legend</b><br>
    <span style="color:green">●</span> No Impact<br>
    <span style="color:yellow">●</span> Low Impact<br>
    <span style="color:orange">●</span> Medium Impact<br>
    <span style="color:red">●</span> High Impact<br>
    <span style="color:darkred">●</span> Storm Points
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def create_impact_charts(impact_data: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Create charts showing economic impact analysis.
    
    Args:
        impact_data: DataFrame with calculated impact data
    
    Returns:
        Tuple of (bar_chart, scatter_plot) plotly figures
    """
    # Filter to affected provinces only
    affected = impact_data[impact_data['affected'] == True].copy()
    
    if len(affected) == 0:
        # Create empty charts if no provinces affected
        fig1 = go.Figure()
        fig1.add_annotation(text="No provinces significantly affected", 
                           x=0.5, y=0.5, showarrow=False)
        fig1.update_layout(title="Economic Impact by Province")
        
        fig2 = go.Figure()
        fig2.add_annotation(text="No impact data to display", 
                           x=0.5, y=0.5, showarrow=False)
        fig2.update_layout(title="Impact Score vs Distance")
        
        return fig1, fig2
    
    # Sort by economic impact for better visualization
    affected = affected.sort_values('economic_impact_usd', ascending=True)
    
    # Bar chart of economic impact
    fig1 = px.bar(
        affected.tail(15),  # Top 15 most affected
        x='economic_impact_usd',
        y='province',
        orientation='h',
        title='Economic Impact by Province (Top 15 Most Affected)',
        labels={'economic_impact_usd': 'Economic Impact (USD)', 'province': 'Province'},
        color='impact_score',
        color_continuous_scale='Reds'
    )
    fig1.update_layout(height=600)
    
    # Scatter plot of impact vs distance
    fig2 = px.scatter(
        affected,
        x='closest_distance_km',
        y='impact_score',
        size='population',
        color='economic_impact_usd',
        hover_name='province',
        title='Storm Impact Score vs Distance from Storm',
        labels={
            'closest_distance_km': 'Distance from Storm (km)',
            'impact_score': 'Impact Score',
            'economic_impact_usd': 'Economic Impact (USD)'
        },
        color_continuous_scale='Reds'
    )
    
    return fig1, fig2


def validate_data_format(df: pd.DataFrame, data_type: str) -> Tuple[bool, List[str]]:
    """
    Validate the format of uploaded data.
    
    Args:
        df: DataFrame to validate
        data_type: Either 'provinces' or 'forecast'
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if data_type == 'provinces':
        required_cols = ['province', 'lat', 'lon']
        optional_cols = ['population', 'gdp_per_capita']
        
        # Check required columns
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check data types and ranges
        if 'lat' in df.columns:
            if not df['lat'].dtype in [np.float64, np.int64]:
                try:
                    df['lat'] = pd.to_numeric(df['lat'])
                except:
                    errors.append("Latitude column must contain numeric values")
            if df['lat'].min() < -90 or df['lat'].max() > 90:
                errors.append("Latitude values must be between -90 and 90")
        
        if 'lon' in df.columns:
            if not df['lon'].dtype in [np.float64, np.int64]:
                try:
                    df['lon'] = pd.to_numeric(df['lon'])
                except:
                    errors.append("Longitude column must contain numeric values")
            if df['lon'].min() < -180 or df['lon'].max() > 180:
                errors.append("Longitude values must be between -180 and 180")
    
    elif data_type == 'forecast':
        required_cols = ['lat', 'lon', 'intensity']
        
        # Check required columns
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check data types and ranges
        if 'lat' in df.columns:
            if not df['lat'].dtype in [np.float64, np.int64]:
                try:
                    df['lat'] = pd.to_numeric(df['lat'])
                except:
                    errors.append("Latitude column must contain numeric values")
        
        if 'lon' in df.columns:
            if not df['lon'].dtype in [np.float64, np.int64]:
                try:
                    df['lon'] = pd.to_numeric(df['lon'])
                except:
                    errors.append("Longitude column must contain numeric values")
        
        if 'intensity' in df.columns:
            if not df['intensity'].dtype in [np.float64, np.int64]:
                try:
                    df['intensity'] = pd.to_numeric(df['intensity'])
                except:
                    errors.append("Intensity column must contain numeric values")
            if df['intensity'].min() < 0 or df['intensity'].max() > 200:
                errors.append("Intensity values should be between 0 and 200")
    
    return len(errors) == 0, errors