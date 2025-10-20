# 🌪️ Philippines Storm Impact Analyzer

A comprehensive Streamlit web application for analyzing and visualizing the economic impact of storms on Philippine provinces. This tool combines meteorological forecast data with geographic and economic information to estimate potential damage and affected areas.

![Philippines Storm Impact Analyzer](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 🎯 Features

- **Interactive Philippines Map**: Visualize provinces, storm tracks, and impact zones
- **Economic Impact Modeling**: Calculate estimated economic losses based on storm intensity and proximity
- **Real-time Analysis**: Upload custom forecast data or use sample datasets
- **Multiple Visualizations**: Charts, maps, and tables for comprehensive analysis
- **Data Export**: Download analysis results in CSV format
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this project**:
   ```bash
   git clone <repository-url>
   cd pjdsc
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**:
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in your terminal

## 📊 Data Formats

### Province Data (CSV)

The province data should contain information about Philippine administrative divisions:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `province` | String | ✅ | Name of the province |
| `lat` | Float | ✅ | Latitude in decimal degrees |
| `lon` | Float | ✅ | Longitude in decimal degrees |
| `population` | Integer | ❌ | Population count (default: 1,000,000) |
| `gdp_per_capita` | Float | ❌ | GDP per capita in USD (default: $3,000) |

**Example:**
```csv
province,lat,lon,population,gdp_per_capita
Metro Manila,14.5995,120.9842,13484462,8500
Cebu,10.3157,123.8854,5113436,4100
Davao del Sur,6.7781,125.1753,682679,3600
```

### Storm Forecast Data (CSV)

The storm forecast data should contain meteorological predictions:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `lat` | Float | ✅ | Latitude of storm center in decimal degrees |
| `lon` | Float | ✅ | Longitude of storm center in decimal degrees |
| `intensity` | Float | ✅ | Storm intensity (0-200 scale, where 200 = Cat 5) |
| `timestamp` | String | ❌ | ISO format datetime (for tracking) |

**Example:**
```csv
lat,lon,intensity,timestamp
14.2,125.8,85,2024-01-15 00:00:00
14.0,125.4,95,2024-01-15 06:00:00
13.8,125.0,105,2024-01-15 12:00:00
```

## 🧮 Impact Calculation Model

The economic impact estimation uses a sophisticated model that considers:

### Distance-Based Impact Decay

The impact decreases exponentially with distance from the storm center:

```
distance_factor = exp(-distance / 100km)
```

### Intensity Scaling

Storm intensity is normalized and scaled:

```
intensity_factor = storm_intensity / 100
```

### Economic Impact Formula

```
impact_score = distance_factor × intensity_factor
economic_impact_pct = impact_score × 0.3  (max 30% of local GDP)
economic_impact_usd = population × gdp_per_capita × economic_impact_pct
```

### Impact Classification

- **High Impact**: Impact score > 0.5 (>15% GDP loss)
- **Medium Impact**: 0.2 < Impact score ≤ 0.5 (6-15% GDP loss)
- **Low Impact**: 0.1 < Impact score ≤ 0.2 (3-6% GDP loss)
- **No Significant Impact**: Impact score ≤ 0.1 (<3% GDP loss)

## 🎛️ Using the Application

### 1. Data Input Options

**Option A: Use Sample Data**
- Check "Use Sample Data" in the sidebar
- Sample datasets will be loaded automatically
- Includes 80+ Philippine provinces and a realistic storm track

**Option B: Upload Custom Data**
- Uncheck "Use Sample Data"
- Upload your province CSV file
- Upload your storm forecast CSV file
- Data will be automatically validated

### 2. Analysis Parameters

- **Maximum Impact Distance**: Set the maximum distance (50-500 km) where storms can cause economic impact
- Default is 200km, which covers most regional storm effects

### 3. Viewing Results

**Interactive Map Tab**:
- Pan and zoom the Philippines map
- Click on provinces to see detailed impact information
- View storm track and intensity visualization
- Color-coded impact levels

**Impact Analysis Tab**:
- Bar chart of most affected provinces
- Scatter plot showing impact vs. distance relationship
- Pie chart of impact distribution

**Data Tables Tab**:
- Most affected provinces ranking
- Complete province data with calculated impacts
- Storm forecast point details
- Summary statistics by impact level

**Export Results Tab**:
- Download complete analysis as CSV
- Includes all calculated metrics and metadata
- Timestamped for record keeping

## 🔧 Customization

### Modifying Impact Parameters

You can adjust the impact model in `utils.py`:

```python
# In calculate_storm_impact function
distance_factor = np.exp(-distance / 100)  # Change decay rate
intensity_factor = storm_point['intensity'] / 100  # Change intensity scaling
economic_impact_pct = max_impact * 0.3  # Change max impact percentage
```

### Adding New Visualizations

Add custom charts in `app.py` by extending the `create_impact_charts` function or adding new tabs.

### Custom Map Styling

Modify the `create_philippines_map` function to change:
- Map tiles and styling
- Marker colors and sizes
- Popup content
- Legend appearance

## 📁 Project Structure

```
pjdsc/
├── app.py                     # Main Streamlit application
├── utils.py                   # Core analysis functions
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
└── data/
    ├── sample_provinces.csv   # Sample province data
    └── sample_forecast.csv    # Sample storm forecast
```

## 🛠️ Technical Details

### Dependencies

- **Streamlit**: Web app framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Folium**: Interactive maps
- **Plotly**: Interactive charts
- **SciPy**: Scientific computing (distance calculations)

### Performance Considerations

- The app handles up to 100+ provinces efficiently
- Storm tracks with 50+ points render smoothly
- Large datasets (>1000 points) may require optimization
- Map rendering time scales with number of markers

### Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🤝 Contributing

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly with sample data
5. Submit a pull request

### Reporting Issues

Please report bugs or feature requests through the issue tracker, including:
- Steps to reproduce
- Expected vs. actual behavior
- Data samples (if applicable)
- Screenshots (for UI issues)

## 📋 Troubleshooting

### Common Issues

**"Import could not be resolved" errors**:
```bash
pip install -r requirements.txt
```

**Map not displaying**:
- Check internet connection (map tiles require online access)
- Verify coordinate data is valid (latitude: -90 to 90, longitude: -180 to 180)

**Data validation errors**:
- Ensure CSV headers match expected format exactly
- Check for missing required columns
- Verify numeric data types for coordinates and intensity

**Performance issues**:
- Reduce maximum impact distance
- Limit storm track points to <100
- Use smaller province datasets for testing

### Getting Help

1. Check this README for common solutions
2. Review the error messages in the Streamlit interface
3. Verify your data format matches the specifications
4. Test with the provided sample data first

## 📄 License

This project is created for educational and research purposes. Please ensure you have proper licensing for any production use.

## 🙏 Acknowledgments

- Philippine geographical data from open government sources
- Storm modeling techniques based on meteorological research
- Built with the amazing Streamlit framework
- Interactive mapping powered by Folium and OpenStreetMap

---

**Made with ❤️ for disaster preparedness and impact analysis**