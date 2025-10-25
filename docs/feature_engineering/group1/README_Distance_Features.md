# Distance & Proximity Feature Engineering Pipeline

## Overview

This pipeline processes storm location data to generate **26 GROUP 1 features** for machine learning models predicting storm impacts. The features quantify spatial relationships between storm trajectories and province locations.

## Files Created

### 1. `distance_feature_engineering.py`
Main feature engineering script with modular functions for:
- ✅ Basic Distance Metrics (5 features)
- ✅ Duration Thresholds (5 features)  
- ✅ Forecast Horizons (5 features)
- ✅ Integrated Proximity (3 features)
- ✅ Approach/Departure Dynamics (4 features)
- ✅ Geometric Features (4 features)

### 2. `distance_features_group1.csv`
Pre-computed features for all historical storms (2010-2025):
- **21,074 records** (storm-province pairs)
- **114 storms** across **16 years**
- **82 provinces**
- **26 features** per record

### 3. `example_deployment_usage.py`
Usage examples demonstrating:
- Real-time forecast feature extraction
- Single storm-province processing
- Batch processing
- ML pipeline integration

---

## 📊 The 26 Features

### Group 1.1: Basic Distance Metrics (5 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `min_distance_km` | Closest approach distance | km |
| `mean_distance_km` | Average distance throughout event | km |
| `max_distance_km` | Farthest distance | km |
| `distance_range_km` | Distance variability | km |
| `distance_std_km` | Standard deviation of distances | km |

### Group 1.2: Duration Thresholds (5 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `hours_under_50km` | Time within 50km | hours |
| `hours_under_100km` | Time within 100km | hours |
| `hours_under_200km` | Time within 200km | hours |
| `hours_under_300km` | Time within 300km | hours |
| `hours_under_500km` | Time within 500km | hours |

### Group 1.3: Forecast Horizons (5 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `distance_at_current` | Distance NOW | km |
| `distance_at_12hr` | Distance at +12hr forecast | km |
| `distance_at_24hr` | Distance at +24hr forecast | km |
| `distance_at_48hr` | Distance at +48hr forecast | km |
| `distance_at_72hr` | Distance at +72hr forecast | km |

### Group 1.4: Integrated Proximity (3 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `integrated_proximity` | Sum of inverse-square distances | 1/km² |
| `weighted_exposure_hours` | Time-weighted exposure (exponential decay) | dimensionless |
| `proximity_peak` | Maximum proximity value | 1/km² |

### Group 1.5: Approach/Departure Dynamics (4 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `approach_speed_kmh` | How fast storm approaches | km/h |
| `departure_speed_kmh` | How fast storm departs | km/h |
| `time_approaching_hours` | Duration of approach phase | hours |
| `time_departing_hours` | Duration of departure phase | hours |

### Group 1.6: Geometric Features (4 features)
| Feature | Description | Units |
|---------|-------------|-------|
| `bearing_at_closest_deg` | Direction at closest approach | degrees (0-360) |
| `bearing_variability_deg` | Storm direction change | std(degrees) |
| `approach_angle_deg` | Angle between storm motion and province | degrees (0-180) |
| `did_cross_province` | Boolean: Direct hit? | 0 or 1 |

---

## 🚀 Quick Start

### Process All Historical Data
```python
from distance_feature_engineering import process_all_storm_data

# Generate features for all storms
features_df = process_all_storm_data(
    storm_location_dir='storm_location_data',
    output_file='distance_features_group1.csv',
    time_interval_hours=3.0
)
```

### Process Single Storm
```python
from distance_feature_engineering import process_single_storm_file
from pathlib import Path

# Process one storm file
features = process_single_storm_file(
    file_path=Path('storm_location_data/2021/2021_Odette.csv'),
    time_interval_hours=3.0
)
```

### Deployment: Real-time Feature Extraction
```python
from distance_feature_engineering import extract_features_for_deployment

# From live forecast data
features = extract_features_for_deployment(
    distances=[1250.5, 1100.3, 950.8, 750.2, 600.1],  # km
    bearings=[85.5, 82.3, 78.9, 75.1, 71.8],  # degrees
    timestamps=['2024-11-15 00:00:00', '2024-11-15 12:00:00', ...],
    time_interval_hours=12.0
)
```

---

## 📈 Example Output

### Top 5 Most Impacted Provinces (Typhoon Odette 2021)
```
         Province  min_distance_km  hours_under_300km  did_cross_province
   Southern Leyte            31.11               18.0                   1
Surigao del Norte            48.28               18.0                   1
             Cebu            54.99               18.0                   0
         Guimaras            56.66               24.0                   0
            Bohol            63.04               18.0                   0
```

### Feature Statistics (All Storms 2010-2025)
```
       min_distance_km  mean_distance_km  hours_under_300km  approach_speed_kmh
count     21074.000000      21074.000000       21074.000000        21074.000000
mean        823.657903       1757.387405           4.086457           11.107110
std         539.615244        792.536875          10.238536            8.871908
min           1.800000        129.837692           0.000000            0.000000
25%         396.105000       1151.937844           0.000000            4.610176
50%         761.135000       1595.092840           0.000000           10.823800
75%        1170.767500       2217.321045           0.000000           16.442214
max        4422.780000       5260.338442         117.000000          137.702083
```

---

## 🔧 Function Reference

### Core Functions

#### `compute_all_distance_features(storm_province_df, time_interval_hours=3.0)`
Compute all 26 features for a single storm-province pair.

**Args:**
- `storm_province_df`: DataFrame with [Timestamp, Distance_KM, Bearing_Degrees]
- `time_interval_hours`: Time between observations (default: 3 hours)

**Returns:** Dictionary with all 26 features

#### `process_single_storm_file(file_path, time_interval_hours=3.0)`
Process a single storm CSV file for all provinces.

**Args:**
- `file_path`: Path to storm location CSV
- `time_interval_hours`: Time between observations

**Returns:** DataFrame with features for all provinces

#### `process_all_storm_data(storm_location_dir, output_file, time_interval_hours=3.0)`
Batch process all storms in directory.

**Args:**
- `storm_location_dir`: Directory with year subdirectories
- `output_file`: Output CSV filename
- `time_interval_hours`: Time between observations

**Returns:** DataFrame with all features

#### `extract_features_for_deployment(distances, bearings, timestamps, time_interval_hours)`
Streamlined function for real-time deployment.

**Args:**
- `distances`: List of distances (km)
- `bearings`: List of bearings (degrees)
- `timestamps`: List of timestamp strings
- `time_interval_hours`: Time between observations

**Returns:** Dictionary with all 26 features

---

## 📁 Input Data Format

### Expected CSV Structure (`storm_location_data/YEAR/YEAR_STORMNAME.csv`)
```csv
Timestamp,Province,Distance_KM,Bearing_Degrees,Direction
2010-03-19 14:00:00,Abra,3905.7,110.44,ESE
2010-03-19 17:00:00,Abra,3871.26,111.3,ESE
...
```

### Output Format (`distance_features_group1.csv`)
```csv
Year,Storm,Province,approach_angle_deg,approach_speed_kmh,...
2010,Agaton,Abra,1.59,16.85,...
2010,Agaton,Agusan del Norte,3.27,16.41,...
...
```

---

## 🎯 Use Cases

### 1. Training ML Models
```python
import pandas as pd

# Load features
features = pd.read_csv('distance_features_group1.csv')

# Split by year
train = features[features['Year'] <= 2020]
test = features[features['Year'] > 2020]

# Select feature columns
X_train = train.drop(['Year', 'Storm', 'Province'], axis=1)
```

### 2. Real-time Impact Prediction
```python
# Get live forecast from JTWC API
forecast_data = fetch_jtwc_forecast(storm_id)

# Extract features for each province
for province in provinces:
    features = extract_features_for_deployment(
        distances=forecast_data['distances'][province],
        bearings=forecast_data['bearings'][province],
        timestamps=forecast_data['timestamps'],
        time_interval_hours=12.0
    )
    
    # Predict impact
    impact_prediction = model.predict([list(features.values())])
```

### 3. Feature Analysis
```python
# Identify most predictive features
import matplotlib.pyplot as plt

features_df = pd.read_csv('distance_features_group1.csv')

# Correlation with actual impacts (if available)
# correlations = features_df.corr()['actual_impact'].sort_values(ascending=False)
```

---

## ⚡ Performance

- **Processing Speed**: ~1-2 seconds per storm
- **Total Processing Time**: ~3-5 minutes for all 114 storms
- **Output Size**: ~2.5 MB CSV file
- **Memory Usage**: < 200 MB

---

## 🔮 Future Enhancements

- [ ] Add Group 2: Storm Intensity Features
- [ ] Add Group 3: Temporal Features
- [ ] Add Group 4: Environmental Features
- [ ] Parallel processing for faster batch operations
- [ ] Integration with live JTWC API
- [ ] Feature importance analysis
- [ ] Automated feature selection

---

## 📝 Notes

- **Missing Values**: Indicated by `-999` (e.g., when forecast horizon not available)
- **Time Intervals**: Adjust `time_interval_hours` based on data resolution
  - Historical data: 3 hours (typical)
  - Forecast data: 6-12 hours (JTWC format)
- **Distance Calculations**: Uses Haversine formula for great-circle distance
- **Bearing Calculations**: 0° = North, 90° = East, 180° = South, 270° = West

---

## 🤝 Integration with Existing Pipeline

This module is designed to work with:
1. `storm_distance_calculator.py` - Generates input distance data
2. Impact prediction models - Consumes these features
3. Real-time forecast systems - Uses deployment functions

---

**Author:** Feature Engineering Pipeline  
**Version:** 1.0  
**Last Updated:** 2024  
**License:** MIT

