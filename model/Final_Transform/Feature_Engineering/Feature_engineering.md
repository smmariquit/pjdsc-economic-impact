# 🔧 **COMPLETE FEATURE ENGINEERING SPECIFICATION**

---

## 📋 **FEATURE ENGINEERING OVERVIEW**

### **Key Principles**

1. **Forecast Compatibility**: All features must work with BOTH:
   - Dense historical tracks (3-6hr intervals, complete lifecycle)
   - Sparse forecast tracks (12-24hr intervals, only future positions)

2. **Aggregation Strategy**: Temporal sequences → Single feature vector per storm-province pair

3. **Missing Data Handling**: Features must gracefully handle:
   - Missing storm intensity data (wind/pressure)
   - Incomplete wind radii
   - Variable-length storm tracks

4. **Target Alignment**: Features describe the ENTIRE storm event (not point-in-time), matching cumulative impact reports

---

## 🌀 **GROUP 1: DISTANCE & PROXIMITY FEATURES**

**Purpose**: Quantify spatial relationship between storm trajectory and province location

**Why These Matter**: Distance is the single most important predictor of impact. Closer storms = higher impact.

---

### **1.1 Basic Distance Metrics**

| Feature Name | Description | Formula | Units | Forecast Compatible? |
|--------------|-------------|---------|-------|---------------------|
| `min_distance_km` | Closest approach distance | `min(distances)` | km | ✅ YES |
| `mean_distance_km` | Average distance throughout event | `mean(distances)` | km | ✅ YES |
| `max_distance_km` | Farthest distance (storm entry/exit) | `max(distances)` | km | ✅ YES |
| `distance_range_km` | Variability in distance | `max(distances) - min(distances)` | km | ✅ YES |
| `distance_std_km` | Standard deviation of distances | `std(distances)` | km | ✅ YES |

**Implementation:**
```python
def compute_basic_distance_features(track_points, province_centroid):
    """
    Works with any number of track points (dense or sparse)
    """
    distances = []
    for point in track_points:
        dist = haversine(
            point.lat, point.lon,
            province_centroid.lat, province_centroid.lon
        )
        distances.append(dist)
    
    return {
        'min_distance_km': min(distances),
        'mean_distance_km': np.mean(distances),
        'max_distance_km': max(distances),
        'distance_range_km': max(distances) - min(distances),
        'distance_std_km': np.std(distances)
    }
```

---

### **1.2 Threshold-Based Duration Features**

| Feature Name | Description | Computation | Units | Forecast Compatible? |
|--------------|-------------|-------------|-------|---------------------|
| `hours_under_50km` | Time within 50km | Count points < 50km × interval | hours | ⚠️ Needs adjustment |
| `hours_under_100km` | Time within 100km | Count points < 100km × interval | hours | ⚠️ Needs adjustment |
| `hours_under_200km` | Time within 200km | Count points < 200km × interval | hours | ⚠️ Needs adjustment |
| `hours_under_300km` | Time within 300km | Count points < 300km × interval | hours | ⚠️ Needs adjustment |
| `hours_under_500km` | Time within 500km | Count points < 500km × interval | hours | ⚠️ Needs adjustment |

**Implementation (Forecast-Compatible):**
```python
def compute_duration_features(track_points, province_centroid, is_forecast=False):
    """
    Handles both dense historical and sparse forecast tracks
    """
    distances = [haversine(p.lat, p.lon, province_centroid.lat, province_centroid.lon) 
                 for p in track_points]
    times = [p.timestamp for p in track_points]
    
    features = {}
    
    for threshold in [50, 100, 200, 300, 500]:
        if is_forecast:
            # For sparse forecasts: interpolate duration
            features[f'hours_under_{threshold}km'] = estimate_duration_interpolated(
                distances, times, threshold
            )
        else:
            # For dense historical: count timesteps
            count = sum(1 for d in distances if d < threshold)
            # Assume 3-hour intervals (adjust if needed)
            features[f'hours_under_{threshold}km'] = count * 3
    
    return features

def estimate_duration_interpolated(distances, times, threshold):
    """
    Estimate time within threshold from sparse forecast points
    """
    # Find first and last points within threshold
    within_threshold = [(i, d, t) for i, (d, t) in enumerate(zip(distances, times)) 
                        if d < threshold]
    
    if len(within_threshold) == 0:
        return 0
    
    # Simple estimate: time between first and last close approach
    first_idx, _, first_time = within_threshold[0]
    last_idx, _, last_time = within_threshold[-1]
    
    duration = (last_time - first_time).total_seconds() / 3600  # hours
    
    # If only one point within threshold, estimate based on adjacent points
    if duration == 0 and len(within_threshold) == 1:
        # Rough estimate: assume 6 hours if sandwiched between farther points
        return 6
    
    return duration
```

---

### **1.3 Distance at Specific Forecast Horizons**

**Purpose**: Align features with operational forecast structure (JTWC format)

| Feature Name | Description | When Applicable | Forecast Compatible? |
|--------------|-------------|-----------------|---------------------|
| `distance_at_current` | Distance NOW | Always | ✅ YES |
| `distance_at_12hr` | Distance at +12hr forecast | If forecast available | ✅ YES |
| `distance_at_24hr` | Distance at +24hr forecast | If forecast available | ✅ YES |
| `distance_at_48hr` | Distance at +48hr forecast | If forecast available | ✅ YES |
| `distance_at_72hr` | Distance at +72hr forecast | If forecast available | ✅ YES |

**Implementation:**
```python
def compute_forecast_horizon_features(track_points, province_centroid):
    """
    Extract distances at standard forecast intervals
    Works for both training (simulated) and deployment (actual forecasts)
    """
    features = {}
    
    # Map forecast horizons to track points
    # For training: extract from historical track at these intervals
    # For deployment: use JTWC forecast positions directly
    
    forecast_horizons = [0, 12, 24, 48, 72]  # hours
    
    for i, tau in enumerate(forecast_horizons):
        if i < len(track_points):
            point = track_points[i]
            dist = haversine(point.lat, point.lon, 
                           province_centroid.lat, province_centroid.lon)
            
            if tau == 0:
                features['distance_at_current'] = dist
            else:
                features[f'distance_at_{tau}hr'] = dist
        else:
            # Not enough forecast points (storm may dissipate)
            if tau == 0:
                features['distance_at_current'] = track_points[-1].distance
            else:
                features[f'distance_at_{tau}hr'] = -999  # Missing indicator
    
    return features
```

---

### **1.4 Integrated Proximity Metrics**

**Purpose**: Weight storm positions by their proximity (closer = more weight)

| Feature Name | Description | Formula | Units | Forecast Compatible? |
|--------------|-------------|---------|-------|---------------------|
| `integrated_proximity` | Sum of inverse-square distances | `Σ(1/distance²)` | 1/km² | ✅ YES |
| `weighted_exposure_hours` | Time-weighted exposure | `Σ(hours × e^(-distance/100))` | dimensionless | ✅ YES |
| `proximity_peak` | Maximum proximity value | `max(1/distance²)` | 1/km² | ✅ YES |

**Implementation:**
```python
def compute_integrated_proximity_features(track_points, province_centroid):
    """
    Aggregate proximity over entire storm track
    """
    distances = [haversine(p.lat, p.lon, province_centroid.lat, province_centroid.lon) 
                 for p in track_points]
    
    # Integrated proximity (sum of inverse-square distances)
    # Add small constant to avoid division by zero
    integrated_proximity = sum(1 / (d**2 + 1) for d in distances)
    
    # Weighted exposure (exponential decay with distance)
    weighted_exposure = sum(np.exp(-d / 100) for d in distances)
    
    # Peak proximity
    proximity_peak = max(1 / (d**2 + 1) for d in distances)
    
    return {
        'integrated_proximity': integrated_proximity,
        'weighted_exposure_hours': weighted_exposure,
        'proximity_peak': proximity_peak
    }
```

---

### **1.5 Approach/Departure Dynamics**

| Feature Name | Description | Computation | Units | Forecast Compatible? |
|--------------|-------------|-------------|-------|---------------------|
| `approach_speed_kmh` | How fast storm is getting closer | `Δdistance / Δtime` (before closest) | km/h | ✅ YES |
| `departure_speed_kmh` | How fast storm is moving away | `Δdistance / Δtime` (after closest) | km/h | ✅ YES |
| `time_approaching_hours` | Duration of approach phase | Time before closest approach | hours | ⚠️ Adjust for sparse |
| `time_departing_hours` | Duration of departure phase | Time after closest approach | hours | ⚠️ Adjust for sparse |

**Implementation:**
```python
def compute_approach_departure_features(track_points, province_centroid):
    """
    Analyze storm approach and departure dynamics
    """
    distances = [haversine(p.lat, p.lon, province_centroid.lat, province_centroid.lon) 
                 for p in track_points]
    times = [p.timestamp for p in track_points]
    
    # Find closest approach index
    closest_idx = np.argmin(distances)
    
    features = {}
    
    # Approach phase (before closest)
    if closest_idx > 0:
        approach_distances = distances[:closest_idx+1]
        approach_times = times[:closest_idx+1]
        
        # Calculate approach speed
        distance_change = approach_distances[0] - approach_distances[-1]
        time_change = (approach_times[-1] - approach_times[0]).total_seconds() / 3600
        
        if time_change > 0:
            features['approach_speed_kmh'] = distance_change / time_change
            features['time_approaching_hours'] = time_change
        else:
            features['approach_speed_kmh'] = 0
            features['time_approaching_hours'] = 0
    else:
        features['approach_speed_kmh'] = 0
        features['time_approaching_hours'] = 0
    
    # Departure phase (after closest)
    if closest_idx < len(distances) - 1:
        departure_distances = distances[closest_idx:]
        departure_times = times[closest_idx:]
        
        distance_change = departure_distances[-1] - departure_distances[0]
        time_change = (departure_times[-1] - departure_times[0]).total_seconds() / 3600
        
        if time_change > 0:
            features['departure_speed_kmh'] = distance_change / time_change
            features['time_departing_hours'] = time_change
        else:
            features['departure_speed_kmh'] = 0
            features['time_departing_hours'] = 0
    else:
        features['departure_speed_kmh'] = 0
        features['time_departing_hours'] = 0
    
    return features
```

---

### **1.6 Geometric Features**

| Feature Name | Description | Units | Forecast Compatible? |
|--------------|-------------|-------|---------------------|
| `bearing_at_closest_deg` | Direction from province to storm at closest approach | degrees (0-360) | ✅ YES |
| `bearing_variability_deg` | How much storm direction changes | std(bearings) | degrees | ✅ YES |
| `approach_angle_deg` | Angle between storm motion and province direction | degrees (0-180) | ✅ YES |
| `did_cross_province` | Boolean: Did storm pass directly over? | 0 or 1 | ✅ YES |

**Implementation:**
```python
def compute_geometric_features(track_points, province_centroid):
    """
    Geometric relationships between storm and province
    """
    distances = [haversine(p.lat, p.lon, province_centroid.lat, province_centroid.lon) 
                 for p in track_points]
    bearings = [calculate_bearing(province_centroid.lat, province_centroid.lon,
                                   p.lat, p.lon) 
                for p in track_points]
    
    closest_idx = np.argmin(distances)
    
    features = {
        'bearing_at_closest_deg': bearings[closest_idx],
        'bearing_variability_deg': np.std(bearings),
        'did_cross_province': 1 if distances[closest_idx] < 50 else 0
    }
    
    # Approach angle (storm motion vs. province direction)
    if closest_idx > 0:
        # Storm motion direction at closest approach
        storm_direction = calculate_bearing(
            track_points[closest_idx-1].lat, track_points[closest_idx-1].lon,
            track_points[closest_idx].lat, track_points[closest_idx].lon
        )
        
        # Direction from storm to province
        province_direction = calculate_bearing(
            track_points[closest_idx].lat, track_points[closest_idx].lon,
            province_centroid.lat, province_centroid.lon
        )
        
        # Angle between them (0-180 degrees)
        approach_angle = abs(storm_direction - province_direction)
        if approach_angle > 180:
            approach_angle = 360 - approach_angle
        
        features['approach_angle_deg'] = approach_angle
    else:
        features['approach_angle_deg'] = -999  # Unknown
    
    return features

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 (0-360 degrees)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing
```

---

## **GROUP 1 SUMMARY: 26 Distance Features**

| Subgroup | Feature Count | Key Features |
|----------|---------------|--------------|
| Basic Distance | 5 | min, mean, max, range, std |
| Duration Thresholds | 5 | hours_under_50/100/200/300/500km |
| Forecast Horizons | 5 | distance_at_current/12hr/24hr/48hr/72hr |
| Integrated Proximity | 3 | integrated_proximity, weighted_exposure, proximity_peak |
| Approach/Departure | 4 | approach_speed, departure_speed, time_approaching/departing |
| Geometric | 4 | bearing_at_closest, bearing_variability, approach_angle, did_cross |

---

## ☔ **GROUP 2: WEATHER EXPOSURE FEATURES**

**Purpose**: Quantify actual meteorological hazards experienced by province

**Why These Matter**: Open-Meteo data is 100% complete and more reliable than incomplete storm intensity data. This is your MOST IMPORTANT feature group!

---

### **2.1 Peak Intensity Metrics**

| Feature Name | Description | Aggregation | Units | Data Source |
|--------------|-------------|-------------|-------|-------------|
| `max_wind_gust_kmh` | Maximum wind gust across all days | `max(wind_gusts_10m_max)` | km/h | Open-Meteo |
| `max_wind_speed_kmh` | Maximum sustained wind | `max(wind_speed_10m_max)` | km/h | Open-Meteo |
| `max_hourly_precip_mm` | Peak hourly precipitation rate | `max(precipitation_rate)` | mm/hr | Open-Meteo (if available) |
| `max_daily_precip_mm` | Peak daily rainfall | `max(precipitation_sum)` | mm/day | Open-Meteo |

**Implementation:**
```python
def compute_peak_intensity_features(weather_data):
    """
    Extract peak values across entire storm event
    
    Parameters:
    weather_data: DataFrame with columns [date, wind_gusts_10m_max, 
                  wind_speed_10m_max, precipitation_sum, ...]
    """
    return {
        'max_wind_gust_kmh': weather_data['wind_gusts_10m_max'].max(),
        'max_wind_speed_kmh': weather_data['wind_speed_10m_max'].max(),
        'max_daily_precip_mm': weather_data['precipitation_sum'].max()
    }
```

---

### **2.2 Accumulation & Duration Metrics**

| Feature Name | Description | Aggregation | Units |
|--------------|-------------|-------------|-------|
| `total_precipitation_mm` | Total rainfall during event | `sum(precipitation_sum)` | mm |
| `days_with_rain` | Number of days with measurable rain | `count(precipitation_sum > 1)` | days |
| `days_with_heavy_rain` | Days with >50mm rain | `count(precipitation_sum > 50)` | days |
| `days_with_very_heavy_rain` | Days with >100mm rain | `count(precipitation_sum > 100)` | days |
| `days_with_strong_wind` | Days with gusts >60km/h | `count(wind_gusts_10m_max > 60)` | days |
| `days_with_damaging_wind` | Days with gusts >90km/h | `count(wind_gusts_10m_max > 90)` | days |
| `consecutive_rain_days` | Longest streak of rainy days | Max consecutive days > 10mm | days |
| `total_precipitation_hours` | Hours with precipitation | `sum(precipitation_hours)` | hours |

**Implementation:**
```python
def compute_accumulation_duration_features(weather_data):
    """
    Accumulation and duration metrics
    """
    features = {
        'total_precipitation_mm': weather_data['precipitation_sum'].sum(),
        'days_with_rain': (weather_data['precipitation_sum'] > 1).sum(),
        'days_with_heavy_rain': (weather_data['precipitation_sum'] > 50).sum(),
        'days_with_very_heavy_rain': (weather_data['precipitation_sum'] > 100).sum(),
        'days_with_strong_wind': (weather_data['wind_gusts_10m_max'] > 60).sum(),
        'days_with_damaging_wind': (weather_data['wind_gusts_10m_max'] > 90).sum(),
        'total_precipitation_hours': weather_data['precipitation_hours'].sum()
    }
    
    # Consecutive rainy days
    rainy_days = (weather_data['precipitation_sum'] > 10).astype(int)
    features['consecutive_rain_days'] = max_consecutive_ones(rainy_days.values)
    
    return features

def max_consecutive_ones(arr):
    """Find longest streak of 1s in binary array"""
    max_count = 0
    current_count = 0
    for val in arr:
        if val == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count
```

---

### **2.3 Temporal Distribution Features**

| Feature Name | Description | Computation | Interpretation |
|--------------|-------------|-------------|----------------|
| `precipitation_concentration_index` | How concentrated rainfall is | Gini coefficient of daily rainfall | 0=uniform, 1=single-day deluge |
| `rain_during_closest_approach` | Rainfall when storm was nearest | Precipitation on day of min_distance | mm |
| `wind_gust_persistence_score` | How sustained high winds are | Fraction of days with gusts >50km/h | 0-1 |
| `mean_daily_precipitation_mm` | Average rainfall per day | `mean(precipitation_sum)` | mm/day |
| `precip_variability` | Day-to-day rainfall variability | `std(precipitation_sum)` | mm |

**Implementation:**
```python
def compute_temporal_distribution_features(weather_data, closest_approach_date):
    """
    How weather hazards are distributed over time
    """
    # Precipitation concentration index (Gini coefficient)
    daily_rain = weather_data['precipitation_sum'].values
    pci = gini_coefficient(daily_rain)
    
    # Rainfall during closest approach
    closest_day_data = weather_data[weather_data['date'] == closest_approach_date]
    if len(closest_day_data) > 0:
        rain_during_closest = closest_day_data['precipitation_sum'].iloc[0]
    else:
        rain_during_closest = 0
    
    # Wind persistence
    days_with_strong_wind = (weather_data['wind_gusts_10m_max'] > 50).sum()
    total_days = len(weather_data)
    wind_persistence = days_with_strong_wind / total_days if total_days > 0 else 0
    
    return {
        'precipitation_concentration_index': pci,
        'rain_during_closest_approach': rain_during_closest,
        'wind_gust_persistence_score': wind_persistence,
        'mean_daily_precipitation_mm': weather_data['precipitation_sum'].mean(),
        'precip_variability': weather_data['precipitation_sum'].std()
    }

def gini_coefficient(values):
    """
    Calculate Gini coefficient (0 = perfectly equal, 1 = maximum inequality)
    """
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n+1)/n
```

---

### **2.4 Combined Hazard Features**

| Feature Name | Description | Formula | Units |
|--------------|-------------|---------|-------|
| `wind_rain_product` | Combined wind-rain intensity | `max_wind_gust × total_precipitation` | (km/h)×mm |
| `compound_hazard_score` | Normalized combined hazard | `(wind_z + rain_z) / 2` | z-score |
| `compound_hazard_days` | Days with both high wind AND rain | `count((wind>60) & (rain>30))` | days |
| `wet_and_windy_hours` | Hours with simultaneous hazards | Estimated from daily data | hours |

**Implementation:**
```python
def compute_combined_hazard_features(weather_data):
    """
    Features capturing simultaneous wind and rain hazards
    """
    max_wind = weather_data['wind_gusts_10m_max'].max()
    total_rain = weather_data['precipitation_sum'].sum()
    
    # Simple product
    wind_rain_product = max_wind * total_rain
    
    # Days with both hazards
    compound_days = (
        (weather_data['wind_gusts_10m_max'] > 60) & 
        (weather_data['precipitation_sum'] > 30)
    ).sum()
    
    # Normalized compound hazard (z-scores)
    # Would need population statistics for proper normalization
    # Placeholder: simple normalization
    wind_normalized = max_wind / 100  # Normalize by typical max
    rain_normalized = total_rain / 200  # Normalize by typical accumulation
    compound_hazard_score = (wind_normalized + rain_normalized) / 2
    
    return {
        'wind_rain_product': wind_rain_product,
        'compound_hazard_score': compound_hazard_score,
        'compound_hazard_days': compound_days
    }
```

---

## **GROUP 2 SUMMARY: 22 Weather Features**

| Subgroup | Feature Count | Key Features |
|----------|---------------|--------------|
| Peak Intensity | 4 | max_wind_gust, max_wind_speed, max_daily_precip |
| Accumulation/Duration | 8 | total_precipitation, days_with_heavy_rain, consecutive_rain_days |
| Temporal Distribution | 5 | precipitation_concentration_index, rain_during_closest |
| Combined Hazards | 3 | wind_rain_product, compound_hazard_score |
| Variability | 2 | precip_variability, wind_gust_persistence |

---

## 🌪️ **GROUP 3: STORM INTENSITY FEATURES**

**Purpose**: Capture storm strength characteristics (when available)

**⚠️ Important**: These features have 40-70% missing data! Must handle gracefully.

---

### **3.1 Intensity at Critical Moments**

| Feature Name | Description | Source | Completeness | Missing Value |
|--------------|-------------|--------|--------------|---------------|
| `wind_at_closest_approach_kt` | Wind speed when nearest | TOKYO_WIND or USA_WIND | ~60-70% | -999 |
| `max_wind_in_track_kt` | Peak intensity during event | `max(TOKYO_WIND)` | ~60-70% | -999 |
| `pressure_at_closest_hpa` | Pressure at closest approach | TOKYO_PRES | ~55% | -999 |
| `min_pressure_in_track_hpa` | Minimum pressure (strongest) | `min(TOKYO_PRES)` | ~55% | -999 |

**Implementation:**
```python
def compute_intensity_at_critical_moments(track_points, closest_approach_idx):
    """
    Extract intensity data at key moments (handle missing data)
    """
    features = {}
    
# Factor to convert 1-minute wind (USA) to 10-minute wind (TOKYO)
    USA_TO_TOKYO_FACTOR = 0.88
    
    closest_point = track_points[closest_approach_idx]
    
    # Wind at closest approach (standardized to 10-minute sustained knots)
    if hasattr(closest_point, 'TOKYO_WIND') and not np.isnan(closest_point.TOKYO_WIND):
        # Use the 10-minute wind data directly
        features['wind_at_closest_approach_10min_kt'] = closest_point.TOKYO_WIND
    elif hasattr(closest_point, 'USA_WIND') and not np.isnan(closest_point.USA_WIND):
        # Convert the 1-minute wind data to its 10-minute equivalent
        features['wind_at_closest_approach_10min_kt'] = closest_point.USA_WIND * USA_TO_TOKYO_FACTOR
    else:
        # No valid wind data found
        features['wind_at_closest_approach_10min_kt'] = -999
    
    # Maximum wind in track
    tokyo_winds = [p.TOKYO_WIND for p in track_points if hasattr(p, 'TOKYO_WIND') and not np.isnan(p.TOKYO_WIND)]
    usa_winds = [p.USA_WIND for p in track_points if hasattr(p, 'USA_WIND') and not np.isnan(p.USA_WIND)]
    
    if len(tokyo_winds) > 0:
        features['max_wind_in_track_kt'] = max(tokyo_winds)
    elif len(usa_winds) > 0:
        features['max_wind_in_track_kt'] = max(usa_winds)
    else:
        features['max_wind_in_track_kt'] = -999
    
    # Pressure at closest approach
    if hasattr(closest_point, 'TOKYO_PRES') and not np.isnan(closest_point.TOKYO_PRES):
        features['pressure_at_closest_hpa'] = closest_point.TOKYO_PRES
    else:
        features['pressure_at_closest_hpa'] = -999
    
    # Minimum pressure in track
    pressures = [p.TOKYO_PRES for p in track_points if hasattr(p, 'TOKYO_PRES') and not np.isnan(p.TOKYO_PRES)]
    if len(pressures) > 0:
        features['min_pressure_in_track_hpa'] = min(pressures)
    else:
        features['min_pressure_in_track_hpa'] = -999
    
    return features
```

---

### **3.2 Intensity Evolution Features**

| Feature Name | Description | Computation | Completeness |
|--------------|-------------|-------------|--------------|
| `is_intensifying` | Is storm strengthening toward closest approach? | Boolean | ~50% |
| `wind_change_approaching_kt` | Wind speed change during approach | Wind_closest - Wind_entry | ~50% |
| `intensification_rate_kt_per_day` | Rate of intensification | ΔWind / Δtime | ~50% |


**Implementation:**
```python
def compute_intensity_evolution_features(track_points, closest_approach_idx):
    """
    How storm intensity changed over time
    """
    features = {}
    
    # Collect wind speeds with valid data
    winds_with_time = [
        (i, p.TOKYO_WIND if hasattr(p, 'TOKYO_WIND') else p.USA_WIND, p.timestamp)
        for i, p in enumerate(track_points)
        if (hasattr(p, 'TOKYO_WIND') and not np.isnan(p.TOKYO_WIND)) or
           (hasattr(p, 'USA_WIND') and not np.isnan(p.USA_WIND))
    ]
    
    if len(winds_with_time) < 2:
        # Not enough data
        features['is_intensifying'] = -999
        features['wind_change_approaching_kt'] = -999
        features['intensification_rate_kt_per_day'] = -999
        return features
    
    # Check if intensifying toward closest approach
    winds_before_closest = [w for i, w, t in winds_with_time if i <= closest_approach_idx]
    if len(winds_before_closest) >= 2:
        features['is_intensifying'] = 1 if winds_before_closest[-1] > winds_before_closest[0] else 0
        features['wind_change_approaching_kt'] = winds_before_closest[-1] - winds_before_closest[0]
        
        # Intensification rate
        time_diff = (winds_with_time[closest_approach_idx][2] - winds_with_time[0][2]).total_seconds() / 86400  # days
        if time_diff > 0:
            features['intensification_rate_kt_per_day'] = features['wind_change_approaching_kt'] / time_diff
        else:
            features['intensification_rate_kt_per_day'] = 0
    else:
        features['is_intensifying'] = -999
        features['wind_change_approaching_kt'] = -999
        features['intensification_rate_kt_per_day'] = -999
    
    return features
```

---