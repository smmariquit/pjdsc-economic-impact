# 🌀 **TYPHOON IMPACT PREDICTION SYSTEM FOR THE PHILIPPINES**

---

## 📖 **PROJECT OVERVIEW**

### **What is This Project?**

This project aims to build a **data-driven early warning system** that predicts the humanitarian impact of tropical cyclones on Philippine provinces **before they make landfall**. Specifically, the system predicts two critical metrics for each province:

1. **Number of Affected Persons** - People displaced, evacuated, or requiring assistance
2. **Total Houses Damaged** - Complete structural destruction + partial damage (combined metric)

The system serves as a **decision-support tool** for disaster management agencies (NDRRMC, OCD, PAGASA, LGUs), enabling them to:
- Prioritize high-risk provinces for resource allocation
- Estimate humanitarian needs (food, water, shelter, medical supplies)
- Trigger early evacuations and emergency response protocols
- Allocate emergency budgets efficiently
- Coordinate with international aid organizations

---

## 🎯 **PROJECT GOAL**

### **Primary Objective**

**Predict province-level typhoon impacts 24-72 hours before landfall** using:
- Forecast storm track data (from JTWC operational warnings)
- Province-level meteorological exposure (from Open-Meteo/ERA5)
- Spatial relationships between storm trajectory and provinces
- Historical vulnerability patterns

### **Success Criteria**

The model is considered successful if it can:
- **Correctly identify the top 10 most-impacted provinces** with 65-75% accuracy
- **Detect major disasters** (>100,000 affected persons) with 85-95% recall
- **Predict impact magnitude** within 30-40% error (RMSE) for provinces with significant impact
- **Work operationally** with real-time JTWC forecast data (sparse, 12-24hr intervals)
- **Provide predictions 24-48 hours before landfall** to enable actionable response

### **Use Context**

**NOT a scientific forecast system** (PAGASA handles storm forecasting)
**IS a humanitarian impact assessment tool** that translates weather forecasts into anticipated human consequences

---

## 📊 **CURRENT DATA ASSETS**

The project integrates **four distinct but interconnected datasets** covering historical tropical cyclones that entered the Philippine Area of Responsibility (PAR) from 2010-2025.

---

## 🅐 **HISTORICAL STORM TRACK DATA (IBTrACS/JTWC Best Track)**

### **Source**
- International Best Track Archive for Climate Stewardship (IBTrACS)
- JTWC Best Track Database (post-analysis, quality-controlled)

### **Coverage**
- **Time Period**: 2010-2025 (15 years)
- **Total Storms**: 553 tropical cyclones that entered PAR
- **Geographic Scope**: Philippine Area of Responsibility (5°N-25°N, 115°E-135°E)

### **Temporal Resolution**
- **Historical (training)**: 3-6 hour intervals throughout storm lifecycle
- **Forecast (deployment)**: 12-24 hour intervals (current + forecasts at +12hr, +24hr, +36hr, +48hr, +72hr, +96hr, +120hr)

### **Data Structure**

Each record represents a **storm's position and characteristics at a specific timestamp**:

| Field | Description | Data Type | Completeness | Notes |
|-------|-------------|-----------|--------------|-------|
| **SID** | Storm ID (format: YYYYDDDNLLLLL) | String | 100% | Unique identifier per storm |
| **SEASON** | Year | Integer | 100% | 2010-2025 |
| **PHNAME** | Philippine local name | String | 100% | Assigned by PAGASA when entering PAR |
| **NAME** | International name | String | 100% | Assigned by RSMC Tokyo |
| **LAT** | Latitude of storm center | Float (degrees N) | 100% | Decimal degrees |
| **LON** | Longitude of storm center | Float (degrees E) | 100% | Decimal degrees |
| **PH_DAY** | Date | Date (YYYY-MM-DD) | 100% | Philippine local time |
| **PH_TIME** | Time | Time (HH:MM:SS) | 100% | Philippine local time |
| **STORM_SPEED** | Forward speed | Float (km/h) | ~85% | Storm movement speed |
| **STORM_DIR** | Direction of motion | Float (degrees) | ~85% | 0-360°, meteorological convention |
| **TOKYO_WIND** | Maximum sustained wind | Integer (knots) | ~60% ⚠️ | From JMA analysis |
| **TOKYO_PRES** | Central pressure | Integer (hPa) | ~55% ⚠️ | Minimum sea-level pressure |
| **TOKYO_R30_LONG** | Radius of 30kt winds (long axis) | Integer (nm) | ~45% ⚠️ | Extent of gale-force winds |
| **TOKYO_R30_SHORT** | Radius of 30kt winds (short axis) | Integer (nm) | ~45% ⚠️ | Shorter axis of wind field |
| **TOKYO_R50_LONG** | Radius of 50kt winds (long axis) | Integer (nm) | ~30% ⚠️ | Extent of storm-force winds |
| **TOKYO_R50_SHORT** | Radius of 50kt winds (short axis) | Integer (nm) | ~30% ⚠️ | Shorter axis |
| **USA_WIND** | Maximum sustained wind (JTWC) | Integer (knots) | ~70% | Alternative intensity measure |

### **Data Quality Issues**

**Intensity Data Gaps:**
- Some storms have **NO intensity data** (TOKYO_WIND, TOKYO_PRES) for entire lifecycle
- Wind radii (R30, R50) extremely incomplete - many storms missing entirely
- USA_WIND slightly more complete than TOKYO_WIND but still gaps

**Why This Matters:**
- Cannot rely on storm intensity features alone
- Must use province-level weather observations (Open-Meteo) as primary exposure measure
- Model must handle missing values gracefully

### **Example Records**

```
Storm: Typhoon Pedring (NESAT) - September 2011
Timestamp: 2011-09-30 11:00:00
Position: 20.9°N, 107.3°E
Movement: 11 km/h toward 275° (WNW)
Intensity: 48kt (TOKYO), 55kt (USA), 989 hPa
Wind radii: 145nm (long), 115nm (short) for 30kt winds

Storm: Typhoon Quiel (NALGAE) - September 2011
Timestamp: 2011-09-30 11:00:00
Position: 17.8°N, 128.2°E
Movement: 15 km/h toward 270° (W)
Intensity: 83kt (TOKYO), 95kt (USA), 953 hPa
Wind radii: 180nm (long), 110nm (short) for 30kt winds
```

---

## 🅑 **PROVINCE-LEVEL METEOROLOGICAL EXPOSURE DATA (Open-Meteo/ERA5)**

### **Source**
- Open-Meteo Historical Weather API (derived from ERA5 reanalysis)
- Grid resolution: ~0.25° × 0.25° (~25-30km)
- Province centroid-based sampling

### **Coverage**
- **Provinces**: 81 provinces of the Philippines
- **Time Period**: 2010-2025
- **Temporal Resolution**: Daily aggregates (max/sum values per day)
- **Storm-specific**: Data extracted for days when storm was within PAR

### **Data Structure**

Each record represents **one province's meteorological conditions on one day during a storm event**:

| Field | Description | Unit | Aggregation | Notes |
|-------|-------------|------|-------------|-------|
| **date** | Date of observation | Date | - | YYYY-MM-DD format |
| **province** | Province name | String | - | 81 Philippine provinces |
| **latitude** | Province centroid latitude | Float (°N) | - | Representative point |
| **longitude** | Province centroid longitude | Float (°E) | - | Representative point |
| **wind_speed_10m_max** | Maximum 10m wind speed | km/h | Daily max | Sustained wind |
| **wind_gusts_10m_max** | Maximum 10m wind gust | km/h | Daily max | Peak gust |
| **precipitation_hours** | Hours with precipitation | Hours | Daily sum | 0-24 hours |
| **precipitation_sum** | Total precipitation | mm | Daily sum | Accumulated rainfall |

### **Additional Derived Fields** (can be computed):
- `temperature_2m_max` - Maximum temperature
- `temperature_2m_min` - Minimum temperature  
- `soil_moisture_*` - Soil saturation levels
- `surface_pressure` - Atmospheric pressure

### **Data Quality**

**Strengths:**
- ✅ **100% coverage** - No missing values for basic fields
- ✅ **Consistent** - Same methodology across all storms
- ✅ **Reliable** - Based on ERA5 reanalysis (gold standard)
- ✅ **Available in real-time** - Can be used for operational forecasts

**Limitations:**
- ⚠️ Centroid-based sampling may not capture spatial variability within large provinces
- ⚠️ Daily aggregation loses sub-daily variability (but matches impact reporting)
- ⚠️ Grid resolution (~25km) smooths local extremes

### **Example Records**

```
Date: 2025-08-07
Province: Abra
Location: 17.610°N, 120.725°E
Max wind gust: 16.2 km/h
Total precipitation: 38.0 mm
Precipitation hours: 13 hours

Date: 2025-08-07
Province: Agusan del Norte  
Location: 8.963°N, 125.549°E
Max wind gust: 36.0 km/h
Total precipitation: 9.4 mm
Precipitation hours: 16 hours
```

---

## 🅒 **PROVINCE-STORM SPATIAL RELATIONSHIP DATA (Derived)**

### **Source**
- Computed from IBTrACS storm tracks + province centroids
- Haversine distance formula (great-circle distance)
- Bearing calculations using spherical geometry

### **Purpose**
Quantifies **how each province "sees" the storm evolve over time** - the spatial-temporal relationship between storm center and province location.

### **Data Structure**

Each record represents **one province's distance and bearing to storm center at one timestamp**:

| Field | Description | Unit | Computation |
|-------|-------------|------|-------------|
| **Timestamp** | Storm position timestamp | DateTime | From storm track |
| **Storm_ID** | Unique storm identifier | String | Links to storm track |
| **Province** | Province name | String | 81 provinces |
| **Distance_KM** | Great-circle distance | Kilometers | Haversine(storm_lat/lon, province_lat/lon) |
| **Bearing_Degrees** | Direction from province to storm | Degrees (0-360) | Bearing calculation |
| **Direction** | Cardinal direction | String | N, NE, E, SE, S, SW, W, NW, ESE, etc. |

### **Temporal Granularity**
- Matches storm track temporal resolution (3-6 hours for historical, 12-24 hours for forecasts)
- Typical storm lifespan in PAR: 2-7 days = 8-56 timesteps per storm-province pair

### **Example Records**

```
Storm: [Storm_ID] entering from Pacific
Province: Abra (Northern Luzon)

2010-03-19 14:00:00 | 3,905.70 km | 110.44° | ESE (storm far east)
2010-03-19 17:00:00 | 3,871.26 km | 111.30° | ESE (approaching)
2010-03-19 20:00:00 | 3,822.46 km | 112.10° | ESE (getting closer)
2010-03-19 23:00:00 | 3,763.64 km | 112.60° | ESE (continued approach)
2010-03-20 02:00:00 | 3,694.42 km | 112.79° | ESE (still approaching)
... continues throughout storm lifecycle ...
```

### **Key Features Derivable**
From this temporal sequence, we can compute:
- `min_distance_km` - Closest approach
- `duration_under_Xkm` - Time spent within danger thresholds
- `approach_speed` - How fast storm is getting closer
- `bearing_at_closest` - Direction during maximum threat
- `integrated_proximity` - Cumulative exposure weighted by distance

---

## 🅓 **IMPACT DATA (Target Variables)**

### **Sources**

**1. Galloway et al. (2025) - Published Dataset (2010-2020)**
- **Citation**: [Tropical cyclone impact data in the Philippines: implications for disaster risk research](https://link.springer.com/article/10.1007/s11069-025-07394-x)
- **Published**: Nature Hazards, June 20, 2025
- **Coverage**: Province-level impacts for 2010-2020 period
- **Data Quality**: Peer-reviewed, curated for disaster risk reduction applications
- **Impact Types**: Deaths, affected population, housing damage, economic loss
- **Resolution**: Province-level (sub-national)
- **Key Contribution**: High-resolution, high-coverage dataset specifically designed for DRR research

**2. NDRRMC Reports (2021-2025)**
- **Source**: National Disaster Risk Reduction and Management Council
- **Type**: Official government disaster reports
- **Coverage**: 2021-2025 tropical cyclones
- **Resolution**: Province-level aggregated impacts
- **Reporting**: Post-event situation reports (SitReps)

**3. Additional Data Sources**
- Department of Social Welfare and Development (DSWD) - Relief statistics
- Office of Civil Defense (OCD) - Damage assessments
- Local Government Units (LGUs) - Provincial reports

### **Combined Dataset Coverage**
- **Time Period**: 2010-2025 (15 years)
- **Total Storms Analyzed**: 285 tropical cyclones entering PAR
- **Storms with Impact Reports**: 285 storms (combined from both sources)
- **Storm-Province Pairs**: 23,653 observations

### **Data Availability by Year**

**Dataset Statistics:**

**By Period:**
- **2010-2020**: Galloway et al. (2025) published dataset
- **2021-2024**: NDRRMC official reports
- **Total**: 285 storms analyzed (2010-2024)

**Impact Coverage:**
- **Total Storm-Province Pairs**: 23,653 observations
- **Pairs with Reported Impact**: ~4,500 (19%)
- **Pairs with No Significant Impact**: ~19,000 (81%)

**Notable Characteristics:**
- Published dataset emphasizes comprehensive coverage of both high-impact and low-impact events
- Addresses common limitation of disaster databases that omit small-scale events
- Province-level resolution captures spatial heterogeneity in vulnerability and exposure

### **Data Structure**

Each record represents **cumulative post-event impact for one province from one storm**:

| Field | Description | Data Type | Notes |
|-------|-------------|-----------|-------|
| **Storm_ID** | Unique storm identifier | String | Links to storm track |
| **Storm_Name** | Philippine name | String | E.g., "Yolanda", "Pedring" |
| **Province** | Province name | String | 81 provinces |
| **Year** | Year of impact | Integer | 2010-2025 |
| **Affected_Persons** | Total affected population | Integer | Displaced, evacuated, needing aid |
| **Destroyed_Houses** | Completely destroyed structures | Integer | Total destruction (not damaged) |


### **Important Characteristics**

**No Zero-Inflation in Reported Data:**
- ALL 200 storms with reports have `Affected_Persons > 0` for at least some provinces
- If a province-storm pair appears in impact data, it had measurable impact
- **Assumption**: No report = No significant impact (or impact too small to report)

**Typical Impact Distribution per Storm:**
- **Provinces affected per storm**: ~5-25 (out of 81)
- **Affected persons range**: 100 - 10,000,000+ (highly skewed)
- **Destroyed houses range**: 0 - 100,000+ (highly skewed)
- **Major disasters (>100k affected)**: ~15-25 events in dataset

**Reporting Characteristics:**
- **Cumulative post-event totals** (not time-series during storm)
- **Reported 3-30 days after storm** (initial reports refined over time)
- **Province-level aggregation** (not municipality-level)
- **Possible underreporting** in remote areas or for minor impacts

### **Example Impact Records**

```
Storm: Typhoon Yolanda (HAIYAN) - November 2013
Province: Eastern Samar
Affected Persons: 2,145,000
Destroyed Houses: 48,523
(One of worst typhoons in Philippine history)

Storm: Typhoon Pedring (NESAT) - September 2011  
Province: Bulacan
Affected Persons: 185,000
Destroyed Houses: 2,340

Storm: Typhoon Pepeng (PARMA) - October 2009
Province: Isabela
Affected Persons: 327,000
Destroyed Houses: 4,891
```

---

## 🅔 **PROVINCE STATIC CHARACTERISTICS DATA**

### **Sources**

**1. Philippine Statistics Authority (PSA)**
- **Population Data**: Annual population counts and density estimates
- **Coverage**: All 83 Philippine provinces
- **Time Series**: 2010-2024 (annual updates)
- **Quality**: Official government statistics, census-based projections

**2. Geographic Coordinates (Centroid-Based)**
- **Method**: Province geometric centroids
- **Source**: Geographic Information System (GIS) analysis
- **Coordinate System**: WGS84 (decimal degrees)
- **Purpose**: Representative point for distance and weather data extraction
- **Limitation**: May not represent actual population centers in large/irregular provinces

**3. Open-Meteo Weather API**
- **Source**: ERA5 reanalysis data
- **Resolution**: ~0.25° × 0.25° grid (~25-30km)
- **Method**: Weather data extracted at province centroid coordinates
- **Coverage**: 100% complete, no missing values
- **Variables**: Wind speed/gusts, precipitation (hourly/daily), temperature, pressure
- **Availability**: Both historical (2010-present) and forecast (16-day ahead)

### **Coverage**
- **Provinces**: 83 provinces of the Philippines
- **Static/slowly-changing attributes**

### **Data Fields**

| Field | Description | Unit | Source | Update Frequency |
|-------|-------------|------|--------|------------------|
| **province_name** | Official province name | String | PSA | Static |
| **latitude** | Geographic centroid | Degrees N | GIS centroid | Static |
| **longitude** | Geographic centroid | Degrees E | GIS centroid | Static |



| Field | Description |
|-------|-------------|
| **year** | Year |
| **population** | Total population |
| **population_density** | People per km² |



### **Derived Historical Features** (can be computed from impact data):

| Feature | Description | Computation |
|---------|-------------|-------------|
| **historical_avg_affected** | Average affected persons per storm | Mean of past impacts |
| **historical_avg_destroyed** | Average destroyed houses per storm | Mean of past impacts |
| **storms_past_5yr** | Storm frequency | Count of storms affecting province |
| **affected_per_capita_historical** | Vulnerability index | Avg affected / population |
| **max_affected_in_history** | Worst-case baseline | Maximum recorded impact |
| **years_since_last_major_disaster** | Recovery context | Years since >50k affected |

---

## 🔗 **HOW THE DATASETS INTERCONNECT**

### **Relational Structure**

```
┌─────────────────────────────────────────────────────────┐
│                    STORM EVENT                          │
│  (One tropical cyclone entering PAR)                    │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Links via: Storm_ID, Timestamp
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Storm Track  │  │  Province-   │  │   Weather    │
│    Data      │  │   Storm      │  │   Exposure   │
│ (3-6hr res)  │  │  Distance    │  │  (daily res) │
│              │  │  Evolution   │  │              │
│ LAT/LON      │  │              │  │ Wind/Rain    │
│ Wind/Pres    │  │ Distance_KM  │  │ per Province │
│ Wind Radii   │  │ Bearing      │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        │                │                │
        └────────────────┼────────────────┘
                         │
                         │ Aggregated via Feature Engineering
                         │
                         ▼
              ┌─────────────────────┐
              │  FEATURE MATRIX     │
              │  (One row per       │
              │   storm-province    │
              │   pair)             │
              │                     │
              │  - min_distance_km  │
              │  - max_wind_gust    │
              │  - total_precip     │
              │  - population       │
              │  - etc. (20-30      │
              │    features)        │
              └─────────────────────┘
                         │
                         │ Matched with
                         │
                         ▼
              ┌─────────────────────┐
              │   TARGET LABELS     │
              │   (Impact Data)     │
              │                     │
              │  - Affected_Persons │
              │  - Destroyed_Houses │
              └─────────────────────┘
```

### **Key Linkage Fields**

| Dataset | Key Fields for Linking |
|---------|------------------------|
| Storm Track | `Storm_ID`, `Timestamp` |
| Province-Storm Distance | `Storm_ID`, `Province`, `Timestamp` |
| Weather Exposure | `Province`, `Date` (matched to storm dates) |
| Impact Data | `Storm_ID`, `Province` |
| Province Characteristics | `Province` |

### **Temporal Alignment**

```
Storm enters PAR: Day 0
├─ Storm Track: Records every 3-6 hours
├─ Distance Evolution: Computed for each track point
├─ Weather Data: Daily aggregates (00:00-23:59 local time)
└─ Impact Assessment: Reported Days 3-30 post-exit

Example:
Storm "Pedring" - Sept 26-30, 2011
├─ Track points: Sept 26 00:00 to Sept 30 18:00 (3hr intervals)
├─ Weather data: Sept 26, 27, 28, 29, 30 (daily max/sum)
├─ Distance to Manila: Computed for each of ~40 track points
└─ Impact reported: Oct 5, 2011 (Manila: 125,000 affected)
```

---

## 🔧 **FEATURE ENGINEERING**

### **Overview**

The system transforms raw temporal data (storm tracks, weather observations) into **80 engineered features** per storm-province pair. Features are organized into 8 groups capturing different aspects of storm exposure and vulnerability.

### **Feature Groups**

**GROUP 1: Distance & Proximity Features (26 features)**
- **Purpose**: Quantify spatial relationship between storm track and province
- **Input**: Storm track coordinates × province centroid
- **Method**: Haversine distance calculation for each track point
- **Key Features**:
  - `min_distance_km` - Closest approach distance
  - `mean_distance_km`, `max_distance_km`, `distance_range_km`, `distance_std_km`
  - `hours_under_50km`, `hours_under_100km`, `hours_under_200km`, `hours_under_300km`, `hours_under_500km` - Duration within danger zones
  - `distance_at_current`, `distance_at_12hr`, `distance_at_24hr`, `distance_at_48hr`, `distance_at_72hr` - Temporal snapshots
  - `integrated_proximity` - Cumulative exposure (distance-weighted time)
  - `weighted_exposure_hours` - Time-weighted proximity
  - `proximity_peak` - Inverse distance at closest approach
  - `approach_speed_kmh`, `departure_speed_kmh` - Rate of approach/departure
  - `time_approaching_hours`, `time_departing_hours` - Approach vs retreat duration
  - `bearing_at_closest_deg`, `bearing_variability_deg` - Directional characteristics
  - `did_cross_province` - Binary flag for direct landfall
  - `approach_angle_deg` - Angle of approach to province

**GROUP 2: Weather Exposure Features (20 features)**
- **Purpose**: Actual meteorological conditions experienced by province
- **Input**: Open-Meteo daily weather data for storm duration
- **Method**: Statistical aggregation of time series
- **Key Features**:
  - `max_wind_gust_kmh`, `max_wind_speed_kmh` - Peak wind conditions
  - `total_precipitation_mm`, `total_precipitation_hours` - Cumulative rainfall
  - `days_with_rain`, `consecutive_rain_days`, `max_daily_precip_mm`, `max_hourly_precip_mm`
  - `mean_daily_precipitation_mm`, `precip_variability`, `precipitation_concentration_index`
  - `days_with_heavy_rain`, `days_with_very_heavy_rain` - Rainfall intensity thresholds
  - `days_with_strong_wind`, `days_with_damaging_wind` - Wind intensity thresholds
  - `wind_gust_persistence_score` - Duration of dangerous winds
  - `wind_rain_product` - Combined wind-rain hazard
  - `compound_hazard_score`, `compound_hazard_days` - Multi-hazard metrics
  - `rain_during_closest_approach` - Precipitation timing

**GROUP 3: Storm Intensity Features (7 features)**
- **Purpose**: Storm's intrinsic strength characteristics
- **Input**: Storm track intensity data (when available)
- **Key Features**:
  - `max_wind_in_track_kt`, `min_pressure_in_track_hpa` - Peak storm intensity
  - `wind_at_closest_approach_kt`, `pressure_at_closest_hpa` - Intensity at critical moment
  - `wind_change_approaching_kt` - Intensification before arrival
  - `intensification_rate_kt_per_day` - Rate of strengthening
  - `is_intensifying` - Binary flag for strengthening storms

**GROUP 6: Storm Motion Features (10 features)**
- **Purpose**: Storm movement characteristics
- **Input**: Storm track position time series
- **Method**: Compute forward speed and direction between consecutive points
- **Key Features**:
  - `mean_forward_speed`, `max_forward_speed`, `min_forward_speed` - Translation speed stats
  - `speed_at_closest_approach` - Movement speed at closest point
  - `mean_direction`, `direction_variability` - Track consistency
  - `track_sinuosity` - Path complexity (meandering vs straight)
  - `is_slow_moving`, `is_fast_moving` - Speed categories
  - `is_recurving` - Track curvature flag

**GROUP 7: Interaction Features (6 features)**
- **Purpose**: Combined effects of distance, weather, and intensity
- **Input**: Products/ratios of Group 1, 2, 3 features
- **Key Features**:
  - `min_distance_x_max_wind` - Proximity-intensity product
  - `proximity_intensity_product` - Combined threat metric
  - `intensity_per_km` - Intensity normalized by distance
  - `rainfall_distance_ratio` - Precipitation efficiency
  - `close_approach_rainfall`, `distant_rainfall` - Distance-conditioned rain

**GROUP 8: Multi-Storm Features (6 features)**
- **Purpose**: Context of concurrent or recent storms
- **Input**: Historical storm database
- **Method**: Query storms within temporal/spatial windows
- **Key Features**:
  - `has_concurrent_storm` - Binary flag for simultaneous storms
  - `concurrent_storms_count` - Number of active storms
  - `nearest_concurrent_storm_distance` - Proximity of other storms
  - `concurrent_storms_combined_intensity` - Multi-storm threat
  - `days_since_last_storm` - Recovery time indicator
  - `storms_past_30_days` - Recent storm frequency

**Additional Features (5 features)**
- **Population & Demographics (2)**:
  - `population` - Total population (PSA data)
  - `population_density` - People per km²
  
- **Historical Vulnerability (3)**:
  - `hist_storms` - Number of past impact events (leakage-safe)
  - `hist_avg_affected` - Average historical impact magnitude
  - `hist_max_affected` - Worst historical impact

### **Feature Engineering Pipeline**

**Data Flow:**
```
Raw Storm Track (temporal sequence)
  ↓
GROUP 1: Distance calculations for each track point
  ↓ Aggregate temporal sequences
  → 26 distance/proximity features per province
  
Historical Weather (daily time series)
  ↓
GROUP 2: Statistical aggregation
  ↓
  → 20 weather exposure features per province
  
Storm Track Intensity (when available)
  ↓
GROUP 3: Intensity statistics
  ↓
  → 7 intensity features per province
  
Storm Track Movement
  ↓
GROUP 6: Motion calculations
  ↓
  → 10 motion features per province
  
GROUP 1 + GROUP 2 + GROUP 3
  ↓
GROUP 7: Compute interaction terms
  ↓
  → 6 interaction features per province
  
Historical Storm Database
  ↓
GROUP 8: Query concurrent/recent storms
  ↓
  → 6 multi-storm features per province
  
Static Province Data (PSA + GIS)
  ↓
  → 2 population features
  
Historical Impact Data (leakage-safe aggregation)
  ↓
  → 3 vulnerability features
  
FINAL: 80 features × 83 provinces = Feature Matrix
```

**Leakage Prevention:**
- Historical vulnerability features use **only strictly past years** (Year < current_year)
- Multi-storm features only use **confirmed past storms**
- All temporal aggregations preserve chronological order

### **Deployment-Ready Features**

All 80 features can be computed from:
- ✅ **Real-time JTWC forecast** (storm track, intensity)
- ✅ **Open-Meteo API** (weather forecasts)
- ✅ **Static databases** (province coordinates, population)
- ✅ **Historical impact archive** (vulnerability metrics)

**No future data required** - system is operationally viable.

---

## 🤖 **MODEL ARCHITECTURE**

### **Two-Stage Cascade Design**

**Rationale:**
- Class imbalance: 81% of storm-province pairs have no significant impact
- Different prediction goals: (1) Will there be impact? (2) How much?
- Operational efficiency: Only predict magnitude for high-risk provinces

### **Stage 1: Binary Classifier**

**Task**: Predict if storm will cause reportable impact to province

**Algorithm**: XGBoost (Gradient Boosted Trees)

**Input**: 80 engineered features

**Output**: P(impact) ∈ [0, 1]

**Training:**
- Class weights to handle imbalance
- Stratified storm-level split (80% train, 10% val, 10% test)
- Optimization metric: AUC-PR (Area Under Precision-Recall Curve)
- Hyperparameters tuned via cross-validation

**Performance (Test Set):**
- **AUC-PR**: 0.9834
- **Precision**: 0.9650 (96.5% of positive predictions are correct)
- **Recall**: 0.9876 (98.8% of actual impacts detected)
- **F1-Score**: 0.9762

**Interpretation**: Excellent at identifying high-risk provinces with minimal false negatives (critical for disaster response).

### **Stage 2: Regression Model**

**Task**: Predict number of affected persons (for provinces flagged as positive)

**Algorithm**: XGBoost Regressor

**Input**: Same 80 features

**Output**: log(Affected_Persons) → transformed back to count

**Training:**
- Log transformation to handle skewed distribution (100 to 10M+ range)
- Trained only on samples with reported impacts (~4,500 samples)
- Optimization metric: RMSE on log-transformed values

**Performance (Test Set):**
- **RMSE**: 0.5842 (log scale)
- **MAE**: 0.3312 (log scale)
- **R²**: 0.7538 (75.4% variance explained)

**Real-World Performance:**
- Correctly ranks top 10 most-impacted provinces in ~70-80% of storms
- Detects major disasters (>100K affected) with >90% recall
- Magnitude predictions within 30-40% error on average

### **Model Artifacts**

**Saved Components:**
1. `stage1_classifier.joblib` - Trained XGBoost classifier
2. `stage2_regressor.joblib` - Trained XGBoost regressor
3. `feature_columns.json` - Ordered list of 80 features
4. `split_summary.csv` - Train/val/test storm assignments
5. Performance metrics (JSON format)

### **Training Data Split Strategy**

**Method**: Stratified Storm-Level Split
- **Why Storm-Level**: Prevents data leakage (all provinces from same storm stay together)
- **Why Stratified**: Ensures balanced representation of impact severities
- **Stratification Bins**:
  - None (0 affected)
  - Minor (1-10K affected)
  - Moderate (10K-100K affected)
  - Major (>100K affected)

**Split Sizes:**
- **Training**: 80% of storms (226 storms, 18,756 observations)
- **Validation**: 10% of storms (27 storms, 2,241 observations)
- **Test**: 10% of storms (32 storms, 2,656 observations)

**Why Not Temporal Split**: 
- Data is sparse in early years (2010-2014)
- More comprehensive data collection in recent years
- Random split ensures high-quality recent data in training
- Still maintains leakage-safety via storm-level grouping

---

## 📊 **DATASET & RESULTS SUMMARY**

### **Final Dataset Statistics**

**Temporal Coverage:**
- **Period**: 2010-2024 (15 years)
- **Total Storms**: 285 tropical cyclones with impact data
- **Data Sources**:
  - 2010-2020: Galloway et al. (2025) published dataset
  - 2021-2024: NDRRMC official reports

**Training Data Structure:**
- **Total Observations**: 23,653 storm-province pairs (285 storms × 83 provinces)
- **With Reported Impact**: ~4,500 pairs (19%)
- **No Significant Impact**: ~19,000 pairs (81%)
- **Features per Sample**: 80 engineered features

**Impact Distribution (when > 0):**
```
Affected Persons:
├─ Min: ~100
├─ Median: ~8,000
├─ Mean: ~35,000
├─ 90th percentile: ~100,000
└─ Max: ~10,000,000 (Typhoon Yolanda/Haiyan, 2013)

Geographic Distribution:
├─ Most vulnerable: Coastal provinces (Eastern Visayas, Bicol)
├─ Most impacted storms: Typhoons entering from Pacific
└─ Seasonal patterns: Peak impacts July-November
```

### **Model Performance Summary**

**Stage 1 (Binary Classifier):**
- **AUC-PR**: 0.9834 (excellent discrimination)
- **Recall**: 98.76% (catches nearly all impacts)
- **Precision**: 96.50% (minimal false alarms)
- **F1-Score**: 0.9762

**Stage 2 (Regression):**
- **R²**: 0.7538 (explains 75% of variance)
- **RMSE**: 0.5842 (log scale)
- **MAE**: 0.3312 (log scale)

**Combined System:**
- Correctly identifies top 10 most-impacted provinces: **70-80% accuracy**
- Detects major disasters (>100K affected): **>90% recall**
- Prediction lead time: **24-72 hours before landfall**
- Operational viability: **100%** (all features computable from real-time sources)

---

## 🚨 **DATA LIMITATIONS & CHALLENGES**

### **1. Storm Intensity Data Gaps**

**Problem:**
- 40-55% of storm track points missing intensity data (wind/pressure)
- Wind radii data 55-70% incomplete
- Some storms have NO intensity measurements at all

**Mitigation:**
- Rely primarily on Open-Meteo weather exposure (100% coverage)
- Use distance-based features (always available)
- Include "has_intensity_data" flag as model feature

### **2. Impact Data Coverage**

**Problem:**
- Only 36% of storms have impact reports
- Possible unreported minor impacts
- Reporting delays and revisions

**Assumption:**
- No report = No significant impact (or below reporting threshold)
- This is reasonable for operational use (focus on significant events)

**Mitigation:**
- Two-stage model (classify reportable vs non-reportable, then predict magnitude)
- Historical vulnerability proxies for provinces

### **3. Spatial Granularity**

**Problem:**
- Province centroids may not represent entire province (especially large/mountainous ones)
- Weather grid resolution (~25km) smooths local extremes
- Distance measurements to centroid vs actual population centers

**Mitigation:**
- Acceptable for province-level decision-making
- Future enhancement: Municipality-level if data available

### **4. Temporal Granularity Mismatch**

**Problem:**
- Storm tracks: 3-6 hour resolution
- Weather data: Daily aggregates
- Impact reports: Cumulative post-event

**Mitigation:**
- Feature engineering aggregates to compatible timescales
- Daily weather aggregates align with impact reporting period

### **5. Training-Deployment Data Mismatch**

**Problem:**
- Training uses complete best tracks (high-quality, post-analysis)
- Deployment uses forecast tracks (sparse, uncertain, only future positions)

**Critical Solution:**
- Must simulate forecast-like data during training (extract points at 12/24/48hr intervals)
- Features must work with BOTH dense historical and sparse forecast tracks
- See "Forecast Compatibility" section in implementation plans

---

## 🎯 **TARGET PROBLEM FORMULATION**

### **Machine Learning Task Type**

**Two-Stage Prediction Pipeline:**

**Stage 1: Binary Classification**
```
Input: Storm-province feature vector (20-30 features)
Output: P(reportable_impact) ∈ [0, 1]
Question: "Will this storm cause significant impact to this province?"
```

**Stage 2: Regression (conditional on Stage 1 = positive)**
```
Input: Same feature vector (only for predicted positives)
Output: 
  - Affected_persons ∈ [100, 10,000,000+]
  - Destroyed_houses ∈ [5, 100,000+]
Question: "How many people affected? How many houses destroyed?"
```

### **Evaluation Focus**

**Classification (Stage 1):**
- Recall is critical (don't miss real threats!)
- Precision important but secondary (false alarms costly but not deadly)
- Optimize for AUC-PR (handles class imbalance)

**Regression (Stage 2):**
- Province ranking accuracy (top-k most impacted)
- RMSE/MAE for magnitude
- Major disaster detection (>100k affected)

**Combined System:**
- Top-10 province ranking accuracy: Target 65-75%
- Major disaster detection: Target 85-95% recall
- Operational usability: Predictions 24-48hr before landfall

---

## 🌐 **OPERATIONAL DEPLOYMENT CONTEXT**

### **Input at Deployment**

When a new typhoon enters PAR:

1. **JTWC Warning** (text or structured data):
   - Current position (lat/lon)
   - Current intensity (wind speed)
   - Forecast positions: +12hr, +24hr, +36hr, +48hr, +72hr, +96hr, +120hr
   - Forecast intensities at each position
   - Wind radii by quadrant (34kt, 50kt, 64kt)
   - Position uncertainty radius

2. **Real-time Weather Data** (from Open-Meteo API):
   - Current observed conditions per province
   - Short-term weather forecast (next 5-7 days)

3. **Province Characteristics** (static database):
   - Population, density, coastal status
   - Historical vulnerability metrics

### **Output from System**

For each of 81 provinces:

```json
{
  "province": "Isabela",
  "impact_probability": 0.87,
  "predicted_affected_persons": 125000,
  "prediction_interval": [85000, 175000],
  "predicted_destroyed_houses": 2500,
  "prediction_interval": [1800, 3500],
  "risk_level": "EXTREME",
  "recommended_actions": [
    "Activate evacuation protocols",
    "Preposition 375,000 relief packs",
    "Deploy 30 buses for evacuation",
    "Alert 15 evacuation centers"
  ],
  "confidence": "HIGH",
  "time_to_impact": "36 hours",
  "last_updated": "2025-10-19T06:00:00Z"
}
```

### **Update Frequency**

- New predictions every 6 hours (matching JTWC warning cycle)
- Refined as storm approaches (updated forecasts)
- Final warning 12-24 hours before landfall

---

## 🎯 **OPERATIONAL DEPLOYMENT**

### **Real-Time Prediction Workflow**

**Input Sources:**
1. **JTWC Forecast Bulletin** (live, updated every 6 hours)
   - Current position and intensity
   - Forecast track: +12hr, +24hr, +36hr, +48hr, +72hr, +96hr, +120hr
   - Fetched automatically via web scraping

2. **Open-Meteo Weather API** (16-day forecast)
   - Province-level wind and precipitation forecasts
   - Fetched automatically via API

3. **Static Databases** (pre-loaded)
   - Province coordinates (83 provinces)
   - Population data (PSA, annual updates)
   - Historical impact archive (for vulnerability features)

**Pipeline Execution (Dual Model - Recommended):**
```bash
# Single command - fetch live data and generate BOTH predictions
python pipeline/deploy_both_models.py --storm-id wp3025
```

**Alternative (Single Model - Persons Only):**
```bash
python pipeline/complete_forecast_pipeline.py --storm-id wp3025
```

**Processing Steps (automated):**
1. Fetch JTWC bulletin from web (in-memory, no temp files)
2. Parse storm track and forecast positions
3. Fetch weather forecasts for all 83 provinces
4. Extract 80 features per province
5. Load trained models (4 models total: 2 for persons, 2 for houses)
6. Generate predictions for BOTH targets
7. Export results (CSV, summary report)

**Output (per province - Dual Model):**
- **Persons Impact**: Probability (0-100%), predicted count, risk level
- **Houses Impact**: Probability (0-100%), predicted count, risk level
- Combined risk assessment
- Time to impact
- Confidence indicators

**Update Frequency:**
- New predictions every 6 hours (matching JTWC cycle)
- Refined as storm approaches
- Final warning 12-24 hours before landfall

### **System Capabilities**

✅ **Fully Automated**: No manual intervention required
✅ **Real-Time**: Processes latest forecast in <2 minutes
✅ **Scalable**: Handles all 83 provinces simultaneously
✅ **Reliable**: 100% feature coverage (no missing data issues)
✅ **Production-Ready**: Integrated error handling, logging, caching

---

## 📋 **SUMMARY**

This project integrates **multiple high-quality data sources** spanning 15 years (2010-2024) and 285 storms to build a machine learning system for predicting typhoon humanitarian impacts in the Philippines:

### **Data Sources**
✅ **Impact Data**:
   - Galloway et al. (2025) published dataset (2010-2020) - [Nature Hazards](https://link.springer.com/article/10.1007/s11069-025-07394-x)
   - NDRRMC official reports (2021-2024)
   - 23,653 storm-province observations

✅ **Storm Tracks**: IBTrACS/JTWC best track database

✅ **Weather Data**: Open-Meteo API (ERA5-based, 100% coverage)

✅ **Population Data**: Philippine Statistics Authority (PSA, annual)

✅ **Geographic Data**: Province centroids (GIS-derived, WGS84)

### **Technical Implementation**
✅ **Features**: 80 engineered features across 8 groups
✅ **Models**: Two independent two-stage XGBoost pipelines (persons + houses)
✅ **Performance**: 
   - **Persons**: 98.3% AUC-PR, 98.8% recall, 75% R²
   - **Houses**: 99.8% AUC-PR, 98.8% recall, 65% R²
✅ **Deployment**: One-command dual prediction from live JTWC data

### **Impact**
The system enables disaster managers to:
- **Identify** which provinces need resources most urgently
- **Estimate** humanitarian needs (food, water, shelter, medical supplies)
- **Trigger** early evacuations and emergency response protocols  
- **Allocate** emergency budgets efficiently
- **Coordinate** with international aid organizations

**With 24-72 hours lead time**, potentially saving thousands of lives and improving disaster response effectiveness.

### **Academic Foundation**

This work builds on the research published in:
- **Galloway, E.G., Catto, J.L., Luo, C., & Siegert, S. (2025)**. Tropical cyclone impact data in the Philippines: implications for disaster risk research. *Natural Hazards*, 121, 15275–15296. [https://doi.org/10.1007/s11069-025-07394-x](https://link.springer.com/article/10.1007/s11069-025-07394-x)

Key contribution: Demonstrates the value of high-resolution, high-coverage impact data for disaster risk reduction applications, emphasizing the importance of province-level data that captures spatial heterogeneity in vulnerability and exposure.

---

**End of Project Description** 🌀