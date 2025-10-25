# 🌀🌀 **GROUP 8: MULTI-STORM & COMPOUND IMPACT FEATURES**

---

## 📋 **CRITICAL SCENARIO: CONCURRENT & RAPID SUCCESSION STORMS**

### **Why This Matters**

The Philippines frequently experiences:
- **Concurrent storms**: 2-3 typhoons active in PAR simultaneously
- **Rapid succession**: New storm arrives before recovery from previous storm
- **Peak season clustering**: Multiple storms within 1-2 weeks (especially Aug-Nov)

### **Operational Impacts**

When storms overlap or occur in rapid succession:
- ✗ Evacuees from first storm still in shelters (no capacity)
- ✗ Rescue teams already deployed elsewhere
- ✗ Emergency supplies depleted
- ✗ Soil already saturated (amplified flooding)
- ✗ Infrastructure weakened from previous storm
- ✗ Media/aid attention split between events
- ✗ Cumulative psychological trauma

**Example Real Events:**
- September 2011: Pedring + Quiel (simultaneous)
- December 2014: Multiple storms within 2 weeks
- November 2020: Rolly + Ulysses (8 days apart, same areas hit)

---

## 🎯 **FEATURE GROUPS**

### **8.1 Concurrent Storm Detection**

| Feature Name | Description | Computation | Purpose |
|--------------|-------------|-------------|---------|
| `has_concurrent_storm` | Another storm active in PAR | Boolean (0/1) | Resource competition flag |
| `concurrent_storms_count` | Number of other active storms | Count | Resource strain level |
| `nearest_concurrent_storm_distance` | Distance to nearest other storm | km | Interaction potential |
| `concurrent_storms_combined_intensity` | Sum of all active storm winds | km/h | Total regional threat |

**Implementation:**
```python
def compute_concurrent_storm_features(current_storm_id, current_storm_date, all_active_storms):
    """
    Detect and quantify concurrent storm activity
    
    Parameters:
    current_storm_id: ID of storm being evaluated
    current_storm_date: Date/time of evaluation
    all_active_storms: DataFrame of all storms with their active periods
    """
    # Find other storms active on this date
    concurrent = all_active_storms[
        (all_active_storms['storm_id'] != current_storm_id) &
        (all_active_storms['start_date'] <= current_storm_date) &
        (all_active_storms['end_date'] >= current_storm_date)
    ]
    
    has_concurrent = 1 if len(concurrent) > 0 else 0
    count = len(concurrent)
    
    if count > 0:
        # Find nearest concurrent storm
        current_position = get_storm_position(current_storm_id, current_storm_date)
        
        distances = []
        intensities = []
        for _, other_storm in concurrent.iterrows():
            other_position = get_storm_position(other_storm['storm_id'], current_storm_date)
            dist = haversine(
                current_position['lat'], current_position['lon'],
                other_position['lat'], other_position['lon']
            )
            distances.append(dist)
            intensities.append(other_storm.get('max_wind', 0))
        
        nearest_distance = min(distances) if distances else 9999
        combined_intensity = sum(intensities)
    else:
        nearest_distance = 9999
        combined_intensity = 0
    
    return {
        'has_concurrent_storm': has_concurrent,
        'concurrent_storms_count': count,
        'nearest_concurrent_storm_distance': nearest_distance,
        'concurrent_storms_combined_intensity': combined_intensity
    }
```

---

### **8.2 Recent Storm History (Rapid Succession)**

| Feature Name | Description | Computation | Purpose |
|--------------|-------------|-------------|---------|
| `days_since_last_storm` | Time since previous storm exited PAR | Days | Recovery time |
| `storms_past_30_days` | Recent storm frequency | Count | Cumulative stress |


**Implementation:**
```python
def compute_recent_storm_history(province_name, current_storm_date, impact_history, storm_history):
    """
    Quantify recent storm impacts on this province
    
    Parameters:
    province_name: Province being evaluated
    current_storm_date: Date of current storm
    impact_history: Historical impacts (BEFORE current storm!)
    storm_history: All storm records
    """
    # Recent time windows
    last_7_days = current_storm_date - pd.Timedelta(days=7)
    last_14_days = current_storm_date - pd.Timedelta(days=14)
    last_30_days = current_storm_date - pd.Timedelta(days=30)
    
    # Storms that affected this province recently
    recent_impacts = impact_history[
        (impact_history['province'] == province_name) &
        (impact_history['date'] < current_storm_date) &
        (impact_history['date'] >= last_30_days)
    ]
    
    # All storms in PAR recently (even if didn't impact this province)
    all_recent_storms = storm_history[
        (storm_history['end_date'] >= last_30_days) &
        (storm_history['end_date'] < current_storm_date)
    ]
    
    # Days since last storm (any storm, anywhere in PAR)
    if len(all_recent_storms) > 0:
        last_storm_exit = all_recent_storms['end_date'].max()
        days_since_last = (current_storm_date - last_storm_exit).days
    else:
        days_since_last = 365  # No recent storms
    
    # Province-specific recent impacts
    affected_7_days = 1 if len(recent_impacts[recent_impacts['date'] >= last_7_days]) > 0 else 0
    affected_14_days = 1 if len(recent_impacts[recent_impacts['date'] >= last_14_days]) > 0 else 0
    
    cumulative_affected = recent_impacts['affected_persons'].sum()
    storms_count_30d = len(all_recent_storms)
    
    return {
        'days_since_last_storm': days_since_last,
        'storms_past_30_days': storms_count_30d,
        'affected_by_storm_past_7_days': affected_7_days,
        'affected_by_storm_past_14_days': affected_14_days,
        'cumulative_affected_past_30_days': cumulative_affected
    }
```

---

### **8.3 Compound Environmental Effects**

| Feature Name | Description | Computation | Purpose |
|--------------|-------------|-------------|---------|
| `soil_saturation_proxy` | Recent rainfall accumulation | Rain in past 14 days | Flood amplification |
| `consecutive_storm_exposure` | Multiple storms in short period | Boolean | Infrastructure fatigue |
| `recovery_time_available` | Days between impacts | Days | Resilience indicator |
| `is_back_to_back_impact` | Hit by storm < 7 days ago | Boolean | Critical compounding |

**Implementation:**
```python
def compute_compound_environmental_effects(province_name, current_storm_date, weather_history, impact_history):
    """
    Environmental compounding from recent storms
    
    Parameters:
    province_name: Province being evaluated
    current_storm_date: Current storm date
    weather_history: Historical weather data
    impact_history: Impact records
    """
    # Get weather from past 14 days
    past_14_days = current_storm_date - pd.Timedelta(days=14)
    recent_weather = weather_history[
        (weather_history['province'] == province_name) &
        (weather_history['date'] >= past_14_days) &
        (weather_history['date'] < current_storm_date)
    ]
    
    # Soil saturation proxy (cumulative recent rainfall)
    soil_saturation_proxy = recent_weather['precipitation_sum'].sum()
    
    # Check for consecutive impacts
    recent_impacts = impact_history[
        (impact_history['province'] == province_name) &
        (impact_history['date'] < current_storm_date) &
        (impact_history['date'] >= current_storm_date - pd.Timedelta(days=30))
    ]
    
    if len(recent_impacts) >= 2:
        consecutive_storm_exposure = 1
        # Calculate recovery time (time between last two impacts)
        sorted_dates = sorted(recent_impacts['date'].values)
        recovery_time = (sorted_dates[-1] - sorted_dates[-2]).days
    else:
        consecutive_storm_exposure = 0
        recovery_time = 365
    
    # Back-to-back impact (< 7 days)
    is_back_to_back = 1 if recovery_time < 7 else 0
    
    return {
        'soil_saturation_proxy': soil_saturation_proxy,
        'consecutive_storm_exposure': consecutive_storm_exposure,
        'recovery_time_available': recovery_time,
        'is_back_to_back_impact': is_back_to_back
    }
```

---

### **8.4 Resource Competition Features**

| Feature Name | Description | Computation | Purpose |
|--------------|-------------|-------------|---------|
| `active_evacuees_in_shelters` | People from previous storm(s) | Count | Shelter capacity |
| `emergency_response_deployed` | Resources committed elsewhere | Boolean | Availability flag |
| `regional_resource_strain_index` | Combined regional demand | Normalized score | System capacity |

**Implementation:**
```python
def compute_resource_competition_features(province_name, current_storm_date, impact_history, concurrent_storms):
    """
    Estimate resource availability based on other demands
    
    Note: This requires more detailed operational data
    Placeholder implementation with reasonable assumptions
    """
    # Estimate active evacuees (assume people stay in shelters 7-14 days)
    past_14_days = current_storm_date - pd.Timedelta(days=14)
    
    recent_provincial_impacts = impact_history[
        (impact_history['province'] == province_name) &
        (impact_history['date'] >= past_14_days) &
        (impact_history['date'] < current_storm_date)
    ]
    
    # Rough estimate: 30% of recently affected still in shelters
    active_evacuees = recent_provincial_impacts['affected_persons'].sum() * 0.3
    
    # Emergency response deployment
    # If concurrent storms OR recent major impact, response is deployed
    has_concurrent = concurrent_storms['concurrent_storms_count'] > 0
    recent_major = (recent_provincial_impacts['affected_persons'] > 10000).any()
    
    emergency_deployed = 1 if (has_concurrent or recent_major) else 0
    
    # Regional strain index (normalized 0-1)
    # Combine: concurrent storms + recent impacts + evacuees
    strain_score = (
        min(concurrent_storms['concurrent_storms_count'] / 3, 1) * 0.4 +
        min(active_evacuees / 50000, 1) * 0.3 +
        min(len(recent_provincial_impacts) / 3, 1) * 0.3
    )
    
    return {
        'active_evacuees_in_shelters': active_evacuees,
        'emergency_response_deployed': emergency_deployed,
        'regional_resource_strain_index': strain_score
    }
```

---

### **8.5 Interaction: Multi-Storm × Current Storm**

| Feature Name | Description | Formula | Purpose |
|--------------|-------------|---------|---------|
| `compound_risk_multiplier` | Amplification from recent impacts | (1 + recent_impacts) × current_threat | Nonlinear compounding |
| `saturated_soil_rain_product` | Flooding amplification | soil_saturation × total_precipitation | Flash flood risk |
| `depleted_capacity_exposure` | Impact on strained system | population_at_risk × resource_strain | Effective vulnerability |

**Implementation:**
```python
def compute_multistorm_interactions(current_features, multistorm_features, weather_features):
    """
    Interaction effects between current storm and multi-storm context
    """
    # Count recent impacts
    recent_impacts = (
        multistorm_features['affected_by_storm_past_7_days'] +
        multistorm_features['affected_by_storm_past_14_days']
    )
    
    # Current storm threat
    current_threat = current_features.get('proximity_intensity_product', 0)
    
    # Compound risk (nonlinear amplification)
    # Each recent impact increases vulnerability by 30-50%
    compound_multiplier = 1 + (recent_impacts * 0.4)
    compound_risk = current_threat * compound_multiplier
    
    # Saturated soil flooding
    soil_sat = multistorm_features['soil_saturation_proxy']
    new_rain = weather_features['total_precipitation_mm']
    saturated_soil_product = (soil_sat / 100) * new_rain  # Normalized
    
    # Capacity-constrained exposure
    pop_at_risk = current_features.get('population_at_risk', 0)
    strain = multistorm_features['regional_resource_strain_index']
    depleted_capacity = pop_at_risk * (1 + strain)
    
    return {
        'compound_risk_multiplier': compound_multiplier,
        'compound_risk_score': compound_risk,
        'saturated_soil_rain_product': saturated_soil_product,
        'depleted_capacity_exposure': depleted_capacity
    }
```

---

## 📊 **GROUP 8 SUMMARY: 22 Multi-Storm Features**

| Subgroup | Feature Count | Key Features |
|----------|---------------|--------------|
| Concurrent Storms | 4 | has_concurrent_storm, concurrent_storms_count |
| Recent History | 5 | days_since_last_storm, affected_by_storm_past_7_days |
| Compound Environmental | 4 | soil_saturation_proxy, is_back_to_back_impact |
| Resource Competition | 3 | active_evacuees_in_shelters, regional_resource_strain |
| Multi-Storm Interactions | 6 | compound_risk_multiplier, saturated_soil_rain_product |

---

## ⚠️ **CRITICAL IMPLEMENTATION NOTES**

### **Data Requirements**

1. **Storm Tracking Database**: Need active periods for ALL storms (not just current one)
2. **Real-time Updates**: Concurrent storm features need live data
3. **Impact Reporting Timeline**: Recent impacts may not be reported yet
4. **Shelter Occupancy Data**: Ideal but often unavailable (use proxies)

### **Operational Challenges**

```python
# Challenge: Recent impact data may not be available yet
# Solution: Use weather-based proxies for very recent events

def estimate_recent_impact_if_missing(province_name, recent_storm_id, weather_data):
    """
    If impact report not available yet, estimate from weather exposure
    """
    storm_weather = weather_data[
        (weather_data['province'] == province_name) &
        (weather_data['storm_id'] == recent_storm_id)
    ]
    
    if len(storm_weather) == 0:
        return {'estimated_affected': 0, 'confidence': 'none'}
    
    max_wind = storm_weather['wind_gusts_10m_max'].max()
    total_rain = storm_weather['precipitation_sum'].sum()
    
    # Simple heuristic (would use actual model in production)
    if max_wind > 90 or total_rain > 200:
        estimated_impact = 'major'
        estimated_affected = 10000  # Conservative estimate
    elif max_wind > 60 or total_rain > 100:
        estimated_impact = 'moderate'
        estimated_affected = 1000
    else:
        estimated_impact = 'minor'
        estimated_affected = 100
    
    return {
        'estimated_affected': estimated_affected,
        'confidence': 'low',
        'impact_level': estimated_impact
    }
```

---

## 🎯 **WHEN TO USE THESE FEATURES**

### **High Priority Scenarios**

✅ **Use multi-storm features when:**
- Peak typhoon season (Aug-Nov)
- Multiple storms in PAR
- Storm within 2 weeks of previous impact
- Operational planning mode

⚠️ **Lower priority when:**
- Off-season (Feb-May)
- First storm of season
- Remote province with no recent impacts

### **Feature Importance by Context**

| Context | Key Features | Importance |
|---------|--------------|------------|
| **Concurrent storms** | has_concurrent_storm, concurrent_storms_count | 🔴 Critical |
| **Rapid succession** | days_since_last_storm, is_back_to_back_impact | 🔴 Critical |
| **Peak season** | storms_past_30_days, soil_saturation_proxy | 🟠 High |
| **Off-season** | All Group 8 features | 🟡 Low |

---

## 📈 **EXPECTED IMPACT ON PREDICTIONS**

### **Scenarios Where This Matters Most**

**Example 1: Typhoon Ulysses (Nov 2020)**
- Hit 8 days after Typhoon Rolly
- Same provinces (Bicol region)
- Soil still saturated, evacuees in shelters
- **Result**: Higher impacts than storm strength alone would predict

**Traditional model prediction:**
```
Without multi-storm features:
- Min distance: 50km
- Max wind: 120 km/h
- Predicted affected: 50,000

With multi-storm features:
- Days since last storm: 8
- Soil saturation: 300mm (recent)
- Back-to-back impact: TRUE
- Compound risk multiplier: 1.4
- Predicted affected: 70,000 (40% higher)
```

**Example 2: September 2011 (Pedring + Quiel)**
- Two strong typhoons simultaneously in PAR
- Resources split between events
- **Result**: Slower response, higher secondary impacts

---

## 🔄 **INTEGRATION WITH EXISTING FEATURES**

### **Updated Total Feature Count**

| Group | Features | Completeness | Importance |
|-------|----------|--------------|------------|
| Group 1: Distance | 26 | ✅ 100% | 🔴 Critical |
| Group 2: Weather | 22 | ✅ 100% | 🔴 Critical |
| Group 3: Storm Intensity | 10 | ⚠️ 40-70% | 🟡 Medium |
| Group 4: Province Vulnerability | 12 | ✅ 100% | 🟠 High |
| Group 5: Temporal | 10 | ✅ 100% | 🟡 Medium |
| Group 6: Storm Motion | 13 | ✅ 95% | 🟠 High |
| Group 7: Interactions | 11 | ✅ 100% | 🟠 High |
| **Group 8: Multi-Storm** | **22** | **✅ 90%** | **🟠 High (seasonal)** |
| **TOTAL** | **126** | **~93%** | - |

---

## 💡 **SIMPLIFIED MVP VERSION**

For initial implementation, focus on these **8 core multi-storm features**:

```python
ESSENTIAL_MULTISTORM_FEATURES = [
    'has_concurrent_storm',           # Boolean: Other storms active?
    'days_since_last_storm',          # Days: Recovery time
    'affected_by_storm_past_7_days',  # Boolean: Very recent impact
    'storms_past_30_days',            # Count: Recent frequency
    'soil_saturation_proxy',          # mm: Recent cumulative rain
    'is_back_to_back_impact',         # Boolean: < 7 days between
    'compound_risk_multiplier',       # Float: Amplification factor
    'regional_resource_strain_index'  # Float 0-1: Capacity strain
]
```

Add the remaining 14 features later if these prove important.

---

## 🚨 **MODEL TRAINING CONSIDERATIONS**

### **Temporal Leakage Risk**

⚠️ **CRITICAL**: When creating training data, multi-storm features MUST be computed as they would be known at prediction time!

```python
def safe_multistorm_features_for_training(storm_date, lookback_window='real_time'):
    """
    Ensure no future information leaks
    
    Parameters:
    lookback_window: 
        - 'real_time': Use only data available before storm_date
        - 'complete': Use all data (ONLY for analysis, not training!)
    """
    if lookback_window == 'real_time':
        # Only use storms that ENDED before current storm START
        prior_storms = storms[storms['end_date'] < storm_date]
        
        # Only use impacts that were REPORTED before prediction time
        # (typically 3-7 days after storm)
        prior_impacts = impacts[impacts['report_date'] < storm_date]
    
    return compute_recent_storm_history(prior_storms, prior_impacts)
```

### **Cross-Validation Strategy**

Must use **strict temporal splits** to avoid leakage:

```python
# CORRECT: Temporal split
train_storms = storms[storms['year'] <= 2018]
val_storms = storms[storms['year'] == 2019]
test_storms = storms[storms['year'] >= 2020]

# WRONG: Random split (leaks multi-storm context!)
# Don't do this!
```

---

## 📝 **DOCUMENTATION FOR USERS**

When presenting predictions to disaster managers:

```
PREDICTION FOR: Province X, Storm Y
Base Risk Score: 0.75
Multi-Storm Adjustment: +0.15 (ELEVATED)

⚠️ COMPOUND RISK FACTORS:
- Province hit by storm 5 days ago (still recovering)
- 200mm rain in past 2 weeks (saturated soil)
- Another typhoon active 800km away (resource competition)

RECOMMENDATION: 
Increase impact estimate by 30-40% due to compounding effects.
Priority: Expedite evacuation, pre-position additional supplies.
```

---

**End of Group 8: Multi-Storm Features** 🌀🌀

**This addresses concurrent/rapid succession storms!**

