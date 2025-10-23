# Weather API Fix - Historical vs Forecast Data

## Problem
❌ **Error**: Historical storms (2023-2024) failed with:
```
Parameter 'start_date' is out of allowed range from 2025-07-22 to 2025-11-07
```

The Open-Meteo **forecast** API only accepts dates within a 16-day window (past 5 days to +10 days future).

## Solution
✅ **Auto-detection of date type**:
- If storm dates are **>5 days in the past** → Use Historical Weather API
- If storm dates are **current/future** → Use Forecast Weather API

## Technical Changes

### File Modified
`model/Final_Transform/pipeline/fetch_forecast_weather.py`

### Code Added
```python
# Determine if we need historical or forecast API
start_dt_check = pd.to_datetime(start_date)
today = pd.Timestamp.now().normalize()

# Use historical API if start date is more than 5 days in the past
is_historical = (today - start_dt_check).days > 5

if is_historical:
    url = "https://archive-api.open-meteo.com/v1/archive"
    print(f"   Using HISTORICAL weather API (dates are in the past)")
else:
    url = "https://api.open-meteo.com/v1/forecast"
    print(f"   Using FORECAST weather API (dates are current/future)")
```

## API Endpoints

### Historical Weather Archive API
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Date Range**: January 1940 to ~5 days ago
- **Use Case**: Analyzing past storms (Kristine, Leon, Enteng, etc.)
- **Data**: Actual recorded weather observations

### Weather Forecast API  
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Date Range**: ~5 days ago to +16 days future
- **Use Case**: Live typhoon predictions, current threats
- **Data**: Forecasted weather predictions

## Testing

Now you can analyze **any storm**:
- ✅ **Historical Storms (2023-2024)**: Kristine, Leon, Enteng, Carina, Julian, Egay, Betty, Pepito, Nika, Marce
- ✅ **Live Storms (2025)**: Current typhoons threatening the Philippines
- ✅ **Custom JTWC Bulletins**: Paste any JTWC forecast text

## What Happens Now

1. User selects a historical storm (e.g., "Kristine - Oct 18, 2024")
2. System extracts dates: `2024-10-18` to `2024-10-24`
3. **Auto-detects**: These dates are >5 days ago (historical)
4. **Switches**: Uses Archive API instead of Forecast API
5. **Success**: Fetches actual weather data from October 2024
6. Continues with feature engineering → ML prediction → results

---

**Date Fixed**: October 23, 2025  
**Root Cause**: API endpoint hardcoded to forecast-only  
**Status**: ✅ Resolved - Auto-switching implemented
