# 🌐 JTWC Data Source Guide

## Overview

The storm forecast data used by this system comes from the **Joint Typhoon Warning Center (JTWC)**, a U.S. Navy and Air Force organization that provides tropical cyclone forecasts for the Western Pacific and Indian Ocean regions.

---

## 📡 Data Source

### **Primary Source: JTWC Web Products**

**URL Format:** `https://www.metoc.navy.mil/jtwc/products/wpXXYYweb.txt`

Where:
- `XX` = Storm number (e.g., `30` for Tropical Storm 30W)
- `YY` = Storm designation/sequence (e.g., `25`)

**Example:**
- Current Storm: `https://www.metoc.navy.mil/jtwc/products/wp3025web.txt`
- Sample File: `storm_forecast.txt` (snapshot from October 19, 2025)

### **Update Frequency**

- **Regular Updates:** Every 6 hours
- **Next Update Times:** Listed at end of bulletin (e.g., "NEXT WARNINGS AT 222100Z, 230300Z...")
- **Format:** Consistent across all bulletins

---

## 📋 Bulletin Format

### **Standard JTWC Format:**

```
WTPN31 PGTW DDHHMM
MSGID/GENADMIN/JOINT TYPHOON WRNCEN PEARL HARBOR HI//
SUBJ/TROPICAL STORM XXW (NAME) WARNING NR XXX//
RMKS/

WARNING POSITION:
DDHHMM Z --- NEAR XX.XN XXX.XE
MOVEMENT PAST SIX HOURS - XXX DEGREES AT XX KTS

PRESENT WIND DISTRIBUTION:
MAX SUSTAINED WINDS - XXX KT, GUSTS XXX KT

FORECASTS:
12 HRS, VALID AT: DDHHMM Z --- XX.XN XXX.XE
24 HRS, VALID AT: DDHHMM Z --- XX.XN XXX.XE
36 HRS, VALID AT: DDHHMM Z --- XX.XN XXX.XE
...

REMARKS:
[Additional information]
NEXT WARNINGS AT ...
```

---

## 🔄 Fetching Live Data

### **Option 1: Manual Download**

```bash
# Download latest bulletin
curl https://www.metoc.navy.mil/jtwc/products/wp3025web.txt -o storm_forecast.txt

# Run pipeline
python pipeline/complete_forecast_pipeline.py --forecast storm_forecast.txt
```

### **Option 2: Automated Script (Recommended)**

Create `fetch_jtwc_bulletin.py`:

```python
import requests
from datetime import datetime

def fetch_jtwc_bulletin(storm_id, output_file="storm_forecast.txt"):
    """
    Fetch latest JTWC bulletin for a storm.
    
    Args:
        storm_id: Storm identifier (e.g., "wp3025" for Tropical Storm 30W)
        output_file: Where to save the bulletin
    
    Returns:
        Path to saved file or None if failed
    """
    url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded bulletin from: {url}")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Timestamp: {datetime.now().isoformat()}")
        
        return output_file
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to fetch bulletin: {e}")
        return None

# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--storm-id", default="wp3025", help="JTWC storm ID")
    parser.add_argument("--output", default="storm_forecast.txt", help="Output file")
    args = parser.parse_args()
    
    fetch_jtwc_bulletin(args.storm_id, args.output)
```

**Usage:**
```bash
# Fetch latest bulletin
python fetch_jtwc_bulletin.py --storm-id wp3025

# Run pipeline on fresh data
python pipeline/complete_forecast_pipeline.py --forecast storm_forecast.txt
```

### **Option 3: Integrated Pipeline**

Add to `complete_forecast_pipeline.py`:

```python
def fetch_live_forecast(storm_id):
    """Fetch latest JTWC bulletin before processing."""
    import requests
    
    url = f"https://www.metoc.navy.mil/jtwc/products/{storm_id}web.txt"
    response = requests.get(url)
    response.raise_for_status()
    
    # Save to temporary file
    forecast_file = f"forecast_{storm_id}.txt"
    with open(forecast_file, 'w') as f:
        f.write(response.text)
    
    return forecast_file
```

---

## 📊 Data Comparison

### **Sample File (storm_forecast.txt)**

- **Date:** October 19, 2025, 09:00Z
- **Storm:** Tropical Storm 30W (FENGSHEN)
- **Location:** 14.8°N, 119.7°E (near Manila)
- **Winds:** 35 kt (forecast up to 65 kt)
- **Status:** Approaching Philippines

### **Live Data (Current)**

As of the [latest update from JTWC](https://www.metoc.navy.mil/jtwc/products/wp3025web.txt):

- **Date:** October 22, 2025, 15:00Z
- **Storm:** Tropical Storm 30W (FENGSHEN)  
- **Location:** 17.2°N, 109.6°E (northeast of Da Nang, Vietnam)
- **Winds:** 45 kt (weakening)
- **Status:** Moving west-southwest, dissipating over land

**Note:** Live data updates every 6 hours. The format remains consistent, allowing our parser to work with both historical and current bulletins.

---

## 🔍 Finding Active Storms

### **JTWC Main Page**

**URL:** `https://www.metoc.navy.mil/jtwc/jtwc.html`

**Active Storm Products:**
- Each active storm has a product page
- Format: `wpXXYYweb.txt` (Western Pacific)
- Also: `ioXXYY` (Indian Ocean), `shXXYY` (Southern Hemisphere)

### **Storm Identification**

```bash
# List all active Western Pacific storms
curl -s https://www.metoc.navy.mil/jtwc/jtwc.html | grep -o "wp[0-9]*web.txt"

# Download specific storm
curl https://www.metoc.navy.mil/jtwc/products/wp3025web.txt -o forecast.txt
```

---

## ⚙️ Pipeline Compatibility

### **Format Consistency**

Our parser (`parse_jtwc_forecast.py`) handles:

✅ **Standard Fields:**
- WARNING POSITION (current location)
- FORECASTS (12hr, 24hr, 36hr, 48hr, 72hr, 96hr, 120hr)
- MAX SUSTAINED WINDS
- MOVEMENT vectors
- Timestamps (DDHHMM Z format)

✅ **Variable Elements:**
- Different storm numbers/names
- Different forecast horizons
- Optional fields (pressure, wind radii)
- Storm status (intensifying, dissipating)

✅ **Edge Cases:**
- Storms dissipating over land
- Missing forecast points
- Variable wind radii data

### **Testing with Live Data**

```bash
# Test 1: Parse live bulletin
python pipeline/parse_jtwc_forecast.py \
  --input storm_forecast_live.txt \
  --summary

# Test 2: Full pipeline with live data
python pipeline/complete_forecast_pipeline.py \
  --forecast storm_forecast_live.txt \
  --output-dir output_live

# Test 3: Compare with sample
diff storm_forecast.txt storm_forecast_live.txt
```

---

## 🔄 Automated Workflow

### **Cron Job Example**

```bash
#!/bin/bash
# fetch_and_predict.sh

STORM_ID="wp3025"
OUTPUT_DIR="output_$(date +%Y%m%d_%H%M)"

# Fetch latest bulletin
curl https://www.metoc.navy.mil/jtwc/products/${STORM_ID}web.txt \
  -o storm_forecast_latest.txt

# Check if file changed
if cmp -s storm_forecast_latest.txt storm_forecast_previous.txt; then
  echo "No changes, skipping"
  exit 0
fi

# Run prediction pipeline
python pipeline/complete_forecast_pipeline.py \
  --forecast storm_forecast_latest.txt \
  --output-dir ${OUTPUT_DIR}

# Archive
cp storm_forecast_latest.txt storm_forecast_previous.txt

echo "Complete! Results in ${OUTPUT_DIR}"
```

**Cron Schedule (every 6 hours):**
```cron
0 */6 * * * /path/to/fetch_and_predict.sh
```

---

## 📚 JTWC Resources

### **Official Documentation**

- **Main Site:** https://www.metoc.navy.mil/jtwc/jtwc.html
- **Product Description:** https://www.metoc.navy.mil/jtwc/products.html
- **Format Guide:** Standard military message format (MSGID/GENADMIN)

### **Related Products**

- **Graphical:** `wpXXYY.gif` - Track forecast images
- **KML:** `wpXXYY.kml` - Google Earth format
- **RSS:** Storm alerts feed
- **Archives:** Historical bulletins available

### **Alternative Sources**

- **PAGASA (Philippines):** Local warnings in Philippine area
- **IBTrACS:** Historical best track data
- **NOAA/NHC:** Additional Pacific coverage

---

## ⚠️ Important Notes

### **Data Usage**

1. **Public Domain:** JTWC products are U.S. government data (public domain)
2. **No API:** Direct web scraping required (no official API)
3. **Rate Limits:** Be respectful, cache results, don't spam requests
4. **Disclaimer:** Forecasts are predictions, not guarantees

### **Best Practices**

```python
# ✅ Good: Cache and reuse
response = requests.get(url, timeout=10)
with open('cache.txt', 'w') as f:
    f.write(response.text)

# ✅ Good: Check for changes
etag = response.headers.get('ETag')
if etag == cached_etag:
    use_cache()

# ❌ Bad: Polling too frequently
while True:
    fetch_bulletin()  # Don't do this!
    time.sleep(60)    # JTWC updates every 6 hours
```

### **Error Handling**

```python
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Storm bulletin not found (may have dissipated)")
    else:
        print(f"HTTP error: {e}")
except requests.exceptions.Timeout:
    print("Request timed out, try again")
except requests.exceptions.ConnectionError:
    print("Network error, check connection")
```

---

## 🔧 Troubleshooting

### **Issue: Bulletin Not Found (404)**

**Cause:** Storm may have dissipated or number changed

**Solution:**
```bash
# Check JTWC main page for active storms
curl https://www.metoc.navy.mil/jtwc/jtwc.html | grep "wp.*web.txt"
```

### **Issue: Parse Errors**

**Cause:** Format variations or incomplete bulletin

**Solution:**
```bash
# Validate bulletin format
python pipeline/parse_jtwc_forecast.py \
  --input problem_bulletin.txt \
  --summary 2>&1 | tee parse_log.txt
```

### **Issue: Outdated Forecasts**

**Cause:** Using cached/old bulletin

**Solution:**
```bash
# Check bulletin timestamp
head -5 storm_forecast.txt
# Look for DDHHMM in first lines

# Force refresh
rm storm_forecast.txt
curl https://www.metoc.navy.mil/jtwc/products/wp3025web.txt -o storm_forecast.txt
```

---

## 📝 Citation

When using JTWC data, please acknowledge:

```
Data Source: Joint Typhoon Warning Center (JTWC)
U.S. Navy and Air Force
URL: https://www.metoc.navy.mil/jtwc/
```

---

**Last Updated:** 2025-10-23  
**JTWC URL:** https://www.metoc.navy.mil/jtwc/products/wp3025web.txt  
**Sample File:** `storm_forecast.txt` (2025-10-19 snapshot)


