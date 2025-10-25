import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load and prepare data
df = pd.read_csv("data/par_storm_summary_filtered_with_local_names.csv")
df['start_utc'] = pd.to_datetime(df['start_utc'])
df['end_utc'] = pd.to_datetime(df['end_utc'])
df['duration_days'] = (df['end_utc'] - df['start_utc']).dt.total_seconds() / (24 * 3600)
df['month'] = df['start_utc'].dt.month
df['season'] = df['month'].apply(lambda x: 'DJF' if x in [12,1,2] 
                                   else 'MAM' if x in [3,4,5]
                                   else 'JJA' if x in [6,7,8]
                                   else 'SON')

print("="*80)
print("PHILIPPINES STORM DATA - EXPLORATORY DATA ANALYSIS (2010-2025)")
print("="*80)
print(f"\nDataset Shape: {df.shape[0]} storms, {df.shape[1]} columns")
print(f"Date Range: {df['year'].min()} to {df['year'].max()}")
print(f"\nColumn Names: {list(df.columns)}")
print("\n" + "="*80)

# ============================================================================
# 1. HOW MANY STORMS HIT THE PHILIPPINES PER YEAR? IS THERE A TREND?
# ============================================================================
plt.figure(1, figsize=(14, 6))

storms_per_year = df.groupby("year").size()

# Calculate trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(storms_per_year.index, storms_per_year.values)
trend_line = slope * storms_per_year.index + intercept

plt.subplot(1, 2, 1)
plt.plot(storms_per_year.index, storms_per_year.values, marker='o', linewidth=2, markersize=6, label='Actual')
plt.plot(storms_per_year.index, trend_line, 'r--', linewidth=2, alpha=0.7, label=f'Trend (slope={slope:.2f}/year)')
plt.title("Number of Storms Hitting the Philippines Per Year", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Number of Storms", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(storms_per_year.index, storms_per_year.values, color='steelblue', alpha=0.7, edgecolor='black')
plt.title("Storm Count Distribution by Year", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Number of Storms", fontsize=11)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

print("\n1. TEMPORAL TREND ANALYSIS")
print("-" * 80)
print(f"Average storms per year: {storms_per_year.mean():.1f} ± {storms_per_year.std():.1f}")
print(f"Minimum: {storms_per_year.min()} storms (Year: {storms_per_year.idxmin()})")
print(f"Maximum: {storms_per_year.max()} storms (Year: {storms_per_year.idxmax()})")
print(f"Trend: {slope:.3f} storms/year (p-value: {p_value:.4f})")
print(f"R-squared: {r_value**2:.3f}")
if p_value < 0.05:
    trend_direction = "increasing" if slope > 0 else "decreasing"
    print(f"⚠️  Statistically significant {trend_direction} trend detected!")
else:
    print(f"ℹ️  No statistically significant trend (p > 0.05)")

# ============================================================================
# 2. WHICH MONTHS/SEASONS HAVE THE MOST STORM ACTIVITY?
# ============================================================================
plt.figure(2, figsize=(14, 6))

storms_per_month = df.groupby('month').size()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.subplot(1, 2, 1)
bars = plt.bar(storms_per_month.index, storms_per_month.values, color='coral', alpha=0.7, edgecolor='black')
# Highlight peak months
max_month = storms_per_month.idxmax()
bars[max_month-1].set_color('darkred')
plt.title("Storm Frequency by Month (2010-2025)", fontsize=14, fontweight='bold')
plt.xlabel("Month", fontsize=11)
plt.ylabel("Total Number of Storms", fontsize=11)
plt.xticks(storms_per_month.index, month_names, rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
storms_per_season = df.groupby('season').size()
season_order = ['DJF', 'MAM', 'JJA', 'SON']
storms_per_season = storms_per_season.reindex(season_order)
colors_season = ['lightblue', 'lightgreen', 'orange', 'brown']
plt.bar(storms_per_season.index, storms_per_season.values, color=colors_season, alpha=0.7, edgecolor='black')
plt.title("Storm Frequency by Season", fontsize=14, fontweight='bold')
plt.xlabel("Season (DJF=Winter, MAM=Spring, JJA=Summer, SON=Fall)", fontsize=9)
plt.ylabel("Total Number of Storms", fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

print("\n2. SEASONAL PATTERNS")
print("-" * 80)
print("Monthly Distribution:")
for month_num, count in storms_per_month.items():
    print(f"  {month_names[month_num-1]:>3}: {count:>3} storms ({count/len(df)*100:>5.1f}%)")
print(f"\nPeak Month: {month_names[max_month-1]} with {storms_per_month.max()} storms")
print(f"\nSeasonal Distribution:")
for season, count in storms_per_season.items():
    print(f"  {season}: {count:>3} storms ({count/len(df)*100:>5.1f}%)")

# ============================================================================
# 3. STORM DURATION ANALYSIS
# ============================================================================
plt.figure(3, figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(df['duration_days'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(df['duration_days'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["duration_days"].mean():.1f} days')
plt.axvline(df['duration_days'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["duration_days"].median():.1f} days')
plt.title("Distribution of Storm Durations", fontsize=14, fontweight='bold')
plt.xlabel("Duration (days)", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
duration_by_month = df.groupby('month')['duration_days'].mean()
plt.plot(duration_by_month.index, duration_by_month.values, marker='o', linewidth=2, markersize=8, color='green')
plt.title("Average Storm Duration by Month", fontsize=14, fontweight='bold')
plt.xlabel("Month", fontsize=11)
plt.ylabel("Average Duration (days)", fontsize=11)
plt.xticks(duration_by_month.index, month_names, rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()

print("\n3. STORM DURATION ANALYSIS")
print("-" * 80)
print(f"Mean duration: {df['duration_days'].mean():.2f} days")
print(f"Median duration: {df['duration_days'].median():.2f} days")
print(f"Std deviation: {df['duration_days'].std():.2f} days")
print(f"Shortest storm: {df['duration_days'].min():.2f} days ({df.loc[df['duration_days'].idxmin(), 'name']})")
print(f"Longest storm: {df['duration_days'].max():.2f} days ({df.loc[df['duration_days'].idxmax(), 'name']})")

# ============================================================================
# 4. DATA QUALITY CHECK
# ============================================================================
print("\n4. DATA QUALITY ASSESSMENT")
print("-" * 80)
print(f"Missing values:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nStorms with duration <= 0 days: {(df['duration_days'] <= 0).sum()}")
if (df['duration_days'] <= 0).any():
    print("⚠️  Warning: Some storms have zero or negative duration!")
    print(df[df['duration_days'] <= 0][['year', 'name', 'start_utc', 'end_utc', 'duration_days']])

# ============================================================================
# 5. YEAR-OVER-YEAR COMPARISON
# ============================================================================
plt.figure(4, figsize=(14, 6))

plt.subplot(1, 2, 1)
yearly_duration = df.groupby('year')['duration_days'].mean()
plt.bar(yearly_duration.index, yearly_duration.values, color='teal', alpha=0.7, edgecolor='black')
plt.title("Average Storm Duration by Year", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Average Duration (days)", fontsize=11)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
# Storms per month over years (heatmap-style)
monthly_yearly = df.groupby(['year', 'month']).size().unstack(fill_value=0)
im = plt.imshow(monthly_yearly.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
plt.colorbar(im, label='Number of Storms')
plt.title("Storm Activity Heatmap (Year vs Month)", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Month", fontsize=11)
plt.yticks(range(12), month_names)
plt.xticks(range(len(monthly_yearly.index)), monthly_yearly.index, rotation=45)
plt.tight_layout()

print("\n5. INSIGHTS FOR ECONOMIC IMPACT MODELING")
print("-" * 80)
print(f"• Total storms analyzed: {len(df)}")
print(f"• Peak season: {storms_per_season.idxmax()} ({storms_per_season.max()} storms)")
print(f"• Peak month: {month_names[max_month-1]} ({storms_per_month.max()} storms)")
print(f"• Storm frequency shows {'an increasing' if slope > 0 else 'a decreasing'} trend")
print(f"• Average {storms_per_year.mean():.0f} storms per year to consider for annual risk assessment")
print(f"• Typical storm duration: {df['duration_days'].median():.1f} days (median)")
print("\n" + "="*80)
print("EDA Complete! All figures displayed.")
print("="*80)

plt.show()