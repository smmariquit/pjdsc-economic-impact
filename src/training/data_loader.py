import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ID_COLUMNS: List[str] = ["Year", "Storm", "Province"]


@dataclass
class DataPaths:
    feature_root: Path = Path("Feature_Engineering_Data")
    population_csv: Path = Path("Population_data/population_density_all_years.csv")
    impacts_csv: Path = Path("Impact_data/people_affected_all_years.csv")
    houses_csv: Path = Path("Impact_data/houses_all_years.csv")


def _read_feature_group(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def load_feature_matrix(paths: DataPaths) -> pd.DataFrame:
    """
    Load and merge engineered feature groups on [Year, Storm, Province].

    Expects CSV files:
      - group1/distance_features_group1.csv
      - group2/weather_features_group2.csv
      - group3/intensity_features_group3.csv
      - group6/motion_features_group6.csv
      - group7/interaction_features_group7.csv
      - group8/multistorm_features_group8.csv
    """
    group_files: List[Tuple[str, Path]] = [
        ("group1", paths.feature_root / "group1" / "distance_features_group1.csv"),
        ("group2", paths.feature_root / "group2" / "weather_features_group2.csv"),
        ("group3", paths.feature_root / "group3" / "intensity_features_group3.csv"),
        ("group6", paths.feature_root / "group6" / "motion_features_group6.csv"),
        ("group7", paths.feature_root / "group7" / "interaction_features_group7.csv"),
        ("group8", paths.feature_root / "group8" / "multistorm_features_group8.csv"),
    ]

    merged: pd.DataFrame | None = None
    for group_name, file_path in group_files:
        if not file_path.exists():
            # Skip missing optional groups
            continue
        df = _read_feature_group(file_path)
        # Prefix non-ID columns to avoid accidental name clashes across groups
        non_id_cols = [c for c in df.columns if c not in ID_COLUMNS]
        df = df[ID_COLUMNS + non_id_cols]
        df = df.rename(columns={c: f"{group_name}__{c}" for c in non_id_cols})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=ID_COLUMNS, how="outer")

    if merged is None:
        raise FileNotFoundError("No feature groups found under Feature_Engineering_Data")

    # Sort for stability
    merged = merged.sort_values(ID_COLUMNS).reset_index(drop=True)
    return merged


def load_population(paths: DataPaths) -> pd.DataFrame:
    pop = pd.read_csv(paths.population_csv)
    # Standardize column names
    expected = {"Year", "Province", "Population", "PopulationDensity"}
    missing = expected.difference(set(pop.columns))
    if missing:
        raise ValueError(f"Population CSV missing columns: {missing}")
    pop = pop[list(ID_COLUMNS[0:1]) + ["Province", "Population", "PopulationDensity"]]
    return pop


def load_impacts(paths: DataPaths) -> pd.DataFrame:
    impacts = pd.read_csv(paths.impacts_csv)
    # Coerce affected to numeric; drop rows missing Province or Storm
    impacts = impacts.dropna(subset=["Year", "Storm", "Province"])
    # Some entries are non-numeric like "3.532"; coerce and fill NaN to 0
    impacts["Affected"] = pd.to_numeric(impacts["Affected"], errors="coerce").fillna(0.0)
    # Aggregate duplicate entries by Year, Storm, Province (sum)
    impacts = (
        impacts.groupby(ID_COLUMNS, as_index=False)["Affected"].sum()
    )
    return impacts


def compute_province_vulnerability(impacts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe vulnerability features per Province per Year using
    only STRICTLY PAST years (Year < current_year).
    
    Aggregates all storms within each year first, then computes cumulative stats
    using only complete past years. This prevents within-year leakage when a province
    has multiple storms in the same year.
    
    Returns dataframe with columns: Year, Province, hist_storms, hist_avg_affected,
    hist_max_affected.
    """
    impacts = impacts.copy()
    
    # Aggregate to province-year level: max impact across all storms in that year
    yearly_max = (
        impacts.groupby(["Province", "Year"], as_index=False)
        .agg({"Affected": ["max", "sum", "count"]})
    )
    yearly_max.columns = ["Province", "Year", "max_affected", "total_affected", "storm_count"]
    yearly_max["has_impact"] = (yearly_max["max_affected"] > 0).astype(int)
    yearly_max = yearly_max.sort_values(["Province", "Year"])

    def _per_province_year(g: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative stats from STRICTLY PAST years only."""
        g = g.sort_values("Year").copy()
        
        # Use shift(1) to exclude current year from cumulative stats
        cum_years_with_impact = g["has_impact"].cumsum().shift(1, fill_value=0)
        cum_total_affected = g["total_affected"].cumsum().shift(1, fill_value=0.0)
        cum_max_affected = g["max_affected"].shift(1).expanding().max()
        cum_max_affected = cum_max_affected.fillna(0.0)
        
        # Average affected per impacted year
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_affected = np.where(
                cum_years_with_impact.values > 0,
                cum_total_affected.values / cum_years_with_impact.values,
                0.0
            )
        
        g["hist_storms"] = cum_years_with_impact.values.astype(float)
        g["hist_avg_affected"] = avg_affected
        g["hist_max_affected"] = cum_max_affected.values
        
        return g[["Province", "Year", "hist_storms", "hist_avg_affected", "hist_max_affected"]]
    
    vuln_yearly = yearly_max.groupby("Province", group_keys=False).apply(_per_province_year)
    
    # Expand back to storm level by merging on (Province, Year)
    # All storms in same province-year get the same vulnerability values
    vuln = impacts[["Year", "Storm", "Province"]].drop_duplicates().merge(
        vuln_yearly, on=["Province", "Year"], how="left"
    )
    vuln = vuln.fillna({"hist_storms": 0.0, "hist_avg_affected": 0.0, "hist_max_affected": 0.0})
    vuln = vuln[["Year", "Storm", "Province", "hist_storms", "hist_avg_affected", "hist_max_affected"]]
    
    return vuln


def load_houses(paths: DataPaths) -> pd.DataFrame:
    """
    Load house damage data (Total Houses column from houses_all_years.csv).
    Returns DataFrame with [Year, Storm, Province, Total_Houses].
    """
    df = pd.read_csv(paths.houses_csv)
    
    # Keep only relevant columns
    df = df[["Year", "Storm", "Province", "Total Houses"]].copy()
    df = df.rename(columns={"Total Houses": "Total_Houses"})
    
    return df


def compute_province_vulnerability_houses(houses: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe vulnerability features per Province per Year using
    only STRICTLY PAST years (Year < current_year) for HOUSES data.
    
    Returns dataframe with columns: Year, Province, hist_storms_houses, 
    hist_avg_houses, hist_max_houses.
    """
    houses_data = houses.copy()
    houses_data["Houses"] = houses_data["Total_Houses"]
    
    # Aggregate to province-year level: max houses across all storms in that year
    yearly_max = (
        houses_data.groupby(["Province", "Year"], as_index=False)
        .agg({"Houses": ["max", "sum", "count"]})
    )
    yearly_max.columns = ["Province", "Year", "max_houses", "total_houses", "storm_count"]
    yearly_max["has_impact"] = (yearly_max["max_houses"] > 0).astype(int)
    yearly_max = yearly_max.sort_values(["Province", "Year"])

    def _per_province_year(g: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative stats from STRICTLY PAST years only."""
        g = g.sort_values("Year").copy()
        
        # Use shift(1) to exclude current year from cumulative stats
        cum_years_with_impact = g["has_impact"].cumsum().shift(1, fill_value=0)
        cum_total_houses = g["total_houses"].cumsum().shift(1, fill_value=0.0)
        cum_max_houses = g["max_houses"].shift(1).expanding().max()
        cum_max_houses = cum_max_houses.fillna(0.0)
        
        # Average houses per impacted year
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_houses = np.where(
                cum_years_with_impact.values > 0,
                cum_total_houses.values / cum_years_with_impact.values,
                0.0
            )
        
        g["hist_storms_houses"] = cum_years_with_impact.values.astype(float)
        g["hist_avg_houses"] = avg_houses
        g["hist_max_houses"] = cum_max_houses.values
        
        return g[["Province", "Year", "hist_storms_houses", "hist_avg_houses", "hist_max_houses"]]
    
    vuln_yearly = yearly_max.groupby("Province", group_keys=False).apply(_per_province_year)
    
    # Expand back to storm level by merging on (Province, Year)
    vuln = houses_data[["Year", "Storm", "Province"]].drop_duplicates().merge(
        vuln_yearly, on=["Province", "Year"], how="left"
    )
    vuln = vuln.fillna({"hist_storms_houses": 0.0, "hist_avg_houses": 0.0, "hist_max_houses": 0.0})
    vuln = vuln[["Year", "Storm", "Province", "hist_storms_houses", "hist_avg_houses", "hist_max_houses"]]
    
    return vuln


def build_dataset(paths: DataPaths) -> Dict[str, pd.DataFrame]:
    """
    Build merged dataset containing features, population, vulnerability, and labels.
    Returns dict with keys: features (full merged), labels (impacts), dataset (merged)
    """
    features = load_feature_matrix(paths)
    pop = load_population(paths)
    impacts = load_impacts(paths)
    vuln = compute_province_vulnerability(impacts)

    # Merge pop onto features (Year, Province)
    features_plus = features.merge(pop, on=["Year", "Province"], how="left")
    
    # Merge vulnerability onto features (Year, Storm, Province) - all ID columns
    features_plus = features_plus.merge(vuln, on=ID_COLUMNS, how="left")

    # Merge labels (Year, Storm, Province)
    dataset = features_plus.merge(impacts, on=ID_COLUMNS, how="left")
    dataset["Affected"] = dataset["Affected"].fillna(0.0)
    dataset["has_impact"] = (dataset["Affected"] > 0).astype(int)

    return {"features": features_plus, "labels": impacts, "dataset": dataset}


def build_dataset_houses(paths: DataPaths) -> Dict[str, pd.DataFrame]:
    """
    Build merged dataset for HOUSES prediction containing features, population, 
    vulnerability, and house damage labels.
    Returns dict with keys: features (full merged), labels (houses), dataset (merged)
    """
    features = load_feature_matrix(paths)
    pop = load_population(paths)
    houses = load_houses(paths)
    vuln_houses = compute_province_vulnerability_houses(houses)

    # Merge pop onto features (Year, Province)
    features_plus = features.merge(pop, on=["Year", "Province"], how="left")
    
    # Merge houses vulnerability onto features (Year, Storm, Province)
    features_plus = features_plus.merge(vuln_houses, on=ID_COLUMNS, how="left")

    # Merge house damage labels (Year, Storm, Province)
    dataset = features_plus.merge(houses, on=ID_COLUMNS, how="left")
    dataset["Total_Houses"] = dataset["Total_Houses"].fillna(0.0)
    dataset["has_house_impact"] = (dataset["Total_Houses"] > 0).astype(int)

    return {"features": features_plus, "labels": houses, "dataset": dataset}


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns, excluding identifiers and label columns.
    """
    exclude = set(ID_COLUMNS + [
        "Affected", "has_impact",  # Persons labels
        "Total_Houses", "has_house_impact",  # Houses labels
        "Houses destroyed", "Houses damaged", "Total Houses"  # Raw house columns
    ])
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def save_feature_list(feature_cols: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)


