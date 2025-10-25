from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class TemporalSplits:
    train_years: Tuple[int, int]
    val_years: Tuple[int, int]
    test_years: Tuple[int, int]

    def contains(self, year: int, split: str) -> bool:
        lo, hi = getattr(self, f"{split}_years")
        return lo <= year <= hi


def temporal_split(df: pd.DataFrame, years: TemporalSplits) -> Dict[str, pd.DataFrame]:
    def mask(lo: int, hi: int) -> pd.Series:
        return (df["Year"] >= lo) & (df["Year"] <= hi)

    train = df[mask(*years.train_years)].copy()
    val = df[mask(*years.val_years)].copy()
    test = df[mask(*years.test_years)].copy()
    return {"train": train, "val": val, "test": test}


def stratified_storm_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split by storm (not year) with stratification by impact severity.
    Entire storm-province sets stay together. Randomized to include
    high-quality recent data in training.
    
    Returns dict with keys: train, val, test
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # Compute per-storm severity: max affected across provinces
    storm_severity = df.groupby(["Year", "Storm"])["Affected"].agg(["max", "sum", "count"]).reset_index()
    storm_severity.columns = ["Year", "Storm", "max_affected", "total_affected", "provinces_affected"]
    
    # Stratify into bins: none (0), minor (1-10k), moderate (10k-100k), major (100k+)
    def severity_bin(max_aff: float) -> int:
        if max_aff == 0:
            return 0
        elif max_aff < 10000:
            return 1
        elif max_aff < 100000:
            return 2
        else:
            return 3
    
    storm_severity["severity_bin"] = storm_severity["max_affected"].apply(severity_bin)
    
    # Stratified random split per bin
    np.random.seed(random_state)
    storm_splits = []
    
    for bin_id in sorted(storm_severity["severity_bin"].unique()):
        bin_storms = storm_severity[storm_severity["severity_bin"] == bin_id].copy()
        n = len(bin_storms)
        
        # Shuffle
        indices = np.random.permutation(n)
        
        # Compute split sizes
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # Remainder goes to test
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        bin_storms.loc[bin_storms.index[train_idx], "split"] = "train"
        bin_storms.loc[bin_storms.index[val_idx], "split"] = "val"
        bin_storms.loc[bin_storms.index[test_idx], "split"] = "test"
        
        storm_splits.append(bin_storms)
    
    storm_assignments = pd.concat(storm_splits, ignore_index=True)
    storm_assignments = storm_assignments[["Year", "Storm", "split"]]
    
    # Merge back to full dataset
    df_with_split = df.merge(storm_assignments, on=["Year", "Storm"], how="left")
    
    splits = {
        "train": df_with_split[df_with_split["split"] == "train"].drop(columns=["split"]).copy(),
        "val": df_with_split[df_with_split["split"] == "val"].drop(columns=["split"]).copy(),
        "test": df_with_split[df_with_split["split"] == "test"].drop(columns=["split"]).copy(),
    }
    
    return splits


def summarize_split_counts(df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, part in splits.items():
        total = len(part)
        positives = int((part.get("has_impact", pd.Series([])) > 0).sum()) if "has_impact" in part.columns else np.nan
        unique_storms = part[["Year", "Storm"]].drop_duplicates().shape[0] if "Storm" in part.columns else np.nan
        total_affected = int(part["Affected"].sum()) if "Affected" in part.columns else np.nan
        rows.append({
            "split": name,
            "samples": total,
            "positives": positives,
            "unique_storms": unique_storms,
            "total_affected": total_affected,
        })
    return pd.DataFrame(rows)


