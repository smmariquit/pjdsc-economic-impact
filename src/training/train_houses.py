from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from training.data_loader import (
    DataPaths,
    build_dataset_houses,
    save_feature_list,
    select_feature_columns,
)
from training.split_utils import (
    TemporalSplits,
    stratified_storm_split,
    summarize_split_counts,
    temporal_split,
)


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def train_classifier(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Pipeline:
    pos_weight = max(1.0, (len(y) - y.sum()) / max(1.0, y.sum()))
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        tree_method="hist",
    )

    clf = Pipeline(steps=[
        ("pre", build_preprocessor(feature_cols)),
        ("model", model),
    ])
    clf.fit(X[feature_cols], y)
    return clf


def train_regressor(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Pipeline:
    y_log = np.log1p(y.astype(float))
    model = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
    )
    reg = Pipeline(steps=[
        ("pre", build_preprocessor(feature_cols)),
        ("model", model),
    ])
    reg.fit(X[feature_cols], y_log)
    return reg


def evaluate_classifier(clf: Pipeline, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict[str, float]:
    proba = clf.predict_proba(X[feature_cols])[:, 1]
    ap = average_precision_score(y, proba)
    # F2 @ threshold chosen by maximizing F2 on PR curve
    precision, recall, thresholds = precision_recall_curve(y, proba)
    beta2 = 2.0
    f2_scores = (1 + beta2**2) * (precision * recall) / np.clip((beta2**2 * precision + recall), 1e-12, None)
    best_idx = int(np.nanargmax(f2_scores))
    best_thresh = thresholds[max(0, best_idx - 1)] if len(thresholds) > 0 else 0.5
    y_hat = (proba >= best_thresh).astype(int)
    f1 = f1_score(y, y_hat)
    return {"auc_pr": float(ap), "f1@best_f2": float(f1), "best_threshold": float(best_thresh)}


def evaluate_regressor(reg: Pipeline, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict[str, float]:
    y_log = np.log1p(y.astype(float))
    y_pred_log = reg.predict(X[feature_cols])
    rmse_log = float(np.sqrt(np.mean((y_log - y_pred_log) ** 2)))
    mae_log = float(np.mean(np.abs(y_log - y_pred_log)))
    return {"rmse_log": rmse_log, "mae_log": mae_log}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house damage prediction models")
    parser.add_argument("--split_mode", type=str, default="stratified_storm", choices=["temporal", "stratified_storm"])
    parser.add_argument("--train_range", type=str, default="2010-2018", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--val_range", type=str, default="2019-2020", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--test_range", type=str, default="2021-2024", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Only for stratified mode")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Only for stratified mode")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Only for stratified mode")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts_houses")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("TRAINING: HOUSE DAMAGE PREDICTION MODELS")
    print("="*70 + "\n")

    paths = DataPaths()
    built = build_dataset_houses(paths)
    dataset = built["dataset"]
    feature_cols = select_feature_columns(dataset)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples with house damage: {dataset['has_house_impact'].sum()} / {len(dataset)}")
    print()

    # Persist feature list
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(feature_cols, out_dir / "feature_columns_houses.json")

    # Choose split strategy
    if args.split_mode == "temporal":
        def parse_range(s: str) -> Tuple[int, int]:
            lo, hi = s.split("-")
            return (int(lo), int(hi))
        splits = TemporalSplits(
            train_years=parse_range(args.train_range),
            val_years=parse_range(args.val_range),
            test_years=parse_range(args.test_range),
        )
        # Note: temporal_split expects 'Affected' column, need to adapt
        # For houses, we'll use stratified by default
        print("⚠️  Warning: Temporal split expects 'Affected' column. Using stratified instead.")
        parts = stratified_storm_split(
            dataset.rename(columns={"Total_Houses": "Affected"}),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=args.random_state,
        )
        # Rename back
        for key in parts:
            parts[key] = parts[key].rename(columns={"Affected": "Total_Houses"})
    else:  # stratified_storm
        # Need to temporarily rename for stratified_storm_split
        parts = stratified_storm_split(
            dataset.rename(columns={"Total_Houses": "Affected"}),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=args.random_state,
        )
        # Rename back
        for key in parts:
            parts[key] = parts[key].rename(columns={"Affected": "Total_Houses"})

    # Save split summary (adapt to work with houses data)
    summary_dataset = dataset.rename(columns={"Total_Houses": "Affected"})
    summary_parts = {k: v.rename(columns={"Total_Houses": "Affected"}) for k, v in parts.items()}
    summary = summarize_split_counts(summary_dataset, summary_parts)
    summary.to_csv(out_dir / "split_summary_houses.csv", index=False)

    print("="*70)
    print("STAGE 1: HOUSE DAMAGE CLASSIFIER")
    print("="*70)
    print("Training classifier (has_house_impact yes/no)...")
    
    # Stage 1: Classification
    clf = train_classifier(parts["train"], parts["train"]["has_house_impact"].astype(int), feature_cols)
    metrics_val = evaluate_classifier(clf, parts["val"], parts["val"]["has_house_impact"].astype(int), feature_cols)
    metrics_test = evaluate_classifier(clf, parts["test"], parts["test"]["has_house_impact"].astype(int), feature_cols)

    print(f"✓ Validation AUC-PR: {metrics_val['auc_pr']:.4f}")
    print(f"✓ Test AUC-PR: {metrics_test['auc_pr']:.4f}")
    print(f"✓ Test F1: {metrics_test['f1@best_f2']:.4f}\n")

    with (out_dir / "clf_metrics_val_houses.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_val, f, indent=2)
    with (out_dir / "clf_metrics_test_houses.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)

    print("="*70)
    print("STAGE 2: HOUSE DAMAGE REGRESSOR")
    print("="*70)
    print("Training regressor (total houses damaged)...")
    
    # Stage 2: Regression (train on true positives in train set)
    train_pos = parts["train"][parts["train"]["has_house_impact"] == 1]
    reg = train_regressor(train_pos, train_pos["Total_Houses"].astype(float), feature_cols)

    # Evaluate regression on positives in val/test
    val_pos = parts["val"][parts["val"]["has_house_impact"] == 1]
    test_pos = parts["test"][parts["test"]["has_house_impact"] == 1]
    reg_val = evaluate_regressor(reg, val_pos, val_pos["Total_Houses"].astype(float), feature_cols)
    reg_test = evaluate_regressor(reg, test_pos, test_pos["Total_Houses"].astype(float), feature_cols)
    
    print(f"✓ Validation RMSE (log): {reg_val['rmse_log']:.4f}")
    print(f"✓ Test RMSE (log): {reg_test['rmse_log']:.4f}")
    print(f"✓ Test MAE (log): {reg_test['mae_log']:.4f}\n")
    
    with (out_dir / "reg_metrics_val_houses.json").open("w", encoding="utf-8") as f:
        json.dump(reg_val, f, indent=2)
    with (out_dir / "reg_metrics_test_houses.json").open("w", encoding="utf-8") as f:
        json.dump(reg_test, f, indent=2)

    # Persist models
    import joblib

    joblib.dump(clf, out_dir / "stage1_classifier_houses.joblib")
    joblib.dump(reg, out_dir / "stage2_regressor_houses.joblib")

    print("="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nArtifacts saved to: {out_dir.resolve()}\n")
    print("Models:")
    print(f"  - stage1_classifier_houses.joblib")
    print(f"  - stage2_regressor_houses.joblib")
    print("\nMetrics:")
    print(f"  - Classifier Test AUC-PR: {metrics_test['auc_pr']:.4f}")
    print(f"  - Regressor Test RMSE: {reg_test['rmse_log']:.4f}")
    print()


if __name__ == "__main__":
    main()


