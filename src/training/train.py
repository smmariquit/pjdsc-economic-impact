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
    build_dataset,
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
        n_jobs=0,
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
        n_jobs=0,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_mode", type=str, default="stratified_storm", choices=["temporal", "stratified_storm"])
    parser.add_argument("--train_range", type=str, default="2010-2018", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--val_range", type=str, default="2019-2020", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--test_range", type=str, default="2021-2024", help="Only for temporal mode: YYYY-YYYY")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Only for stratified mode")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Only for stratified mode")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Only for stratified mode")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    paths = DataPaths()
    built = build_dataset(paths)
    dataset = built["dataset"]
    feature_cols = select_feature_columns(dataset)

    # Persist feature list
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(feature_cols, out_dir / "feature_columns.json")

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
        parts = temporal_split(dataset, splits)
    else:  # stratified_storm
        parts = stratified_storm_split(
            dataset,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=args.random_state,
        )

    summary = summarize_split_counts(dataset, parts)
    summary.to_csv(out_dir / "split_summary.csv", index=False)

    # Stage 1: Classification
    clf = train_classifier(parts["train"], parts["train"]["has_impact"].astype(int), feature_cols)
    metrics_val = evaluate_classifier(clf, parts["val"], parts["val"]["has_impact"].astype(int), feature_cols)
    metrics_test = evaluate_classifier(clf, parts["test"], parts["test"]["has_impact"].astype(int), feature_cols)

    with (out_dir / "clf_metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_val, f, indent=2)
    with (out_dir / "clf_metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)

    # Stage 2: Regression (train on true positives in train set)
    train_pos = parts["train"][parts["train"]["has_impact"] == 1]
    reg = train_regressor(train_pos, train_pos["Affected"].astype(float), feature_cols)

    # Evaluate regression on positives in val/test
    val_pos = parts["val"][parts["val"]["has_impact"] == 1]
    test_pos = parts["test"][parts["test"]["has_impact"] == 1]
    reg_val = evaluate_regressor(reg, val_pos, val_pos["Affected"].astype(float), feature_cols)
    reg_test = evaluate_regressor(reg, test_pos, test_pos["Affected"].astype(float), feature_cols)
    with (out_dir / "reg_metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(reg_val, f, indent=2)
    with (out_dir / "reg_metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(reg_test, f, indent=2)

    # Persist models
    import joblib

    joblib.dump(clf, out_dir / "stage1_classifier.joblib")
    joblib.dump(reg, out_dir / "stage2_regressor.joblib")

    print(json.dumps({
        "splits": summary.to_dict(orient="records"),
        "clf_val": metrics_val,
        "clf_test": metrics_test,
        "reg_val": reg_val,
        "reg_test": reg_test,
        "artifacts": str(out_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()


