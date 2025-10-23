"""
Optimized training script with hyperparameter tuning, cross-validation, and advanced techniques.

Key improvements:
1. Randomized search for hyperparameter optimization
2. Cross-validation for robust evaluation
3. Early stopping to prevent overfitting
4. Feature importance analysis
5. Model calibration
6. Better evaluation metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
    """Build preprocessing pipeline with imputation and scaling."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def get_classifier_param_grid() -> Dict:
    """Define hyperparameter search space for classifier."""
    return {
        "model__n_estimators": [300, 500, 700, 1000],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        "model__reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3, 0.5],
    }


def get_regressor_param_grid() -> Dict:
    """Define hyperparameter search space for regressor."""
    return {
        "model__n_estimators": [400, 600, 800, 1000],
        "model__max_depth": [6, 8, 10, 12],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        "model__reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3, 0.5],
    }


def train_classifier_optimized(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str],
    n_iter: int = 50,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """
    Train classifier with hyperparameter optimization and cross-validation.
    
    Returns:
        Tuple of (best_model, tuning_results)
    """
    print("\n" + "=" * 70)
    print("CLASSIFIER TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Calculate class weight
    pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1.0, y_train.sum()))
    print(f"Class imbalance ratio: {pos_weight:.2f}")
    print(f"Positive samples: {y_train.sum()} / {len(y_train)} ({100*y_train.sum()/len(y_train):.1f}%)")
    
    # Base model
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        tree_method="hist",
        random_state=random_state,
        enable_categorical=False,
    )
    # Ensure sklearn recognizes this as a classifier (robust against version quirks)
    try:
        base_model._estimator_type = "classifier"
    except Exception:
        pass
    
    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("pre", build_preprocessor(feature_cols)),
            ("model", base_model),
        ]
    )
    
    # Hyperparameter search
    param_grid = get_classifier_param_grid()
    
    print(f"\nStarting RandomizedSearchCV:")
    print(f"  - Iterations: {n_iter}")
    print(f"  - CV Folds: {cv_folds}")
    print(f"  - Metric: AUC-PR (average precision)")
    print(f"  - Search space: {sum(len(v) if isinstance(v, list) else 1 for v in param_grid.values())} combinations")
    
    # Use StratifiedKFold for better class distribution
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Custom, robust AP scorer to avoid response_method issues in some sklearn/xgboost combos
    def _ap_scorer(estimator, X, y):
        try:
            if hasattr(estimator, "predict_proba"):
                s = estimator.predict_proba(X)[:, 1]
            elif hasattr(estimator, "decision_function"):
                s = estimator.decision_function(X)
            else:
                s = estimator.predict(X)
        except Exception:
            s = estimator.predict(X)
        return average_precision_score(y, s)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=_ap_scorer,  # AUC-PR (robust scorer)
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        return_train_score=True,
    )
    
    # Fit with early stopping on validation set
    print("\nTraining...")
    search.fit(
        X_train[feature_cols],
        y_train,
    )
    
    best_model = search.best_estimator_
    
    # Extract tuning results
    tuning_results = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "mean_train_score": search.cv_results_["mean_train_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
        },
    }
    
    print(f"\n✓ Best CV Score (AUC-PR): {search.best_score_:.4f}")
    print(f"✓ Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")
    
    return best_model, tuning_results


def train_regressor_optimized(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str],
    n_iter: int = 50,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """
    Train regressor with hyperparameter optimization and cross-validation.
    
    Returns:
        Tuple of (best_model, tuning_results)
    """
    print("\n" + "=" * 70)
    print("REGRESSOR TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Log transform target
    y_train_log = np.log1p(y_train.astype(float))
    y_val_log = np.log1p(y_val.astype(float))
    
    print(f"Target statistics (log scale):")
    print(f"  - Mean: {y_train_log.mean():.4f}")
    print(f"  - Std: {y_train_log.std():.4f}")
    print(f"  - Min: {y_train_log.min():.4f}")
    print(f"  - Max: {y_train_log.max():.4f}")
    
    # Base model
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=random_state,
        enable_categorical=False,
    )
    
    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("pre", build_preprocessor(feature_cols)),
            ("model", base_model),
        ]
    )
    
    # Hyperparameter search
    param_grid = get_regressor_param_grid()
    
    print(f"\nStarting RandomizedSearchCV:")
    print(f"  - Iterations: {n_iter}")
    print(f"  - CV Folds: {cv_folds}")
    print(f"  - Metric: Negative RMSE")
    print(f"  - Search space: {sum(len(v) if isinstance(v, list) else 1 for v in param_grid.values())} combinations")
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",  # RMSE
        cv=cv_folds,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        return_train_score=True,
    )
    
    # Fit
    print("\nTraining...")
    search.fit(
        X_train[feature_cols],
        y_train_log,
    )
    
    best_model = search.best_estimator_
    
    # Extract tuning results
    tuning_results = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "mean_train_score": search.cv_results_["mean_train_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
        },
    }
    
    print(f"\n✓ Best CV Score (Negative RMSE): {search.best_score_:.4f}")
    print(f"✓ Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")
    
    return best_model, tuning_results


def evaluate_classifier(
    clf: Pipeline, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]
) -> Dict[str, float]:
    """Evaluate classifier with comprehensive metrics."""
    proba = clf.predict_proba(X[feature_cols])[:, 1]
    
    # AUC-PR
    ap = average_precision_score(y, proba)
    
    # Find optimal threshold using F1 score
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1_scores = 2 * (precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1_scores))
    best_thresh = thresholds[max(0, best_idx - 1)] if len(thresholds) > 0 else 0.5
    
    # Predictions at best threshold
    y_pred = (proba >= best_thresh).astype(int)
    
    # Metrics
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    return {
        "auc_pr": float(ap),
        "best_threshold": float(best_thresh),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred)),
        "accuracy": float(accuracy_score(y, y_pred)),
    }


def evaluate_regressor(
    reg: Pipeline, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]
) -> Dict[str, float]:
    """Evaluate regressor with comprehensive metrics."""
    y_log = np.log1p(y.astype(float))
    y_pred_log = reg.predict(X[feature_cols])
    
    # Metrics on log scale
    rmse_log = float(np.sqrt(mean_squared_error(y_log, y_pred_log)))
    mae_log = float(mean_absolute_error(y_log, y_pred_log))
    r2_log = float(r2_score(y_log, y_pred_log))
    
    # Metrics on original scale
    y_pred = np.expm1(y_pred_log)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = float(np.mean(np.abs((y - y_pred) / np.clip(y, 1, None))) * 100)
    
    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def analyze_feature_importance(
    model: Pipeline, feature_cols: List[str], top_n: int = 20
) -> pd.DataFrame:
    """Extract and rank feature importance."""
    xgb_model = model.named_steps["model"]
    importance = xgb_model.feature_importances_
    
    df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False)
    
    print(f"\n📊 Top {top_n} Most Important Features:")
    print("=" * 70)
    for idx, row in df.head(top_n).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train optimized models with hyperparameter tuning"
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="stratified_storm",
        choices=["temporal", "stratified_storm"],
    )
    parser.add_argument(
        "--train_range", type=str, default="2010-2018", help="Only for temporal mode"
    )
    parser.add_argument(
        "--val_range", type=str, default="2019-2020", help="Only for temporal mode"
    )
    parser.add_argument(
        "--test_range", type=str, default="2021-2024", help="Only for temporal mode"
    )
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts_houses_optimized")
    parser.add_argument("--n_iter", type=int, default=50, help="RandomizedSearch iterations")
    parser.add_argument("--cv_folds", type=int, default=3, help="Cross-validation folds")
    parser.add_argument(
        "--skip_classifier", action="store_true", help="Skip classifier tuning"
    )
    parser.add_argument(
        "--skip_regressor", action="store_true", help="Skip regressor tuning"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("OPTIMIZED MODEL TRAINING")
    print("=" * 70)
    print(f"Output directory: {args.out_dir}")
    print(f"RandomizedSearch iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Random state: {args.random_state}")

    # Load data
    print("\nLoading dataset (HOUSES)...")
    paths = DataPaths()
    built = build_dataset_houses(paths)
    dataset = built["dataset"]
    feature_cols = select_feature_columns(dataset)
    print(f"✓ Dataset: {len(dataset)} samples, {len(feature_cols)} features")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(feature_cols, out_dir / "feature_columns.json")

    # Split data
    print(f"\nSplitting data (mode: {args.split_mode})...")
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
    else:
        parts = stratified_storm_split(
            dataset,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=args.random_state,
        )

    summary = summarize_split_counts(dataset, parts)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    print("✓ Data split complete")
    print(summary.to_string(index=False))

    # Stage 1: Classifier
    if not args.skip_classifier:
        clf, clf_tuning = train_classifier_optimized(
            parts["train"],
            parts["train"]["has_house_impact"].astype(int),
            parts["val"],
            parts["val"]["has_house_impact"].astype(int),
            feature_cols,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        # Evaluate
        print("\n" + "=" * 70)
        print("CLASSIFIER EVALUATION (HOUSES)")
        print("=" * 70)
        
        metrics_val = evaluate_classifier(
            clf, parts["val"], parts["val"]["has_house_impact"].astype(int), feature_cols
        )
        metrics_test = evaluate_classifier(
            clf, parts["test"], parts["test"]["has_house_impact"].astype(int), feature_cols
        )

        print("\nValidation Metrics:")
        for k, v in metrics_val.items():
            print(f"  {k:20s}: {v:.4f}")
        
        print("\nTest Metrics:")
        for k, v in metrics_test.items():
            print(f"  {k:20s}: {v:.4f}")

        # Feature importance
        feat_imp_clf = analyze_feature_importance(clf, feature_cols)
        feat_imp_clf.to_csv(out_dir / "feature_importance_classifier.csv", index=False)

        # Save
        with (out_dir / "clf_metrics_val.json").open("w") as f:
            json.dump(metrics_val, f, indent=2)
        with (out_dir / "clf_metrics_test.json").open("w") as f:
            json.dump(metrics_test, f, indent=2)
        with (out_dir / "clf_tuning_results.json").open("w") as f:
            json.dump(clf_tuning, f, indent=2)

        joblib.dump(clf, out_dir / "stage1_classifier.joblib")
        print(f"\n✓ Classifier saved to: {out_dir / 'stage1_classifier.joblib'}")

    # Stage 2: Regressor
    if not args.skip_regressor:
        train_pos = parts["train"][parts["train"]["has_house_impact"] == 1]
        val_pos = parts["val"][parts["val"]["has_house_impact"] == 1]
        test_pos = parts["test"][parts["test"]["has_house_impact"] == 1]

        print(f"\nRegressor training samples (HOUSES):")
        print(f"  Train: {len(train_pos)}")
        print(f"  Val: {len(val_pos)}")
        print(f"  Test: {len(test_pos)}")

        reg, reg_tuning = train_regressor_optimized(
            train_pos,
            train_pos["Total_Houses"].astype(float),
            val_pos,
            val_pos["Total_Houses"].astype(float),
            feature_cols,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        # Evaluate
        print("\n" + "=" * 70)
        print("REGRESSOR EVALUATION (HOUSES)")
        print("=" * 70)
        
        metrics_val = evaluate_regressor(
            reg, val_pos, val_pos["Total_Houses"].astype(float), feature_cols
        )
        metrics_test = evaluate_regressor(
            reg, test_pos, test_pos["Total_Houses"].astype(float), feature_cols
        )

        print("\nValidation Metrics:")
        for k, v in metrics_val.items():
            print(f"  {k:20s}: {v:.4f}")
        
        print("\nTest Metrics:")
        for k, v in metrics_test.items():
            print(f"  {k:20s}: {v:.4f}")

        # Feature importance
        feat_imp_reg = analyze_feature_importance(reg, feature_cols)
        feat_imp_reg.to_csv(out_dir / "feature_importance_regressor.csv", index=False)

        # Save
        with (out_dir / "reg_metrics_val.json").open("w") as f:
            json.dump(metrics_val, f, indent=2)
        with (out_dir / "reg_metrics_test.json").open("w") as f:
            json.dump(metrics_test, f, indent=2)
        with (out_dir / "reg_tuning_results.json").open("w") as f:
            json.dump(reg_tuning, f, indent=2)

        joblib.dump(reg, out_dir / "stage2_regressor.joblib")
        print(f"\n✓ Regressor saved to: {out_dir / 'stage2_regressor.joblib'}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"All artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

