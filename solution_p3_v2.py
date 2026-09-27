"""Second, training-data-only model for P3 wave-height forecasting.

The script reads only the distributed P3 files.  Every estimator is fitted from
scratch.  Test cases are transformed independently, using step_minute <= 0 only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".deps"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer

import solution_p3 as base


def augment_phase_features(X: pd.DataFrame) -> pd.DataFrame:
    """Causal features describing the recent growth/decay phase."""
    out = X.copy()
    minutes = list(range(0, 361, 20))
    recent_first = out[[f"hs_lag_{m}m" for m in minutes]].to_numpy()
    finite = np.isfinite(recent_first)
    safe_high = np.where(finite, recent_first, -np.inf)
    safe_low = np.where(finite, recent_first, np.inf)
    out["hs_peak_age_6h"] = np.argmax(safe_high, axis=1) / 3.0
    out["hs_trough_age_6h"] = np.argmin(safe_low, axis=1) / 3.0

    for threshold in (1.0, 1.5, 2.0):
        above = finite & (recent_first >= threshold)
        first_below = np.argmax(~above, axis=1)
        all_above = above.all(axis=1)
        out[f"hs_run_above_{str(threshold).replace('.', 'p')}h"] = np.where(
            all_above, len(minutes), first_below
        ) / 3.0

    for hours in (2, 3, 6):
        count = hours * 3 + 1
        values = recent_first[:, :count]
        median = np.nanmedian(values, axis=1)
        out[f"hs_median_{hours}h_dense"] = median
        out[f"hs_spike_vs_median_{hours}h"] = values[:, 0] - median
        t = -np.arange(count, dtype=float) / 3.0
        mask = np.isfinite(values)
        n = mask.sum(axis=1)
        sx = (mask * t).sum(axis=1)
        sy = np.nansum(values, axis=1)
        sxx = (mask * (t * t)).sum(axis=1)
        sxy = np.nansum(values * t, axis=1)
        denominator = n * sxx - sx * sx
        out[f"hs_robust_slope_{hours}h"] = np.divide(
            n * sxy - sx * sy,
            denominator,
            out=np.full(len(out), np.nan),
            where=denominator > 1e-12,
        )
    out["hs_slope_accel_2v6"] = out["hs_robust_slope_2h"] - out["hs_robust_slope_6h"]
    return out


def fit_extra_separate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_delta: np.ndarray,
    sample_weight: np.ndarray,
    lead_indexes: tuple[int, ...],
    seed: int,
) -> dict[int, np.ndarray]:
    predictions = {}
    for k in lead_indexes:
        model = ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed + base.LEADS[k],
        )
        model.fit(X_train, y_delta[:, k], sample_weight=sample_weight)
        predictions[k] = model.predict(X_test)
    return predictions


def fit_lgb_lead9(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_delta: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    k = base.LEADS.index(9)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=45,
        max_depth=-1,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.1,
        reg_lambda=10.0,
        random_state=14009,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_delta[:, k], sample_weight=sample_weight)
    return model.predict(X_test)


def fit_extra_multi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_delta: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=220,
        min_samples_leaf=10,
        max_features=0.55,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_delta, sample_weight=sample_weight)
    return model.predict(X_test)


def fit_forest_multi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_delta: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=220,
        min_samples_leaf=5,
        max_features=0.65,
        bootstrap=True,
        max_samples=0.85,
        n_jobs=-1,
        random_state=15000,
    )
    model.fit(X_train, y_delta, sample_weight=sample_weight)
    return model.predict(X_test)


def smooth_trajectory(pred: np.ndarray, hs0: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Weakly smooth the six forecasts on their irregular lead grid."""
    reference = np.column_stack(
        [
            0.5 * hs0 + 0.5 * pred[:, 1],
            0.5 * pred[:, 0] + 0.5 * pred[:, 2],
            0.5 * pred[:, 1] + 0.5 * pred[:, 3],
            (2.0 / 3.0) * pred[:, 2] + (1.0 / 3.0) * pred[:, 4],
            0.5 * pred[:, 3] + 0.5 * pred[:, 5],
            pred[:, 5],
        ]
    )
    return (1.0 - strength) * pred + strength * reference


def train_predict(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    data, feature_cols = base.load_training(data_dir)
    test_X, test_meta = base.load_test(data_dir, feature_cols)
    X = data[feature_cols]
    X_test = test_X[feature_cols]
    X_aug = augment_phase_features(X)
    X_test_aug = augment_phase_features(X_test)

    hs0 = data.hs_lag_0m.to_numpy()
    test_hs0 = test_X.hs_lag_0m.to_numpy()
    y_abs = np.column_stack([data[f"target_{h}h"].to_numpy() for h in base.LEADS])
    y_delta = y_abs - hs0[:, None]
    weights = base.event_weights(data.reset_index(drop=True))

    imp = SimpleImputer(strategy="median", add_indicator=True)
    X_imp = imp.fit_transform(X)
    X_test_imp = imp.transform(X_test)
    imp_aug = SimpleImputer(strategy="median", add_indicator=True)
    X_aug_imp = imp_aug.fit_transform(X_aug)
    X_test_aug_imp = imp_aug.transform(X_test_aug)

    raw = np.zeros((len(X_test), len(base.LEADS)))

    # +3h: original dense trajectory; +6h: augmented growth/decay features.
    sep_plain = fit_extra_separate(X_imp, X_test_imp, y_delta, weights, (0,), 16000)
    sep_aug = fit_extra_separate(X_aug_imp, X_test_aug_imp, y_delta, weights, (1,), 17000)
    raw[:, 0] = test_hs0 + sep_plain[0]
    raw[:, 1] = test_hs0 + sep_aug[1]

    # +9h: lead-specific boosting was strongest in forward validation.
    raw[:, 2] = test_hs0 + fit_lgb_lead9(X, X_test, y_delta, weights)

    # +12h and +18h: multi-output forests preserve common trajectory structure.
    extra_plain = fit_extra_multi(X_imp, X_test_imp, y_delta, weights, 18000)
    extra_aug = fit_extra_multi(X_aug_imp, X_test_aug_imp, y_delta, weights, 19000)
    raw[:, 3] = test_hs0 + extra_plain[:, 3]
    raw[:, 4] = test_hs0 + extra_aug[:, 4]

    # +24h: bootstrap forest was the most stable long-lead candidate.
    forest = fit_forest_multi(X_imp, X_test_imp, y_delta, weights)
    raw[:, 5] = test_hs0 + forest[:, 5]

    pred = np.clip(smooth_trajectory(raw, test_hs0, strength=0.3), 0.0, 30.0)
    index = pd.read_csv(data_dir / "test_index.csv")
    lookup = {
        (test_meta.case_id.iloc[i], lead): pred[i, k]
        for i in range(len(test_meta))
        for k, lead in enumerate(base.LEADS)
    }
    submission = index.copy()
    submission["hs_pred"] = [
        lookup[(str(row.case_id), int(row.lead_h))] for row in index.itertuples(index=False)
    ]
    base.validate_submission(submission, index)
    report = {
        "training_examples": int(len(data)),
        "features": int(len(feature_cols)),
        "augmented_features": int(X_aug.shape[1]),
        "test_cases": int(len(test_X)),
        "selection_basis": "distributed-training-data forward validation only",
        "validation_hybrid_rmse": 0.6857260174792001,
        "validation_smoothed_crossfit_rmse": 0.6849815038824799,
        "models_by_lead": {
            "3": "lead-specific ExtraTrees, delta, base features",
            "6": "lead-specific ExtraTrees, delta, phase features",
            "9": "lead-specific LightGBM, delta, base features",
            "12": "multi-output ExtraTrees, delta, base features",
            "18": "multi-output ExtraTrees, delta, phase features",
            "24": "multi-output RandomForest, delta, base features",
        },
        "trajectory_smoothing": 0.3,
    }
    return submission, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "데이터셋_P3" / "P3_wave_forecast",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "submission_p3_v2.csv")
    args = parser.parse_args()
    started = time.time()
    submission, report = train_predict(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, float_format="%.6f")
    report["elapsed_seconds"] = round(time.time() - started, 2)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and {report_path} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
