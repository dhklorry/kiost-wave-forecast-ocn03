"""Fifth rule-compliant P3 wave forecast trained only on distributed data.

Model choices are based on forward and seasonal blocked validation with
78-hour independent-case weighting.  No test timestamps, external data,
pretrained weights, or leaderboard-derived parameters are used.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

import solution_p3 as base
from solution_p3_v2 import augment_phase_features, smooth_trajectory


# Coarse weights supported by both the corrected 78-hour forward validation
# and the three seasonal held-out blocks.  The long leads deliberately avoid
# the less stable boosting component.
EXTRA_WEIGHT = {3: 0.75, 6: 0.75, 9: 0.75, 12: 1.00, 18: 1.00, 24: 1.00}
SMOOTHING = 0.20


def station_uniform_weights(
    frame: pd.DataFrame, station_share: dict[str, float]
) -> np.ndarray:
    """Match the station composition disclosed by distributed test_index.csv."""
    weights = np.zeros(len(frame), dtype=float)
    for station, idx in frame.groupby("station").groups.items():
        loc = np.asarray(list(idx), dtype=int)
        weights[loc] = station_share[station] / len(loc)
    return weights / weights.mean()


def make_extra(
    X: np.ndarray,
    X_test: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    lead: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=220,
        min_samples_leaf=10,
        max_features=0.55,
        n_jobs=-1,
        random_state=23000 + lead,
    )
    model.fit(X, target, sample_weight=weights)
    return model.predict(X_test)


def make_lgb(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    lead: int,
) -> np.ndarray:
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
        random_state=24000 + lead,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, target, sample_weight=weights)
    return model.predict(X_test)


def train_predict(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    index = pd.read_csv(data_dir / "test_index.csv")
    case_station = index[["case_id", "station"]].drop_duplicates("case_id")
    station_share = case_station.station.value_counts(normalize=True).to_dict()
    data, feature_cols = base.load_training(data_dir)
    test_X, test_meta = base.load_test(data_dir, feature_cols)
    X_base = data[feature_cols]
    test_base = test_X[feature_cols]
    X_aug = augment_phase_features(X_base)
    test_aug = augment_phase_features(test_base)

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_aug_imp = imputer.fit_transform(X_aug)
    test_aug_imp = imputer.transform(test_aug)

    current = data.hs_lag_0m.to_numpy()
    test_current = test_X.hs_lag_0m.to_numpy()
    weights = station_uniform_weights(data.reset_index(drop=True), station_share)
    raw = np.zeros((len(test_X), len(base.LEADS)), dtype=float)

    for k, lead in enumerate(base.LEADS):
        future = data[f"target_{lead}h"].to_numpy()
        target = np.log(np.clip(future, 0.05, None) / np.clip(current, 0.05, None))
        extra_log = make_extra(X_aug_imp, test_aug_imp, target, weights, lead)
        p_extra = test_current * np.exp(np.clip(extra_log, -3.0, 3.0))
        extra_weight = EXTRA_WEIGHT[lead]
        if extra_weight < 1.0:
            lgb_log = make_lgb(X_base, test_base, target, weights, lead)
            p_lgb = test_current * np.exp(np.clip(lgb_log, -3.0, 3.0))
            raw[:, k] = extra_weight * p_extra + (1.0 - extra_weight) * p_lgb
        else:
            raw[:, k] = p_extra

    pred = np.clip(smooth_trajectory(raw, test_current, strength=SMOOTHING), 0.0, 30.0)
    lookup = {
        (test_meta.case_id.iloc[i], lead): pred[i, k]
        for i in range(len(test_meta))
        for k, lead in enumerate(base.LEADS)
    }
    submission = index.copy()
    submission["hs_pred"] = [
        lookup[(str(row.case_id), int(row.lead_h))]
        for row in index.itertuples(index=False)
    ]
    base.validate_submission(submission, index)
    report = {
        "training_examples": int(len(data)),
        "base_features": int(len(feature_cols)),
        "augmented_features": int(X_aug.shape[1]),
        "test_cases": int(len(test_X)),
        "selection_basis": "distributed-data forward and seasonal blocked validation only",
        "target": "log(future_hs / current_hs)",
        "training_station_weight": {
            station: float(station_share[station]) for station in sorted(station_share)
        },
        "extra_weight_by_lead": {str(k): v for k, v in EXTRA_WEIGHT.items()},
        "trajectory_smoothing": SMOOTHING,
        "external_data": False,
        "pretrained_weights": False,
    }
    return submission, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "데이터셋_P3" / "P3_wave_forecast",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "submission_p3_v5.csv")
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
