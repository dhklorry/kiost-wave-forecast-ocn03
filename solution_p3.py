"""Train and generate a rule-compliant submission for Problem 3.

Only the files in ``데이터셋_P3/P3_wave_forecast`` are read.  The model is
trained from scratch and never uses absolute test timestamps, external data,
pretrained weights, or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


LEADS = (3, 6, 9, 12, 18, 24)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
TARGET_STATION_SHARE = {"G-ORS": 0.35, "I-ORS": 0.35, "S-ORS": 0.30}


def rolling_stat(s: pd.Series, rows: int, stat: str) -> pd.Series:
    min_periods = max(2, int(rows * 0.35))
    r = s.rolling(rows, min_periods=min_periods)
    return getattr(r, stat)()


def make_grid_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Create causal features on a complete 10-minute relative/absolute grid."""
    g = grid.sort_index().copy()
    d: dict[str, pd.Series] = {}

    # Exact wave lags.  Wave observations occur every 20 minutes, so every lag
    # below is an integer multiple of two 10-minute rows.
    wave_lag_minutes = tuple(range(0, 361, 20)) + (540, 720, 1080, 1440, 2160, 2880)
    for col in ("hs", "tp", "hmax"):
        s = g[col]
        for minutes in wave_lag_minutes:
            d[f"{col}_lag_{minutes}m"] = s.shift(minutes // 10)

    # Wave summaries retain information about development, peak, variability,
    # and decay over several physically relevant horizons.
    for hours in (1, 3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        for stat in ("mean", "std", "min", "max"):
            d[f"hs_{stat}_{hours}h"] = rolling_stat(g["hs"], rows, stat)
        d[f"hs_count_{hours}h"] = g["hs"].rolling(rows).count()
        d[f"hs_vs_mean_{hours}h"] = g["hs"] - d[f"hs_mean_{hours}h"]
        d[f"hs_drop_from_max_{hours}h"] = g["hs"] - d[f"hs_max_{hours}h"]
        d[f"hs_range_{hours}h"] = d[f"hs_max_{hours}h"] - d[f"hs_min_{hours}h"]

    for col in ("tp", "hmax"):
        for hours in (3, 12, 24, 48):
            rows = hours * 6 + 1
            d[f"{col}_mean_{hours}h"] = rolling_stat(g[col], rows, "mean")
            d[f"{col}_std_{hours}h"] = rolling_stat(g[col], rows, "std")

    wave_rad = np.deg2rad(g["wvdir"])
    wave_sin = np.sin(wave_rad)
    wave_cos = np.cos(wave_rad)
    d["wvdir_sin_0"] = wave_sin
    d["wvdir_cos_0"] = wave_cos
    for hours in (3, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"wvdir_sin_mean_{hours}h"] = rolling_stat(wave_sin, rows, "mean")
        d[f"wvdir_cos_mean_{hours}h"] = rolling_stat(wave_cos, rows, "mean")

    # Meteorological inputs use the native 10-minute frequency.
    met_cols = ("wspd", "gust", "caph", "airt", "relh")
    for col in met_cols:
        s = g[col]
        for hours in (0, 1, 3, 6, 12, 24, 48):
            d[f"{col}_lag_{hours}h"] = s.shift(hours * 6)
        for hours in (1, 3, 6, 12, 24, 48):
            rows = hours * 6 + 1
            d[f"{col}_mean_{hours}h"] = rolling_stat(s, rows, "mean")
            d[f"{col}_std_{hours}h"] = rolling_stat(s, rows, "std")
            if col in ("wspd", "gust"):
                d[f"{col}_max_{hours}h"] = rolling_stat(s, rows, "max")
            if col == "caph":
                d[f"{col}_min_{hours}h"] = rolling_stat(s, rows, "min")

    wind_rad = np.deg2rad(g["wdir"])
    # Direction is circular. Components preserve wrap-around correctly.
    wind_sin = g["wspd"] * np.sin(wind_rad)
    wind_cos = g["wspd"] * np.cos(wind_rad)
    d["wind_sin_0"] = wind_sin
    d["wind_cos_0"] = wind_cos
    d["wdir_sin_0"] = np.sin(wind_rad)
    d["wdir_cos_0"] = np.cos(wind_rad)
    for hours in (3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"wind_sin_mean_{hours}h"] = rolling_stat(wind_sin, rows, "mean")
        d[f"wind_cos_mean_{hours}h"] = rolling_stat(wind_cos, rows, "mean")
    relative_wind = g["wspd"] * np.cos(wind_rad - wave_rad)
    cross_wind = g["wspd"] * np.sin(wind_rad - wave_rad)
    wind_energy = g["wspd"] ** 2
    d["relative_wind_0"] = relative_wind
    d["cross_wind_0"] = cross_wind
    for hours in (3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"relative_wind_mean_{hours}h"] = rolling_stat(relative_wind, rows, "mean")
        d[f"cross_wind_mean_{hours}h"] = rolling_stat(cross_wind, rows, "mean")
        d[f"wind_energy_mean_{hours}h"] = rolling_stat(wind_energy, rows, "mean")

    # Explicit causal changes make the smooth, low-frequency evolution easier
    # for both linear and tree learners to recognize.
    hs0 = d["hs_lag_0m"]
    for hours in (1, 3, 6, 9, 12, 18, 24, 36, 48):
        lag = d[f"hs_lag_{hours * 60}m"]
        d[f"hs_change_{hours}h"] = hs0 - lag
        d[f"hs_slope_{hours}h"] = (hs0 - lag) / hours
    for col in ("wspd", "gust", "caph"):
        now = d[f"{col}_lag_0h"]
        for hours in (3, 6, 12, 24, 48):
            d[f"{col}_change_{hours}h"] = now - d[f"{col}_lag_{hours}h"]

    d["hmax_hs_ratio"] = d["hmax_lag_0m"] / hs0.clip(lower=0.1)
    d["wind_wave_ratio"] = d["wspd_lag_0h"] / hs0.clip(lower=0.1)
    d["hs_sq"] = hs0 * hs0
    d["wspd_sq"] = d["wspd_lag_0h"] ** 2
    f = pd.DataFrame(d, index=g.index)
    return f.replace([np.inf, -np.inf], np.nan)


def make_endpoint_features(grid: pd.DataFrame) -> pd.Series:
    """Fast scalar equivalent of ``make_grid_features(...).loc[last]``."""
    g = grid.sort_index()
    d: dict[str, float] = {}

    def lag(s: pd.Series, rows: int) -> float:
        return float(s.iloc[-1 - rows]) if len(s) > rows else math.nan

    def stat(s: pd.Series, rows: int, name: str) -> float:
        tail = s.iloc[-rows:]
        if tail.count() < max(2, int(rows * 0.35)):
            return math.nan
        return float(getattr(tail, name)())

    wave_lag_minutes = tuple(range(0, 361, 20)) + (540, 720, 1080, 1440, 2160, 2880)
    for col in ("hs", "tp", "hmax"):
        for minutes in wave_lag_minutes:
            d[f"{col}_lag_{minutes}m"] = lag(g[col], minutes // 10)
    for hours in (1, 3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        for name in ("mean", "std", "min", "max"):
            d[f"hs_{name}_{hours}h"] = stat(g["hs"], rows, name)
        d[f"hs_count_{hours}h"] = float(g["hs"].iloc[-rows:].count())
        d[f"hs_vs_mean_{hours}h"] = d["hs_lag_0m"] - d[f"hs_mean_{hours}h"]
        d[f"hs_drop_from_max_{hours}h"] = d["hs_lag_0m"] - d[f"hs_max_{hours}h"]
        d[f"hs_range_{hours}h"] = d[f"hs_max_{hours}h"] - d[f"hs_min_{hours}h"]
    for col in ("tp", "hmax"):
        for hours in (3, 12, 24, 48):
            rows = hours * 6 + 1
            d[f"{col}_mean_{hours}h"] = stat(g[col], rows, "mean")
            d[f"{col}_std_{hours}h"] = stat(g[col], rows, "std")

    wave_rad = np.deg2rad(g["wvdir"])
    wave_sin, wave_cos = np.sin(wave_rad), np.cos(wave_rad)
    d["wvdir_sin_0"] = lag(wave_sin, 0)
    d["wvdir_cos_0"] = lag(wave_cos, 0)
    for hours in (3, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"wvdir_sin_mean_{hours}h"] = stat(wave_sin, rows, "mean")
        d[f"wvdir_cos_mean_{hours}h"] = stat(wave_cos, rows, "mean")

    for col in ("wspd", "gust", "caph", "airt", "relh"):
        for hours in (0, 1, 3, 6, 12, 24, 48):
            d[f"{col}_lag_{hours}h"] = lag(g[col], hours * 6)
        for hours in (1, 3, 6, 12, 24, 48):
            rows = hours * 6 + 1
            d[f"{col}_mean_{hours}h"] = stat(g[col], rows, "mean")
            d[f"{col}_std_{hours}h"] = stat(g[col], rows, "std")
            if col in ("wspd", "gust"):
                d[f"{col}_max_{hours}h"] = stat(g[col], rows, "max")
            if col == "caph":
                d[f"{col}_min_{hours}h"] = stat(g[col], rows, "min")

    wind_rad = np.deg2rad(g["wdir"])
    wind_sin = g["wspd"] * np.sin(wind_rad)
    wind_cos = g["wspd"] * np.cos(wind_rad)
    d["wind_sin_0"] = lag(wind_sin, 0)
    d["wind_cos_0"] = lag(wind_cos, 0)
    d["wdir_sin_0"] = lag(np.sin(wind_rad), 0)
    d["wdir_cos_0"] = lag(np.cos(wind_rad), 0)
    for hours in (3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"wind_sin_mean_{hours}h"] = stat(wind_sin, rows, "mean")
        d[f"wind_cos_mean_{hours}h"] = stat(wind_cos, rows, "mean")
    relative_wind = g["wspd"] * np.cos(wind_rad - wave_rad)
    cross_wind = g["wspd"] * np.sin(wind_rad - wave_rad)
    wind_energy = g["wspd"] ** 2
    d["relative_wind_0"] = lag(relative_wind, 0)
    d["cross_wind_0"] = lag(cross_wind, 0)
    for hours in (3, 6, 12, 24, 48):
        rows = hours * 6 + 1
        d[f"relative_wind_mean_{hours}h"] = stat(relative_wind, rows, "mean")
        d[f"cross_wind_mean_{hours}h"] = stat(cross_wind, rows, "mean")
        d[f"wind_energy_mean_{hours}h"] = stat(wind_energy, rows, "mean")

    hs0 = d["hs_lag_0m"]
    for hours in (1, 3, 6, 9, 12, 18, 24, 36, 48):
        prior = d[f"hs_lag_{hours * 60}m"]
        d[f"hs_change_{hours}h"] = hs0 - prior
        d[f"hs_slope_{hours}h"] = (hs0 - prior) / hours
    for col in ("wspd", "gust", "caph"):
        for hours in (3, 6, 12, 24, 48):
            d[f"{col}_change_{hours}h"] = d[f"{col}_lag_0h"] - d[f"{col}_lag_{hours}h"]
    d["hmax_hs_ratio"] = d["hmax_lag_0m"] / max(hs0, 0.1)
    d["wind_wave_ratio"] = d["wspd_lag_0h"] / max(hs0, 0.1)
    d["hs_sq"] = hs0 * hs0
    d["wspd_sq"] = d["wspd_lag_0h"] ** 2
    return pd.Series(d, dtype=float).replace([np.inf, -np.inf], np.nan)


def load_training(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    wave = pd.read_csv(data_dir / "train_wave.csv", parse_dates=["time"])
    atmos = pd.read_csv(data_dir / "train_atmos.csv", parse_dates=["time"])
    frames: list[pd.DataFrame] = []

    for station in STATIONS:
        w = wave.loc[wave.station == station].drop(columns="station").set_index("time").sort_index()
        a = atmos.loc[atmos.station == station].drop(columns="station").set_index("time").sort_index()
        idx = pd.date_range(w.index.min(), w.index.max(), freq="10min")
        grid = pd.DataFrame(index=idx).join(w).join(a)
        feat = make_grid_features(grid)
        feat["station_G"] = float(station == "G-ORS")
        feat["station_I"] = float(station == "I-ORS")
        feat["station_S"] = float(station == "S-ORS")
        feat["station"] = station
        feat["anchor_time"] = feat.index
        for lead in LEADS:
            feat[f"target_{lead}h"] = grid["hs"].shift(-(lead * 6))

        valid = feat["hs_lag_0m"].ge(1.5)
        valid &= feat[[f"target_{h}h" for h in LEADS]].notna().all(axis=1)
        valid &= (feat.index >= feat.index.min() + pd.Timedelta(hours=48))
        frames.append(feat.loc[valid].reset_index(drop=True))

    data = pd.concat(frames, ignore_index=True)
    non_features = {"station", "anchor_time", *(f"target_{h}h" for h in LEADS)}
    feature_cols = [c for c in data.columns if c not in non_features]
    return data, feature_cols


def load_test(data_dir: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = pd.read_parquet(data_dir / "test_context.parquet")
    rows: list[pd.Series] = []
    meta: list[dict[str, str]] = []
    for case_id, case in context.groupby("case_id", sort=False):
        station = str(case["station"].iloc[0])
        grid = case.sort_values("step_minute").set_index("step_minute")
        last = make_endpoint_features(grid)
        last["station_G"] = float(station == "G-ORS")
        last["station_I"] = float(station == "I-ORS")
        last["station_S"] = float(station == "S-ORS")
        rows.append(last.reindex(feature_cols))
        meta.append({"case_id": str(case_id), "station": station})
    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(meta)


def event_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance stations and reduce domination by densely sampled long storms."""
    weights = np.zeros(len(frame), dtype=float)
    for station, idx in frame.groupby("station").groups.items():
        loc = np.asarray(list(idx), dtype=int)
        times = frame.loc[loc, "anchor_time"].astype("int64").to_numpy() / 3.6e12
        order = np.argsort(times)
        sorted_times = times[order]
        left = np.searchsorted(sorted_times, sorted_times - 39.0, side="left")
        right = np.searchsorted(sorted_times, sorted_times + 39.0, side="right")
        local_density = np.maximum(1, right - left)
        local_weight = 1.0 / np.sqrt(local_density)
        local_weight *= TARGET_STATION_SHARE[station] / local_weight.sum()
        weights[loc[order]] = local_weight
    return weights / weights.mean()


def evaluation_like_weights(frame: pd.DataFrame, repeats: int = 16) -> np.ndarray:
    """Monte-Carlo approximation of independently spaced evaluation cases."""
    weights = np.zeros(len(frame), dtype=float)
    for station, idx in frame.groupby("station").groups.items():
        loc = np.asarray(list(idx), dtype=int)
        times = frame.loc[loc, "anchor_time"].astype("int64").to_numpy() / 3.6e12
        order = np.argsort(times)
        sorted_times = times[order]
        for repeat in range(repeats):
            rng = np.random.default_rng(91021 + repeat * 101 + STATIONS.index(station) * 1009)
            cursor = sorted_times[0] + rng.uniform(0.0, 78.0)
            end = sorted_times[-1]
            while cursor <= end:
                left = np.searchsorted(sorted_times, cursor, side="left")
                right = np.searchsorted(sorted_times, cursor + 78.0, side="left")
                if right > left:
                    chosen = int(rng.integers(left, right))
                    weights[loc[order[chosen]]] += 1.0
                    cursor = sorted_times[chosen] + 78.0
                else:
                    cursor += 78.0
        station_sum = weights[loc].sum()
        if station_sum > 0:
            weights[loc] *= TARGET_STATION_SHARE[station] / station_sum
    if weights.sum() <= 0:
        return event_weights(frame)
    return weights / weights.mean()


def fold_definitions() -> list[tuple[str, str, str]]:
    # Expanding-window splits: model selection never trains on data later than
    # its validation block.  The final block is deliberately largest.
    return [
        ("2024-01-01", "2024-10-01", "2024-12-01"),
        ("2024-01-01", "2025-02-01", "2025-04-01"),
        ("2024-01-01", "2025-04-01", "2025-07-01"),
    ]


def lgb_params(seed: int, variant: str) -> dict:
    common = dict(
        objective="regression",
        verbosity=-1,
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.04,
        n_estimators=320,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.15,
        reg_lambda=5.0,
    )
    if variant == "smooth":
        common.update(num_leaves=15, max_depth=5, min_child_samples=90)
    else:
        common.update(num_leaves=27, max_depth=7, min_child_samples=55)
    return common


def fit_lgb(X, y, weights, seed: int, variant: str) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**lgb_params(seed, variant))
    model.fit(X, y, sample_weight=weights)
    return model


def weighted_rmse(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def cross_validate(data: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[int, np.ndarray], dict]:
    X_all = data[feature_cols]
    hs0_all = data["hs_lag_0m"].to_numpy()
    records = []

    for fold_id, (_, val_start, val_end) in enumerate(fold_definitions()):
        train_mask = data.anchor_time < pd.Timestamp(val_start, tz="Asia/Seoul")
        val_mask = (data.anchor_time >= pd.Timestamp(val_start, tz="Asia/Seoul")) & (
            data.anchor_time < pd.Timestamp(val_end, tz="Asia/Seoul")
        )
        tr = np.flatnonzero(train_mask.to_numpy())
        va = np.flatnonzero(val_mask.to_numpy())
        if len(tr) == 0 or len(va) == 0:
            continue
        w_tr = event_weights(data.iloc[tr].reset_index(drop=True))
        # Use smooth event-density weights for stable model selection.  A much
        # smaller 78-hour Monte-Carlo subset is retained as a diagnostic helper
        # above, but is intentionally not used to tune the final ensemble.
        w_va = event_weights(data.iloc[va].reset_index(drop=True))

        imputer = SimpleImputer(strategy="median", add_indicator=True)
        X_tr_extra = imputer.fit_transform(X_all.iloc[tr])
        X_va_extra = imputer.transform(X_all.iloc[va])
        y_tr_extra = np.column_stack(
            [data.iloc[tr][f"target_{lead}h"].to_numpy() - hs0_all[tr] for lead in LEADS]
        )
        extra = ExtraTreesRegressor(
            n_estimators=140,
            min_samples_leaf=10,
            max_features=0.55,
            n_jobs=-1,
            random_state=7000 + fold_id,
        )
        extra.fit(X_tr_extra, y_tr_extra, sample_weight=w_tr)
        p_extra_all = hs0_all[va, None] + extra.predict(X_va_extra)
        cat = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=520,
            depth=7,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            random_seed=9000 + fold_id,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        cat.fit(X_all.iloc[tr], y_tr_extra, sample_weight=w_tr)
        p_cat_all = hs0_all[va, None] + cat.predict(X_all.iloc[va])

        ridge_models = {}
        for lead in LEADS:
            y_tr = data.iloc[tr][f"target_{lead}h"].to_numpy()
            ridge = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=120.0),
            )
            ridge.fit(X_all.iloc[tr], y_tr, ridge__sample_weight=w_tr)
            ridge_models[lead] = ridge

        for lead in LEADS:
            y_tr_abs = data.iloc[tr][f"target_{lead}h"].to_numpy()
            y_tr_delta = y_tr_abs - hs0_all[tr]
            y_va = data.iloc[va][f"target_{lead}h"].to_numpy()

            smooth = fit_lgb(X_all.iloc[tr], y_tr_delta, w_tr, 1000 + fold_id * 17 + lead, "smooth")
            p_persist = hs0_all[va]
            p_smooth = p_persist + smooth.predict(X_all.iloc[va])
            p_ridge = ridge_models[lead].predict(X_all.iloc[va])
            for j, row_idx in enumerate(va):
                records.append(
                    {
                        "row_idx": int(row_idx),
                        "fold": fold_id,
                        "lead": lead,
                        "y": y_va[j],
                        "weight": w_va[j],
                        "persist": p_persist[j],
                        "smooth": p_smooth[j],
                        "ridge": p_ridge[j],
                        "extra": p_extra_all[j, LEADS.index(lead)],
                        "cat": p_cat_all[j, LEADS.index(lead)],
                    }
                )

    oof = pd.DataFrame(records)
    model_cols = ["persist", "smooth", "ridge", "extra", "cat"]
    y = oof.y.to_numpy()
    w = oof.weight.to_numpy()
    metrics = {c: weighted_rmse(y, oof[c].to_numpy(), w) for c in model_cols}

    # Non-negative least squares learns a transparent, training-only ensemble.
    sw = np.sqrt(w)
    global_coef, _ = nnls(oof[model_cols].to_numpy() * sw[:, None], y * sw)
    if global_coef.sum() <= 0:
        global_coef = np.ones(len(model_cols))
    global_coef /= global_coef.sum()
    lead_coef: dict[int, np.ndarray] = {}
    blend = np.zeros(len(oof), dtype=float)
    for lead in LEADS:
        mask = oof.lead.eq(lead).to_numpy()
        lead_sw = np.sqrt(w[mask])
        coef, _ = nnls(oof.loc[mask, model_cols].to_numpy() * lead_sw[:, None], y[mask] * lead_sw)
        if coef.sum() <= 0:
            coef = global_coef.copy()
        else:
            coef /= coef.sum()
        # Mild shrinkage guards against a small number of independent storms.
        coef = 0.5 * coef + 0.5 * global_coef
        coef /= coef.sum()
        candidate = oof.loc[mask, model_cols].to_numpy() @ coef
        candidate_rmse = weighted_rmse(y[mask], candidate, w[mask])
        single_errors = {
            name: weighted_rmse(y[mask], oof.loc[mask, name].to_numpy(), w[mask])
            for name in ("extra", "cat")
        }
        best_single = min(single_errors, key=single_errors.get)
        if single_errors[best_single] + 0.005 < candidate_rmse:
            coef = np.zeros(len(model_cols))
            coef[model_cols.index(best_single)] = 1.0
        lead_coef[lead] = coef
        blend[mask] = oof.loc[mask, model_cols].to_numpy() @ coef
    oof["blend"] = blend
    metrics["blend"] = weighted_rmse(y, oof.blend.to_numpy(), w)
    metrics["global_weights"] = dict(zip(model_cols, global_coef.tolist()))
    metrics["weights_by_lead"] = {
        str(lead): dict(zip(model_cols, lead_coef[lead].tolist())) for lead in LEADS
    }
    metrics["n_oof_rows"] = int(len(oof))
    metrics["by_lead"] = {
        str(lead): {
            c: weighted_rmse(
                oof.loc[oof.lead == lead, "y"].to_numpy(),
                oof.loc[oof.lead == lead, c].to_numpy(),
                oof.loc[oof.lead == lead, "weight"].to_numpy(),
            )
            for c in [*model_cols, "blend"]
        }
        for lead in LEADS
    }
    metrics["by_fold"] = {
        str(fold): weighted_rmse(
            oof.loc[oof.fold == fold, "y"].to_numpy(),
            oof.loc[oof.fold == fold, "blend"].to_numpy(),
            oof.loc[oof.fold == fold, "weight"].to_numpy(),
        )
        for fold in sorted(oof.fold.unique())
    }
    return lead_coef, metrics


def fit_final_predict(
    data: pd.DataFrame,
    feature_cols: list[str],
    test_X: pd.DataFrame,
    blend_coef: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    X = data[feature_cols]
    hs0 = data["hs_lag_0m"].to_numpy()
    test_hs0 = test_X["hs_lag_0m"].to_numpy()
    weights = event_weights(data.reset_index(drop=True))
    predictions: dict[int, np.ndarray] = {}

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_extra = imputer.fit_transform(X)
    test_X_extra = imputer.transform(test_X)
    y_extra = np.column_stack([data[f"target_{lead}h"].to_numpy() - hs0 for lead in LEADS])
    extra = ExtraTreesRegressor(
        n_estimators=140,
        min_samples_leaf=10,
        max_features=0.55,
        n_jobs=-1,
        random_state=8000,
    )
    extra.fit(X_extra, y_extra, sample_weight=weights)
    p_extra_all = test_hs0[:, None] + extra.predict(test_X_extra)
    cat = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=520,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        random_seed=10000,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    cat.fit(X, y_extra, sample_weight=weights)
    p_cat_all = test_hs0[:, None] + cat.predict(test_X)

    for lead in LEADS:
        y_abs = data[f"target_{lead}h"].to_numpy()
        y_delta = y_abs - hs0
        smooth = fit_lgb(X, y_delta, weights, 3000 + lead, "smooth")
        ridge = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=120.0),
        )
        ridge.fit(X, y_abs, ridge__sample_weight=weights)
        candidates = np.column_stack(
            [
                test_hs0,
                test_hs0 + smooth.predict(test_X),
                ridge.predict(test_X),
                p_extra_all[:, LEADS.index(lead)],
                p_cat_all[:, LEADS.index(lead)],
            ]
        )
        predictions[lead] = np.clip(candidates @ blend_coef[lead], 0.0, 30.0)
    return predictions


def validate_submission(sub: pd.DataFrame, index: pd.DataFrame) -> None:
    expected_cols = ["case_id", "station", "lead_h", "hs_pred"]
    if list(sub.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {list(sub.columns)}")
    if len(sub) != len(index):
        raise ValueError("Submission row count differs from test_index.csv")
    if not sub.iloc[:, :3].reset_index(drop=True).equals(index.reset_index(drop=True)):
        raise ValueError("Submission keys/order differ from test_index.csv")
    values = sub.hs_pred.to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 30)).any():
        raise ValueError("Predictions must be finite and within [0, 30]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "데이터셋_P3" / "P3_wave_forecast",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "submission_p3.csv")
    args = parser.parse_args()
    start = time.time()

    data, feature_cols = load_training(args.data_dir)
    test_X, test_meta = load_test(args.data_dir, feature_cols)
    print(f"training examples={len(data):,}, features={len(feature_cols)}, test cases={len(test_X)}")

    blend_coef, metrics = cross_validate(data, feature_cols)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    pred = fit_final_predict(data, feature_cols, test_X, blend_coef)
    index = pd.read_csv(args.data_dir / "test_index.csv")
    lookup = {(test_meta.case_id.iloc[i], lead): pred[lead][i] for i in range(len(test_meta)) for lead in LEADS}
    sub = index.copy()
    sub["hs_pred"] = [lookup[(str(r.case_id), int(r.lead_h))] for r in index.itertuples(index=False)]
    validate_submission(sub, index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False, float_format="%.6f")

    report = {
        "data_dir": str(args.data_dir.resolve()),
        "output": str(args.output.resolve()),
        "training_examples": int(len(data)),
        "features": int(len(feature_cols)),
        "test_cases": int(len(test_X)),
        "blend": metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and {report_path} in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
