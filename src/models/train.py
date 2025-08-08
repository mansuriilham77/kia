from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import yaml
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.pipeline.data_prep import load_and_validate, make_supervised
from src.models.evaluate import metrics_all
from src.models.baselines import naive_forecast, seasonal_naive_forecast
from src.models.infer import forecast_iterative_xgb


def load_config(path: str = "config/config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _evaluate_baselines(hist_df: pd.DataFrame, cfg: Dict, horizon: int) -> Dict[str, Dict]:
    y = hist_df[cfg["data"]["target_column"]].values
    n = y.size
    # Split
    y_train = y[: n - horizon]
    y_test = y[n - horizon :]

    preds = {}
    preds["naive"] = naive_forecast(y_train, horizon=horizon)
    preds["seasonal_naive"] = seasonal_naive_forecast(y_train, horizon=horizon, season_length=12)

    scores = {}
    for name, yhat in preds.items():
        scores[name] = metrics_all(y_test, yhat)
    return scores


def _fit_xgb(sup_df: pd.DataFrame, feature_names: List[str], cutoff_date: pd.Timestamp, cfg: Dict):
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]

    train_mask = sup_df[dcol] <= cutoff_date
    X_train = sup_df.loc[train_mask, feature_names]
    y_train = sup_df.loc[train_mask, ycol]

    params = cfg["model"]["xgb_params"]
    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=params.get("random_state", 42),
        tree_method=params.get("tree_method", "auto"),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_pipeline(df: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Latih model dan pilih yang terbaik berdasarkan MAPE pada holdout terakhir.
    Mengembalikan artifact dict:
      - model_name: 'xgboost' | 'seasonal_naive' | 'naive'
      - scores: {model: {MAE, RMSE, MAPE}}
      - cfg: konfigurasi
      - feature_names: daftar fitur (jika xgboost)
      - model: objek model (jika xgboost)
    """
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    holdout = int(cfg["training"]["holdout_months"])
    min_rows = int(cfg["training"]["min_train_rows"])

    # Pastikan df valid & disort
    df = load_and_validate(df, cfg)
    hist = df[[dcol, ycol]].copy()
    n = len(hist)
    if n <= holdout + 1:
        raise ValueError("Data terlalu pendek untuk evaluasi holdout.")

    # Skor baseline pada holdout
    baseline_scores = _evaluate_baselines(hist, cfg, horizon=holdout)

    # Siapkan supervised + fitur
    sup, feature_names = make_supervised(df, cfg)

    # Tentukan cutoff untuk melatih tanpa melihat holdout
    test_first_date = hist[dcol].iloc[n - holdout]
    cutoff_date = test_first_date - pd.offsets.MonthBegin(1)

    # Coba latih XGB bila data memadai
    xgb_ok = sup[sup[dcol] <= cutoff_date].shape[0] >= min_rows
    xgb_scores = None
    model = None

    if xgb_ok:
        try:
            model = _fit_xgb(sup, feature_names, cutoff_date, cfg)
            # Prediksi iteratif sepanjang holdout
            hist_train = hist[hist[dcol] <= cutoff_date]
            fc = forecast_iterative_xgb(
                df_hist=hist_train, artifact={"model": model, "cfg": cfg, "feature_names": feature_names},
                horizon=holdout
            )
            y_test = hist[hist[dcol] > cutoff_date][ycol].values
            xgb_scores = metrics_all(y_test, fc["y_pred"].values)
        except Exception:
            model = None
            xgb_scores = None

    # Pilih model terbaik berdasarkan MAPE
    candidates = []
    for name, s in baseline_scores.items():
        candidates.append((name, s["MAPE"]))
    if xgb_scores is not None:
        candidates.append(("xgboost", xgb_scores["MAPE"]))

    best_name, _ = sorted(candidates, key=lambda x: x[1])[0]

    # Susun skor untuk semua kandidat yang tersedia
    scores = dict(baseline_scores)
    if xgb_scores is not None:
        scores["xgboost"] = xgb_scores

    artifact = {
        "model_name": best_name,
        "scores": scores,
        "cfg": cfg,
    }
    if best_name == "xgboost" and model is not None:
        artifact["model"] = model
        artifact["feature_names"] = feature_names

    return artifact


def save_artifact(artifact: Dict, out_dir: str = "models", filename_prefix: str = "kia_forecast") -> Tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = dict(artifact)  # shallow copy
    model_path = ""
    if meta.get("model_name") == "xgboost" and "model" in meta:
        model_path = str(out / f"{filename_prefix}_model.pkl")
        joblib.dump(meta["model"], model_path)
        # hapus objek model dari meta
        meta.pop("model", None)

    meta_path = str(out / f"{filename_prefix}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model_path, meta_path
