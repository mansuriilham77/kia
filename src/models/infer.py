from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import holidays as pyholidays


def load_artifact(prefix_path: str) -> Dict:
    """
    Muat artefak model dari prefix path (tanpa ekstensi).
    Akan mencari {prefix}.json dan {prefix}_model.pkl (jika ada).
    """
    meta_path = Path(f"{prefix_path}.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("model_name") == "xgboost":
        model_path = Path(f"{prefix_path}_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
        model = joblib.load(model_path)
        meta["model"] = model
    return meta


def _holiday_calendar(country: str, years: List[int]):
    return pyholidays.country_holidays(country, years=sorted(set(years)))


def _count_holidays_in_month(ts: pd.Timestamp, hcal) -> int:
    start = ts
    end = (ts + pd.offsets.MonthEnd(0))
    days = pd.date_range(start, end, freq="D")
    return sum(1 for d in days if d in hcal)


def _build_feature_row(
    history: pd.DataFrame,
    date_col: str,
    target_col: str,
    next_date: pd.Timestamp,
    lags: List[int],
    rollings: List[int],
    country_code: str,
) -> Dict:
    """
    Bangun satu baris fitur untuk tanggal next_date berdasarkan history (berisi periode + target).
    """
    # Calendar features
    feat = {
        "month": next_date.month,
        "quarter": next_date.quarter,
    }
    years = [next_date.year]
    hcal = _holiday_calendar(country_code, years)
    feat["holiday_count"] = _count_holidays_in_month(next_date, hcal)

    # Lag features
    y = history[target_col].values
    for lag in lags:
        if len(y) >= lag:
            feat[f"{target_col}_lag{lag}"] = float(y[-lag])
        else:
            feat[f"{target_col}_lag{lag}"] = np.nan

    # Rolling means
    for win in rollings:
        if len(y) >= win:
            feat[f"{target_col}_roll{win}"] = float(pd.Series(y).tail(win).mean())
        else:
            feat[f"{target_col}_roll{win}"] = np.nan

    return feat


def forecast_iterative_xgb(df_hist: pd.DataFrame, artifact: Dict, horizon: int) -> pd.DataFrame:
    """
    Iterative forecasting untuk model XGBoost.
    - df_hist: Data historis (harus sudah bersih), kolom: [periode, permohonan_kia]
    - artifact: hasil dari load_artifact atau train_pipeline (harus mengandung 'model', 'feature_names', 'cfg')
    - horizon: jumlah bulan ke depan
    """
    cfg = artifact["cfg"]
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    lags = cfg["data"]["lags"]
    rolls = cfg["data"]["rollings"]
    country = cfg["data"].get("holiday_country", "ID")

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    work = df_hist[[dcol, ycol]].copy().sort_values(dcol).reset_index(drop=True)
    preds = []
    dates = []

    last_date = work[dcol].iloc[-1]
    for _ in range(horizon):
        next_date = (last_date + pd.offsets.MonthBegin(1))
        feat_row = _build_feature_row(
            history=work, date_col=dcol, target_col=ycol, next_date=next_date,
            lags=lags, rollings=rolls, country_code=country
        )
        X = pd.DataFrame([feat_row])[feature_names]
        yhat = float(model.predict(X)[0])

        preds.append(yhat)
        dates.append(next_date)

        # Append untuk langkah berikutnya
        work = pd.concat(
            [work, pd.DataFrame({dcol: [next_date], ycol: [yhat]})],
            ignore_index=True,
        )
        last_date = next_date

    return pd.DataFrame({dcol: dates, "y_pred": preds})
