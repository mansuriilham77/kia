from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import holidays as pyholidays


@dataclass
class PrepConfig:
    date_column: str
    target_column: str
    lags: List[int]
    rollings: List[int]
    holiday_country: str = "ID"


def _to_month_start(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return ts.to_period("M").to_timestamp("MS")


def load_and_validate(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]

    if dcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Kolom wajib tidak ditemukan. Harus ada: {dcol}, {ycol}")

    out = df[[dcol, ycol]].copy()
    out[dcol] = out[dcol].apply(_to_month_start)

    # Validasi nilai
    if out[ycol].isna().any():
        raise ValueError("Terdapat nilai kosong pada kolom target.")

    # Pastikan integer non-negatif
    if not np.issubdtype(out[ycol].dtype, np.number):
        try:
            out[ycol] = out[ycol].astype(int)
        except Exception:
            raise ValueError("Kolom target harus bertipe numerik/integer.")
    if (out[ycol] < 0).any():
        raise ValueError("Nilai target tidak boleh negatif.")

    # Drop duplikat bulan (ambil terakhir jika ada)
    out = out.drop_duplicates(subset=[dcol], keep="last")

    out = out.sort_values(dcol).reset_index(drop=True)
    return out


def _holiday_counter(years: List[int], country_code: str):
    # Build holiday calendar for given years
    years = sorted(set(years))
    return pyholidays.country_holidays(country_code=country_code, years=years)


def add_calendar_features(df: pd.DataFrame, prep: PrepConfig) -> pd.DataFrame:
    dcol = prep.date_column

    out = df.copy()
    out["month"] = out[dcol].dt.month
    out["quarter"] = out[dcol].dt.quarter
    out["year"] = out[dcol].dt.year

    # Hitung jumlah hari libur dalam setiap bulan
    years = list(range(out[dcol].dt.year.min(), out[dcol].dt.year.max() + 2))
    hcal = _holiday_counter(years, prep.holiday_country)

    def count_holidays_in_month(ts: pd.Timestamp) -> int:
        start = ts
        end = (ts + pd.offsets.MonthEnd(0))
        days = pd.date_range(start, end, freq="D")
        return sum(1 for d in days if d in hcal)

    out["holiday_count"] = out[dcol].apply(count_holidays_in_month)
    return out


def add_lag_rolling_features(df: pd.DataFrame, prep: PrepConfig) -> pd.DataFrame:
    dcol = prep.date_column
    ycol = prep.target_column

    out = df.copy()
    for lag in prep.lags:
        out[f"{ycol}_lag{lag}"] = out[ycol].shift(lag)

    for win in prep.rollings:
        out[f"{ycol}_roll{win}"] = out[ycol].rolling(window=win, min_periods=win).mean()

    return out


def make_supervised(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    prep = PrepConfig(
        date_column=cfg["data"]["date_column"],
        target_column=cfg["data"]["target_column"],
        lags=cfg["data"]["lags"],
        rollings=cfg["data"]["rollings"],
        holiday_country=cfg["data"].get("holiday_country", "ID"),
    )

    base = add_calendar_features(df, prep)
    sup = add_lag_rolling_features(base, prep)

    # Fitur yang dipakai model (urutan konsisten)
    ycol = prep.target_column
    feat_cols = (
        ["month", "quarter", "holiday_count"]
        + [f"{ycol}_lag{l}" for l in prep.lags]
        + [f"{ycol}_roll{r}" for r in prep.rollings]
    )

    sup = sup.dropna(subset=feat_cols).reset_index(drop=True)
    return sup[[prep.date_column, ycol] + feat_cols], feat_cols
