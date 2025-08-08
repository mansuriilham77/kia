from __future__ import annotations

import numpy as np
from typing import List


def naive_forecast(y: List[float] | np.ndarray, horizon: int) -> np.ndarray:
    """
    Naive: prediksi = nilai bulan terakhir yang diketahui, diulang untuk h langkah.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([])
    last = y[-1]
    return np.array([last] * horizon, dtype=float)


def seasonal_naive_forecast(
    y: List[float] | np.ndarray, horizon: int, season_length: int = 12
) -> np.ndarray:
    """
    Seasonal naive: prediksi h di depan = nilai pada t - season_length.
    Bila panjang data < season_length, fallback ke naive.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < season_length:
        return naive_forecast(y, horizon)
    out = []
    for h in range(1, horizon + 1):
        idx = n - season_length + (h - 1)
        if idx < n:
            out.append(y[idx])
        else:
            # Jika melampaui panjang, ulangi secara musiman
            out.append(out[idx - n])
    return np.asarray(out, dtype=float)
