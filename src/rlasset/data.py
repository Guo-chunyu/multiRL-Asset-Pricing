"""Strict point-in-time panels. Never backfill prices or normalize on future rows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    open: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame):
        required = ["date", "tic", "open", "high", "low", "close", "volume"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Required columns: {required}")
        df = frame[required].copy()
        df["date"] = pd.to_datetime(df.date, errors="raise").dt.normalize()
        if df.isna().any().any() or df.duplicated(["date", "tic"]).any():
            raise ValueError("Missing values or duplicate date/ticker rows")
        df["tic"] = df.tic.astype(str)
        dates = pd.DatetimeIndex(sorted(df.date.unique()))
        tickers = tuple(sorted(df.tic.unique()))
        if len(dates) < 3 or not tickers or len(df) != len(dates) * len(tickers):
            raise ValueError("Panel must contain every ticker on every date; no implicit fill")
        arrays = {}
        for col in required[2:]:
            values = df.pivot(index="date", columns="tic", values=col).reindex(index=dates, columns=tickers).to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{col} must be finite and positive (suspensions require an execution model)")
            arrays[col] = values
        if np.any(arrays["high"] + 1e-7 < np.maximum(arrays["open"], arrays["close"])) or np.any(arrays["low"] - 1e-7 > np.minimum(arrays["open"], arrays["close"])):
            raise ValueError("Inconsistent OHLC ordering")
        return cls(dates=dates, tickers=tickers, **arrays)


def load_panel(path: str | Path) -> MarketPanel:
    return MarketPanel.from_frame(pd.read_csv(path, dtype={"tic": str}))


def chronological_splits(n: int, lookback: int = 20) -> dict[str, tuple[int, int]]:
    """Disjoint state-date ranges; discard the one return crossing each boundary."""
    usable = n - lookback
    if lookback < 2 or usable < 80:
        raise ValueError("Need at least lookback + 80 dates for four chronological partitions")
    edges = [lookback, lookback + int(usable * .50), lookback + int(usable * .70), lookback + int(usable * .85), n]
    return {name: (edges[i], edges[i + 1] - 1) for i, name in enumerate(("train", "meta", "validation", "test"))}


def validate_splits(splits, n, lookback):
    names = ("train", "meta", "validation", "test")
    if set(splits) != set(names):
        raise ValueError("Require train, meta, validation and test intervals")
    previous = lookback - 1
    result = {}
    for name in names:
        start, end = splits[name]
        if not all(isinstance(x, (int, np.integer)) for x in (start, end)):
            raise ValueError("Split indices must be integers")
        if not previous < start < end < n:
            raise ValueError("Intervals must be chronological, disjoint and within the panel")
        result[name] = (int(start), int(end))
        previous = end
    return result


def synthetic_frame(periods: int = 800, assets: int = 4, seed: int = 7) -> pd.DataFrame:
    """Regime-switching artificial data, solely for software/learning smoke tests."""
    if periods < 3 or assets < 1:
        raise ValueError("periods >= 3 and assets >= 1 required")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=periods)
    common = rng.normal(0, .006, periods)
    regimes = np.where((np.arange(periods) // 100) % 2 == 0, 1., -1.)
    returns = common[:, None] + rng.normal(0, .007, (periods, assets)) + regimes[:, None] * np.linspace(-.0008, .0012, assets)
    close = 10 * np.exp(np.cumsum(returns, axis=0))
    previous = np.vstack([np.full(assets, 10.), close[:-1]])
    open_prices = previous * np.exp(rng.normal(0, .003, close.shape))
    high = np.maximum(open_prices, close) * (1 + rng.uniform(.001, .01, close.shape))
    low = np.minimum(open_prices, close) * (1 - rng.uniform(.001, .01, close.shape))
    return pd.DataFrame({"date": np.repeat(dates, assets), "tic": np.tile([f"SYN{i:03}" for i in range(assets)], periods), "open": open_prices.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel(), "volume": rng.integers(10000, 1000000, periods * assets)})
