"""Explicit AKShare provider snapshots with bounded workers and provenance."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd

from .data import MarketPanel


PROVIDERS = {
    "eastmoney": {
        "name": "AKShare / Eastmoney fund_etf_hist_em",
        "endpoint": "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        "volume_unit": "provider-native",
    },
    "tencent": {
        "name": "AKShare / Tencent stock_zh_a_hist_tx",
        "endpoint": "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
        "volume_unit": "shares (normalized by AKShare)",
    },
}


def normalize_akshare(frame, symbol, start, end):
    df = frame.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}).copy()
    columns = ["date", "tic", "open", "high", "low", "close", "volume"]
    df["tic"] = str(symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df.date.between(pd.Timestamp(start), pd.Timestamp(end)), columns]
    if df.empty:
        raise ValueError(f"No data for {symbol} in requested period")
    return df


def _read_snapshot(path, symbol, start, end):
    frame = pd.read_csv(path, dtype={"tic": str}, parse_dates=["date"])
    panel = MarketPanel.from_frame(frame)
    if panel.tickers != (symbol,):
        raise ValueError(f"Unexpected ticker in provider snapshot for {symbol}")
    if not frame.date.between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError(f"Provider snapshot outside requested dates for {symbol}")
    return frame


def _fetch_snapshot(symbol, start, end, folder, cache, timeout, max_attempts, retry_backoff, provider):
    cache_path = cache / f"{symbol}_{start}_{end}_hfq.csv"
    metadata_path = cache_path.with_suffix(".metadata.json")
    if cache_path.is_file() and metadata_path.is_file():
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_identity = {
            "provider_key": provider, "provider": PROVIDERS[provider]["name"],
            "ticker": symbol, "adjust": "hfq", "requested_start": start, "requested_end": end,
        }
        if any(info.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(f"Cached snapshot identity mismatch: {metadata_path}")
        digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        if info.get("sha256") != digest:
            raise RuntimeError(f"Cached snapshot checksum mismatch: {cache_path}")
        print(f"Using cached {symbol} ({start}..{end}, hfq)", flush=True)
        return _read_snapshot(cache_path, symbol, start, end), {**info, "from_cache": True}
    path = Path(folder) / f"{symbol}.csv"
    args = [sys.executable, "-m", "rlasset.download", provider, symbol, start, end, str(path)]
    for attempt in range(1, max_attempts + 1):
        print(f"Fetching {symbol} via {provider} ({start}..{end}, hfq), attempt {attempt}/{max_attempts}...", flush=True)
        try:
            subprocess.run(args, check=True, timeout=timeout, capture_output=True, text=True, errors="replace")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = (getattr(exc, "stderr", "") or str(exc))[-1000:]
            if attempt == max_attempts:
                raise RuntimeError(
                    f"AKShare/{provider} download failed for {symbol} after {max_attempts} attempts. "
                    f"Completed symbols remain cached in {cache}. Provider error: {detail}"
                ) from exc
            pause = min(retry_backoff * (2 ** (attempt - 1)), 30)
            print(f"Transient provider/worker failure for {symbol}; retry in {pause:g}s", flush=True)
            if pause:
                time.sleep(pause)
    frame = _read_snapshot(path, symbol, start, end)
    info = {
        "ticker": symbol, "rows": len(frame), "first": str(frame.date.min().date()),
        "last": str(frame.date.max().date()), "provider": PROVIDERS[provider]["name"],
        "provider_key": provider, "endpoint": PROVIDERS[provider]["endpoint"],
        "volume_unit": PROVIDERS[provider]["volume_unit"],
        "adjust": "hfq", "requested_start": start, "requested_end": end,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "akshare_version": importlib.metadata.version("akshare"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    # Persist every validated symbol immediately; a later failure can resume.
    staging = cache_path.with_suffix(".csv.partial")
    staging.write_bytes(path.read_bytes())
    staging.replace(cache_path)
    metadata_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, {**info, "from_cache": False}


def download_etfs(symbols, start, end, output, timeout=90, max_attempts=3,
                  retry_backoff=2, cache_dir=None, provider="eastmoney"):
    """Download/cache a fixed hfq snapshot with bounded retries and resumability.

    Cache keys include exact requested dates and the adjustment mode. A repeated
    request reuses the same historical snapshot; choose a new ``cache_dir`` to
    deliberately refresh it. Never substitutes another provider or synthetic data.
    """
    if provider not in PROVIDERS:
        raise ValueError("provider must be explicitly 'eastmoney' or 'tencent'")
    symbols = list(symbols)
    if not symbols or len(set(symbols)) != len(symbols) or not all(len(s) == 6 and s.isdigit() for s in symbols):
        raise ValueError("Use distinct six-digit ETF tickers")
    if max_attempts < 1 or retry_backoff < 0 or timeout <= 0:
        raise ValueError("max_attempts/timeout must be positive; retry_backoff nonnegative")
    start = pd.Timestamp(start).strftime("%Y%m%d")
    end = pd.Timestamp(end).strftime("%Y%m%d")
    if start > end:
        raise ValueError("start must not be after end")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir is not None else output.parent / "cache"
    cache = cache_root / f"akshare_{provider}"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    raw_info = []
    with tempfile.TemporaryDirectory(prefix="rlasset-") as folder:
        for symbol in symbols:
            df, info = _fetch_snapshot(symbol, start, end, folder, cache, timeout, max_attempts, retry_backoff, provider)
            frames.append(df)
            raw_info.append(info)
    common_start = max(x.date.min() for x in frames)
    common_end = min(x.date.max() for x in frames)
    frame = pd.concat([x.loc[x.date.between(common_start, common_end)] for x in frames]).sort_values(["date", "tic"])
    panel = MarketPanel.from_frame(frame)  # reject interior missing dates / zero-volume sessions
    frame.to_csv(output, index=False)
    metadata = {
        "kind": "market", "provider": PROVIDERS[provider]["name"], "provider_key": provider,
        "endpoint": PROVIDERS[provider]["endpoint"], "volume_unit": PROVIDERS[provider]["volume_unit"], "adjust": "hfq",
        "requested_start": start, "requested_end": end, "tickers": list(panel.tickers),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "akshare_version": importlib.metadata.version("akshare"),
        "first_date": str(panel.dates[0].date()), "last_date": str(panel.dates[-1].date()),
        "sessions": len(panel.dates), "per_symbol": raw_info,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": "Retrospective adjusted-price snapshot, not point-in-time corporate-action database. Fixed ETF universe has selection bias. Fractional portfolio research only."
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import akshare as ak
    provider, symbol, start, end, target = sys.argv[1:]
    if provider == "eastmoney":
        result = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="hfq")
    elif provider == "tencent":
        exchange = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
        result = ak.stock_zh_a_hist_tx(symbol=exchange + symbol, start_date=start, end_date=end, adjust="hfq", timeout=15)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    normalize_akshare(result, symbol, start, end).to_csv(target, index=False)
