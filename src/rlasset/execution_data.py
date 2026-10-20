"""Strict raw-price snapshots for integer-share execution; never adjusted fallback."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .data import MarketPanel

ENDPOINT = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
ETF_UNIVERSE = ("159915", "510050", "510300", "510500")


def decode_unadjusted(payload, ticker):
    symbol = ("sh" if ticker.startswith("5") else "sz") + ticker
    data = payload.get("data", {}).get(symbol, {})
    if not data.get("day"):
        raise ValueError(f"Missing unadjusted day data for {ticker}; adjusted fallback refused")
    rows = [row[:6] for row in data["day"]]
    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    frame["date"] = pd.to_datetime(frame.date)
    for field in ["open", "close", "high", "low", "volume"]:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    # Tencent ETF volume is hands; AKShare 1.19.1 normalizes these symbols x100.
    frame["volume"] *= 100
    frame["tic"] = ticker
    values = frame[["open", "close", "high", "low", "volume"]].to_numpy()
    if (not np.isfinite(values).all() or (values <= 0).any()
            or (frame.high < frame[["open", "close"]].max(axis=1)).any()
            or (frame.low > frame[["open", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid unadjusted OHLCV bar")
    return frame


def download_execution_prices(output, start="20220101", end="20261016",
                              tickers=ETF_UNIVERSE):
    """Fetch explicit unadjusted day keys with bounded retries and raw JSON cache."""
    if not set(tickers) <= set(ETF_UNIVERSE):
        raise ValueError("Execution volume normalization is audited only for the fixed ETF universe")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "execution_raw_cache"
    cache.mkdir(exist_ok=True)
    all_frames, sources = [], []
    for ticker in tickers:
        frames = []
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            target = cache / f"{ticker}_{year}_unadjusted.json"
            if target.exists():
                payload = json.loads(target.read_text(encoding="utf-8"))
            else:
                symbol = ("sh" if ticker.startswith("5") else "sz") + ticker
                params = {"param": f"{symbol},day,{year}-01-01,{year}-12-31,640,"}
                for attempt in range(3):
                    try:
                        response = requests.get(ENDPOINT, params=params, timeout=20)
                        response.raise_for_status()
                        payload = response.json()
                        decode_unadjusted(payload, ticker)
                        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            frame = decode_unadjusted(payload, ticker)
            # Servers can return data beyond requested year; keep explicit interval.
            frame = frame.loc[frame.date.dt.year == year]
            frames.append(frame)
            sources.append({"file": str(target.resolve()), "ticker": ticker, "year": year,
                            "response_key": "day", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            print(f"Unadjusted {ticker} {year}: {len(frame)} rows", flush=True)
        single = pd.concat(frames).drop_duplicates()
        all_frames.append(single.loc[single.date.between(pd.Timestamp(start), pd.Timestamp(end))])
    result = pd.concat(all_frames).sort_values(["date", "tic"])
    panel = MarketPanel.from_frame(result)
    result.to_csv(output, index=False)
    metadata = {
        "kind": "market", "adjust": "none", "provider": "Tencent raw day endpoint",
        "endpoint": ENDPOINT, "volume_unit": "shares", "volume_multiplier": 100,
        "first_date": str(panel.dates[0].date()), "last_date": str(panel.dates[-1].date()),
        "sessions": len(panel.dates), "tickers": list(panel.tickers),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": "Retrospective daily snapshot; not a tick feed. Corporate cash distributions are excluded from price-only paper P&L.",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/ashare_etf_unadjusted.csv")
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default="20261016")
    args = parser.parse_args()
    download_execution_prices(args.output, args.start, args.end)
