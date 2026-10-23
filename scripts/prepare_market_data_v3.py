"""Prepare immutable raw-market/corporate-action inputs; never inspect returns.

Run after installing the project: python scripts/prepare_market_data_v3.py
The fixed universe and start date were selected for historical availability,
not performance. 510500 was suspended on 2015-04-13/14 for share conversion;
the first subsequent complete panel starts on 2015-04-15. No interior filling.
"""
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

from rlasset.data import MarketPanel
from rlasset.execution_data import download_execution_prices


TICKERS = ("510050", "510300", "510500")
START, END = "2015-04-15", "2026-10-22"
SPLIT_SOURCE = "https://pdf.dfcfw.com/pdf/H2_AN202208291577764682_1.pdf"
SUSPENSION_SOURCE = "https://epaper.stcn.com/paper/zqsb/page/1/2015-04/10/B148/20150410B148_pdf.pdf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_get(path, url, params=None):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=20, headers={
                "User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/",
            })
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    out = Path("data/v3")
    cache = Path("data/cache_v3")
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "ashare_broad_raw.csv"
    if not raw_path.exists():
        download_execution_prices(raw_path, start=START.replace("-", ""),
                                  end=END.replace("-", ""), tickers=TICKERS)
    raw = pd.read_csv(raw_path, dtype={"tic": str})
    panel = MarketPanel.from_frame(raw)
    metadata = json.loads(raw_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["sha256"] == digest(raw_path)
    assert panel.tickers == TICKERS
    assert str(panel.dates[0].date()) == START and str(panel.dates[-1].date()) == END

    cash, provenance = [], []
    for ticker in TICKERS:
        url = f"https://fundf10.eastmoney.com/fhsp_{ticker}.html"
        path = cache / "corporate_actions" / f"{ticker}_fhsp.html"
        text = cached_get(path, url).decode("utf-8")
        tables = pd.read_html(StringIO(text))
        dividends = next(table for table in tables if "\u9664\u606f\u65e5" in table.columns)
        splits = next(table for table in tables if "\u62c6\u5206\u6298\u7b97\u65e5" in table.columns)
        for _, row in dividends.iterrows():
            ex_date = str(row["\u9664\u606f\u65e5"])
            if START <= ex_date <= END:
                amount = re.fullmatch(r"\u6bcf10\u4efd\u6d3e\u73b0\u91d1([0-9.]+)\u5143", str(row["\u6bcf10\u4efd\u5206\u7ea2"]))
                if amount is None:
                    raise ValueError(f"Unrecognized cash distribution: {row.to_dict()}")
                cash.append({"ticker": ticker, "ex_date": ex_date,
                             "pay_date": str(row["\u5206\u7ea2\u53d1\u653e\u65e5"]),
                             "cash_per_share": float(amount[1]) / 10,
                             "record_date": str(row["\u6743\u76ca\u767b\u8bb0\u65e5"]), "source": url})
        relevant_splits = splits.loc[splits["\u62c6\u5206\u6298\u7b97\u65e5"].between(START, END)]
        expected_dates = ["2022-08-26"] if ticker == "510500" else []
        assert relevant_splits["\u62c6\u5206\u6298\u7b97\u65e5"].tolist() == expected_dates
        provenance.append({"ticker": ticker, "url": url, "file": str(path), "sha256": digest(path),
                           "total_dividend_records": len(dividends), "total_split_records": len(splits)})

    actions = pd.DataFrame(cash).sort_values(["ex_date", "ticker"])
    assert not actions.duplicated(["ticker", "ex_date"]).any()
    assert (actions.cash_per_share > 0).all()
    assert (actions.record_date < actions.ex_date).all()
    assert (actions.pay_date >= actions.ex_date).all()
    dates = set(raw.date)
    assert set(actions.ex_date) <= dates and set(actions.record_date) <= dates
    sessions = sorted(dates)
    # Every event's record date is the immediately previous trading session.
    assert all(sessions[sessions.index(row.ex_date) - 1] == row.record_date
               for row in actions.itertuples())
    actions.to_csv(out / "cash_actions.csv", index=False)
    # Do not use Eastmoney's rounded 1.1454: the manager specifies 1.14539.
    split_rows = [{"ticker": "510500", "effective_date": "2022-08-29", "ratio": 1.14539,
                   "record_date": "2022-08-26", "source": SPLIT_SOURCE,
                   "rounding": "ceil_to_integer_share; fractional research may ignore account-size-dependent rounding"}]
    pd.DataFrame(split_rows).to_csv(out / "splits.csv", index=False)

    crosschecks = []
    for ticker in TICKERS:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        target = cache / "provider_audit" / f"{ticker}_20150415_20261022_eastmoney_raw.json"
        payload = json.loads(cached_get(target, url, {
            "secid": "1." + ticker, "klt": "101", "fqt": "0", "beg": "20150415",
            "end": "20261022", "lmt": "10000", "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }))
        em = pd.DataFrame([r.split(",") for r in payload["data"]["klines"]], columns=[
            "date", "open", "close", "high", "low", "volume", "amount", "amplitude",
            "change", "absolute_change", "turnover"])
        em = em.loc[em.date.between(START, END)].set_index("date")
        tx = raw.loc[raw.tic == ticker].set_index("date")
        assert not em.index.duplicated().any() and set(em.index) == set(tx.index)
        em = em.loc[tx.index]
        differences = tx[["open", "high", "low", "close"]] - em[["open", "high", "low", "close"]].astype(float)
        assert np.max(np.abs(differences.to_numpy())) <= 0.001000001
        crosschecks.append({"ticker": ticker, "sessions": len(tx), "date_grid_equal": True,
                            "max_abs_ohlc_difference_yuan": differences.abs().max().to_dict(),
                            "different_ohlc_cells": int((differences.abs() > 1e-8).sum().sum()),
                            "source": url, "source_file": str(target), "source_sha256": digest(target)})

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": START, "end": END, "tickers": list(TICKERS), "sessions": len(panel.dates),
        "rows": len(raw), "cash_events": len(actions), "split_events": len(split_rows),
        "input_sha256": {name: digest(out / name) for name in
                         ["ashare_broad_raw.csv", "cash_actions.csv", "splits.csv"]},
        "strict_panel_validation": "positive finite OHLCV, complete grid, valid OHLC, unique dates/tickers",
        "cash_event_validation": "positive cash/share, record/ex/pay ordering, record date previous session",
        "cash_sources": provenance, "raw_crosschecks": crosschecks,
        "primary_notice_spot_checks": [
            {"ticker": "510300", "record_date": "2026-01-16", "ex_date": "2026-01-19",
             "pay_date": "2026-01-27", "cash_per_share": 0.123,
             "url": "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-01-12/510300_20260112_VTCZ.pdf"},
            {"ticker": "510500", "record_date": "2026-01-15", "ex_date": "2026-01-16",
             "pay_date": "2026-01-21", "cash_per_share": 0.062,
             "url": "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-01-09/510500_20260109_5K7U.pdf"},
        ],
        "precise_split_source": SPLIT_SOURCE,
        "2015_suspension_source": SUSPENSION_SOURCE,
        "start_reason": "510500 officially suspended 2015-04-13/14 for conversion; first subsequent complete panel",
        "limitations": ["Retrospective public daily snapshots, not a point-in-time database.",
                        "Fixed three broad A-share ETFs; no individual-stock universe or survivorship study.",
                        "Tencent and Eastmoney raw prices disagree by at most one CNY 0.001 tick.",
                        "Cash schedule is an Eastmoney aggregate; only selected events checked against manager notices.",
                        "Corporate actions must be applied causally by accounting, never provider hfq price ratios.",
                        "No strategy returns or final-period performance were examined during preparation."],
    }
    (out / "preparation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["start", "end", "sessions", "rows", "cash_events", "split_events", "input_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
