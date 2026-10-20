import pandas as pd
import pytest
import subprocess
from pathlib import Path
import errno
import json


def test_vendor_columns_become_canonical_and_stay_in_requested_period():
    from rlasset.download import normalize_akshare
    df = pd.DataFrame({"日期": ["2020-01-02", "2020-01-03"], "开盘": [1., 2.], "收盘": [1.1, 2.1], "最高": [1.2, 2.2], "最低": [.9, 1.9], "成交量": [100, 200]})
    out = normalize_akshare(df, "159915", "20200103", "20200104")
    assert len(out) == 1
    assert out.iloc[0].tic == "159915"
    assert out.iloc[0].open == 2.
    with pytest.raises(ValueError):
        normalize_akshare(df, "159915", "20300101", "20300103")


def _write_worker_result(path, symbol):
    # Complete, hand-checked provider-boundary fixture; no synthetic substitute
    # is installed in production, and the normalizer/panel/cache remain real.
    pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
        "tic": [symbol] * 3,
        "open": [1., 1.1, 1.2], "close": [1.1, 1.2, 1.3],
        "high": [1.2, 1.3, 1.4], "low": [.9, 1., 1.1],
        "volume": [100., 200., 300.],
    }).to_csv(path, index=False)


def test_transient_worker_failure_retries_and_returns_real_validated_panel(tmp_path, monkeypatch):
    from rlasset.download import download_etfs
    from rlasset.data import load_panel

    attempts = []
    def worker(args, **kwargs):
        attempts.append(args[-4])
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, args, stderr="ProxyError")
        _write_worker_result(args[-1], args[-4])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", worker)
    target = tmp_path / "panel.csv"
    meta = download_etfs(["510300"], "20200102", "20200106", target,
                         max_attempts=2, retry_backoff=0)
    assert len(attempts) == 2
    assert meta["kind"] == "market"
    assert load_panel(target).close[:, 0].tolist() == [1.1, 1.2, 1.3]


def test_partial_success_is_cached_and_resumed_without_redownload(tmp_path, monkeypatch):
    from rlasset.download import download_etfs
    from rlasset.data import load_panel

    target = tmp_path / "panel.csv"
    first_calls = []
    def interrupted_worker(args, **kwargs):
        symbol = args[-4]
        first_calls.append(symbol)
        if symbol == "510500":
            raise subprocess.TimeoutExpired(args, 1)
        _write_worker_result(args[-1], symbol)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", interrupted_worker)
    with pytest.raises(RuntimeError, match="510500"):
        download_etfs(["510300", "510500"], "20200102", "20200106", target,
                      max_attempts=2, retry_backoff=0)
    assert not target.exists()
    assert first_calls == ["510300", "510500", "510500"]
    resumed_calls = []
    def resumed_worker(args, **kwargs):
        resumed_calls.append(args[-4])
        _write_worker_result(args[-1], args[-4])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", resumed_worker)
    meta = download_etfs(["510300", "510500"], "20200102", "20200106", target,
                         max_attempts=2, retry_backoff=0)
    assert resumed_calls == ["510500"]
    assert load_panel(target).tickers == ("510300", "510500")
    assert meta["per_symbol"][0]["from_cache"] is True


def test_explicit_providers_have_separate_caches_and_metadata(tmp_path, monkeypatch):
    from rlasset.download import download_etfs

    calls = []
    def worker(args, **kwargs):
        calls.append(args[-5])
        _write_worker_result(args[-1], args[-4])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", worker)
    common = dict(symbols=["510300"], start="20200102", end="20200106",
                  output=tmp_path / "panel.csv", max_attempts=1)
    eastmoney = download_etfs(**common, provider="eastmoney")
    tencent = download_etfs(**common, provider="tencent")
    tencent_again = download_etfs(**common, provider="tencent")
    assert calls == ["eastmoney", "tencent"]
    assert eastmoney["provider"] == "AKShare / Eastmoney fund_etf_hist_em"
    assert tencent["provider"] == "AKShare / Tencent stock_zh_a_hist_tx"
    assert tencent_again["per_symbol"][0]["from_cache"] is True
    with pytest.raises(ValueError, match="provider"):
        download_etfs(**common, provider="automatic")


def test_cache_persistence_does_not_rename_across_filesystems(tmp_path, monkeypatch):
    from rlasset.download import download_etfs

    original_replace = Path.replace
    def cross_device_replace(source, target):
        # On Windows the worker temp directory can be on C: and the cache on W:.
        if source.parent != Path(target).parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_replace(source, target)
    def worker(args, **kwargs):
        _write_worker_result(args[-1], args[-4])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(Path, "replace", cross_device_replace)
    monkeypatch.setattr(subprocess, "run", worker)
    meta = download_etfs(["510300"], "20200102", "20200106", tmp_path / "panel.csv")
    assert meta["sessions"] == 3


@pytest.mark.parametrize("field,value", [("provider_key", "tencent"), ("adjust", "qfq"),
                                        ("ticker", "510500"), ("requested_start", "20190101")])
def test_cache_rejects_mislabeled_source_or_request(tmp_path, monkeypatch, field, value):
    from rlasset.download import download_etfs

    def worker(args, **kwargs):
        _write_worker_result(args[-1], args[-4])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", worker)
    output = tmp_path / "panel.csv"
    download_etfs(["510300"], "20200102", "20200106", output)
    cache_meta = tmp_path / "cache/akshare_eastmoney/510300_20200102_20200106_hfq.metadata.json"
    changed = json.loads(cache_meta.read_text(encoding="utf-8"))
    changed[field] = value
    cache_meta.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        download_etfs(["510300"], "20200102", "20200106", output)
