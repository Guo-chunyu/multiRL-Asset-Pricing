# Published research results

**2026-10-23 correction:** v1/v2 Tencent hfq price ratios distort ETF returns.
Their adjusted-price performance interpretation has been withdrawn; historical
files remain for audit. See [data correction](../docs/data-audit-v3.md).
The raw-price paper ledgers remain price-only records, excluding dividends.

Current study: [v3 conclusions](../docs/results-v3.md),
[full report](runs/research_v3/report.md), [metrics CSV](runs/research_v3/metrics.csv),
[paired intervals CSV](runs/research_v3/excess_intervals.csv),
[independent accounting audit](runs/verification_v3.json).

This is a read-only export of the completed local experiments, with numerical
results preserved. Workspace paths are replaced with `${PROJECT_ROOT}` and
user-profile paths with `${USERPROFILE}`; machine hostnames are removed from
JUnit records. `export_manifest.json` records both original and published hashes.

- [Chinese study conclusions](../docs/results-v2.md)
- [Five-algorithm statistical report and figures](runs/research_v2/report.md)
- [Main experiment summary](runs/comparison_v2/summary.json)
- [Walk-forward summary](runs/walkforward_v2/walkforward_summary.json)
- [Paper execution summary](runs/paper_v2/summary.json)
- [Verification evidence](runs/verification_v2.json)

These files are historical, retrospective research evidence. They are not live
broker statements, profitability guarantees, or untouched out-of-sample results.

Model checkpoint ZIPs, downloaded market data, Python environments, vendored
repositories, and third-party paper full text are not published. Fresh runs write
to `runs/`, which is ignored by Git. Install dependencies and obtain data using
the root README before starting a fresh experiment; this archive cannot be used
as a `--resume` directory.

To regenerate this export after a local study:

```powershell
.\.venv\Scripts\python.exe scripts/export_public_results.py
```
