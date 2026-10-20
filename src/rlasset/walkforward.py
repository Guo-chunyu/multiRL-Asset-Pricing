"""Expanding historical training with chronologically disjoint test blocks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_panel
from .experiment import ExperimentConfig, run_experiment


def expanding_splits(n: int, lookback: int = 20, folds: int = 3) -> list[dict[str, tuple[int, int]]]:
    """Reserve final 30% of dates for near-equal, nonoverlapping test blocks.

    Each fold puts a fixed 15% of all dates into meta training and 10% into
    validation immediately before its test block. The base-training start is
    fixed at ``lookback`` and its end expands. All boundaries are inclusive
    state-date indices, so each period omits the return crossing its start.
    Earlier test observations may legitimately become history for a later fold.
    """
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool)
           for value in (n, lookback, folds)):
        raise ValueError("n, lookback, and folds must be integers")
    if n < 1 or lookback < 2 or folds < 1:
        raise ValueError("Require n > 0, lookback >= 2, and folds >= 1")
    test_start = 7 * n // 10
    meta_size = 15 * n // 100
    validation_size = n // 10
    test_size = n - test_start
    if min(meta_size, validation_size, test_size // folds) < 2:
        raise ValueError("Insufficient dates: every meta, validation, and test block needs at least two dates")
    first_train_end = test_start - validation_size - meta_size - 1
    if first_train_end - lookback + 1 < 2:
        raise ValueError("Insufficient dates for base training before meta, validation, and test")

    quotient, remainder = divmod(test_size, folds)
    plans = []
    for i in range(folds):
        length = quotient + (i < remainder)
        test_end = test_start + length - 1
        validation_start = test_start - validation_size
        meta_start = validation_start - meta_size
        plans.append({
            "train": (int(lookback), int(meta_start - 1)),
            "meta": (int(meta_start), int(validation_start - 1)),
            "validation": (int(validation_start), int(test_start - 1)),
            "test": (int(test_start), int(test_end)),
        })
        test_start = test_end + 1
    return plans


def run_walk_forward(data_path, output, config=None, folds: int = 3, resume=False):
    """Train a fresh full experiment in each expanding fold and save summaries.

    Each seed retains its own validation-selected out-of-sample result per fold.
    Fold means are descriptive, not independent-sample significance estimates.
    We do not splice reset portfolios into a purported continuous equity curve.
    """
    config = config or ExperimentConfig()
    data_path = Path(data_path).resolve()
    panel = load_panel(data_path)
    plans = expanding_splits(len(panel.dates), config.lookback, folds)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not resume)):
        raise FileExistsError(f"Output is not empty: {output}. Use a new walk-forward folder.")
    output.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for i, split in enumerate(plans, start=1):
        directory = output / f"fold_{i:02d}"
        extra = {"resume": True} if resume else {}
        result = run_experiment(data_path, directory, config, split_indices=split, **extra)
        test_start, test_end = split["test"]
        fold_results.append({
            "fold": i,
            "directory": str(directory),
            "splits": {name: list(interval) for name, interval in split.items()},
            "test_first_decision": panel.dates[test_start].isoformat(),
            "test_first_return": panel.dates[test_start + 1].isoformat(),
            "test_last_valuation": panel.dates[test_end].isoformat(),
            "results": result,
        })

    seed_aggregates = {}
    for seed in config.seeds:
        selected = [fold["results"]["seeds"][str(seed)] for fold in fold_results]
        returns = [float(item["selected_test_metrics"]["total_return"]) for item in selected]
        seed_aggregates[str(seed)] = {
            "selected_strategies": [item["selected_strategy"] for item in selected],
            "fold_returns": returns,
            "mean_fold_return": float(np.mean(returns)),
            "std_fold_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
        }
    summary = {
        "method": "expanding_walk_forward",
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "config": asdict(config),
        "fold_count": len(plans),
        "folds": fold_results,
        "test_returns_overlap": False,
        "seed_aggregates": seed_aggregates,
        "interpretation": (
            "Each fold trains fresh models using only its prior base/meta periods; "
            "validation selects the strategy before that fold's test. Test state dates "
            "and test return dates are disjoint across folds for each seed. Later folds "
            "may use earlier test observations after they become historical data. "
            "Each fold resets to cash and omits the boundary-crossing return. Means "
            "and standard deviations describe fold returns, not a continuously held "
            "portfolio or independent-market confidence intervals."
        ),
    }
    # Normalize tuples from dataclass fields so the returned object matches JSON.
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "walkforward_summary.json").write_text(serialized, encoding="utf-8")
    return json.loads(serialized)
