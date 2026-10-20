"""Command-line entry points. All defaults describe research, not live trading."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="A-share ETF portfolio research: FinRL multi-algorithm / DQN")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch and fingerprint adjusted ETF history")
    download.add_argument("--symbols", nargs="+", default=["510300", "510500", "159915", "510050"])
    download.add_argument("--start", default="20180101")
    download.add_argument("--end", default="20261016", help="Last completed session; explicit fixed date for reproducibility")
    download.add_argument("--output", default="data/processed/ashare_etf.csv")
    download.add_argument("--provider", choices=["eastmoney", "tencent"], default="eastmoney")
    for command in ("train", "demo", "walk-forward"):
        help_text = {"train": "Full market experiment", "demo": "Explicit synthetic smoke experiment", "walk-forward": "Expanding training with disjoint market test windows"}[command]
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--output", default=f"runs/{command}")
        sub.add_argument("--steps", type=int, default=5000 if command == "train" else 512)
        sub.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29] if command == "train" else [7])
        sub.add_argument("--lookback", type=int, default=20)
        sub.add_argument("--cost-bps", type=float, default=5.)
        sub.add_argument("--max-weight", type=float, default=.5)
        sub.add_argument("--backend", choices=["finrl", "sb3"], default="finrl")
        sub.add_argument("--encoder", choices=["shared", "mlp"], default="shared")
        sub.add_argument("--algorithms", nargs="+", choices=["td3", "ddpg", "sac", "ppo", "a2c"], default=None,
                         help="Explicit research mode: include every trained RL algorithm in the selector pool")
        sub.add_argument("--risk-levels", nargs="+", type=float, default=[0., 2., 8.],
                         help="TD3 risk levels; use 2 for one TD3 model in a cross-algorithm comparison")
        sub.add_argument("--resume", action="store_true", help="Reuse completed checkpoints only if data/config/splits match")
        if command in ("train", "walk-forward"):
            sub.add_argument("--data", required=True)
            if command == "walk-forward":
                sub.add_argument("--folds", type=int, default=3)
        else:
            sub.add_argument("--periods", type=int, default=800)
    evaluate = commands.add_parser("evaluate", help="Reload checkpoints and reproduce held-out test")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--cost-bps", type=float, default=None, help="Frozen-policy transaction cost stress; no retraining")
    theory = commands.add_parser("theory", help="Analytic DDPO quadratic benchmark; no RL training")
    theory.add_argument("--output", default="runs/theory.json")
    theory.add_argument("--seed", type=int, default=7)
    theory.add_argument("--horizon", type=int, default=5000)
    theory.add_argument("--episodes", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "download":
        from .download import download_etfs
        result = download_etfs(args.symbols, args.start, args.end, args.output, provider=args.provider)
    elif args.command in ("train", "demo", "walk-forward"):
        from .experiment import ExperimentConfig, run_experiment
        config = ExperimentConfig(total_timesteps=args.steps, seeds=tuple(args.seeds), lookback=args.lookback, cost_bps=args.cost_bps, max_weight=args.max_weight, backend=args.backend, shared_encoder=args.encoder == "shared",
                                  algorithms=tuple(args.algorithms) if args.algorithms is not None else ("td3", "ddpg"),
                                  selector_all_algorithms=args.algorithms is not None, risk_levels=tuple(args.risk_levels))
        if args.command == "demo":
            from .data import synthetic_frame
            output = Path(args.output)
            data = output.parent / f"{output.name}_synthetic.csv"
            # Refuse before touching the existing data or run artifacts.
            if args.resume:
                if not data.is_file() or not (output / "manifest.json").is_file():
                    raise ValueError("Demo resume requires its existing data and manifest")
            else:
                if (output.exists() and any(output.iterdir())) or data.exists():
                    raise FileExistsError("Choose a fresh demo output path")
                data.parent.mkdir(parents=True, exist_ok=True)
                synthetic_frame(args.periods, assets=4, seed=args.seeds[0]).to_csv(data, index=False)
            data_kind = "synthetic"
        else:
            data, data_kind = Path(args.data), "market"
        if args.command == "walk-forward":
            from .walkforward import run_walk_forward
            result = run_walk_forward(data, args.output, config, folds=args.folds, resume=args.resume)
            print(f"Summary: {Path(args.output).resolve() / 'walkforward_summary.json'}")
            result = {"fold_count": result["fold_count"], "seed_aggregates": result["seed_aggregates"]}
        else:
            result = run_experiment(data, args.output, config, data_kind=data_kind, resume=args.resume)
            print(f"Report: {Path(args.output).resolve() / 'report.md'}")
            result = {"data_kind": result["data_kind"], "aggregate": result["aggregate"]}
    elif args.command == "evaluate":
        from .experiment import evaluate_saved
        result = evaluate_saved(args.run, args.seed, cost_bps=args.cost_bps)
    else:
        from .theory import evaluate_quadratic_benchmark
        result = evaluate_quadratic_benchmark(seed=args.seed, horizon=args.horizon, episodes=args.episodes)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
