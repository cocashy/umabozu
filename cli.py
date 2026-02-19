import argparse
import logging
import os
import sys
from datetime import date, timedelta

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Umabozu CLI for horse racing prediction."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update race and profile data")
    update_parser.add_argument(
        "--days-back",
        type=int,
        help="Number of days back to update data from today. "
        "Mutually exclusive with --start-date/--end-date.",
    )
    update_parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data update (YYYY-MM-DD). Use with --end-date.",
    )
    update_parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data update (YYYY-MM-DD). Use with --start-date.",
    )
    update_parser.add_argument(
        "--retry-tmp-only",
        action="store_true",
        help="Retry incomplete HTML downloads and exit.",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the prediction model")
    train_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use Optuna hyperparameter optimization (slower but better accuracy).",
    )
    train_parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20). Only used with --optimize.",
    )
    train_parser.add_argument(
        "--test-ratio",
        type=float,
        help="Ratio of data to use for testing (e.g., 0.2 for 20%%). "
        "If specified, data will be split and test performance logged.",
    )

    # Forecast command
    forecast_parser = subparsers.add_parser(
        "forecast", help="Forecast for a live upcoming race using its race_id"
    )
    forecast_parser.add_argument(
        "race_id", type=str, help="Race ID for the upcoming race to forecast"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show database status and date range"
    )
    status_parser.add_argument(
        "--local",
        action="store_true",
        help="Check for gaps every day (local racing). Default checks weekends.",
    )

    # Simulate command (backtesting)
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run backtesting simulation with train/test split and ROI calculation",
    )
    simulate_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (default: 0.2 = 20%%).",
    )
    simulate_parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models/lgbm_model.pkl",
        help=(
            "Ignored. simulate always retrains to avoid leakage. "
            "Kept for backward compatibility."
        ),
    )
    simulate_parser.add_argument(
        "--retrain",
        action="store_true",
        help="Deprecated. simulate always retrains to avoid leakage.",
    )
    simulate_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use Optuna optimization. Only used with --retrain.",
    )
    simulate_parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help=(
            "Number of Optuna trials (default: 20). "
            "Only used with --retrain --optimize."
        ),
    )

    args = parser.parse_args()

    # Lazy import: only import UmabozuApp when a command is actually being executed
    if args.command:
        from umabozu.app import UmabozuApp

        app = UmabozuApp()
    else:
        parser.print_help()
        return

    if args.command == "update":
        if (
            (args.days_back and (args.start_date or args.end_date))
            or (args.start_date and not args.end_date)
            or (args.end_date and not args.start_date)
        ):
            logger.error(
                "Invalid arguments for update. Use either --days-back OR "
                "--start-date and --end-date together."
            )
            sys.exit(1)
        app.update_data(
            days_back=args.days_back,
            start_date=args.start_date,
            end_date=args.end_date,
            retry_tmp_only=args.retry_tmp_only,
        )
    elif args.command == "train":
        app.train_latest_model(
            optimize=args.optimize,
            n_trials=args.n_trials,
            test_ratio=args.test_ratio,
        )
    elif args.command == "forecast":
        app.forecast_race(race_id=args.race_id)
    elif args.command == "status":
        app.show_db_status(local=args.local)
    elif args.command == "simulate":
        app.simulate_backtest(
            test_ratio=args.test_ratio,
            model_path=args.model_path,
            retrain=args.retrain,
            optimize=args.optimize,
            n_trials=args.n_trials,
        )


if __name__ == "__main__":
    main()
