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

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict for historical races in DB for a specific date"
    )
    predict_parser.add_argument(
        "target_date",
        type=str,
        nargs="?",
        default=date.today().strftime("%Y-%m-%d"),
        help="Target date for prediction (YYYY-MM-DD). Defaults to today.",
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
        app.train_latest_model()
    elif args.command == "predict":
        app.predict_upcoming(target_date=args.target_date)
    elif args.command == "forecast":
        app.forecast_race(race_id=args.race_id)
    elif args.command == "status":
        app.show_db_status(local=args.local)


if __name__ == "__main__":
    main()
