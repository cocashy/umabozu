import argparse
import logging
import os
import sys
from datetime import date, timedelta

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from umabozu.app import UmabozuApp

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

    args = parser.parse_args()
    app = UmabozuApp()

    if args.command == "update":
        if (args.days_back and (args.start_date or args.end_date)) or (
            args.start_date and not args.end_date
        ) or (args.end_date and not args.start_date):
            logger.error(
                "Invalid arguments for update. Use either --days-back OR "
                "--start-date and --end-date together."
            )
            sys.exit(1)
        app.update_data(
            days_back=args.days_back,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.command == "train":
        app.train_latest_model()
    elif args.command == "predict":
        app.predict_upcoming(target_date=args.target_date)
    elif args.command == "forecast":
        app.forecast_race(race_id=args.race_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
