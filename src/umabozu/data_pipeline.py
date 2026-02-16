import datetime
import logging

import polars as pl

from umabozu.data_collection.batch_collector import (  # TID252
    collect_profiles,
    collect_races,
)
from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import generate_features
from umabozu.data_preprocessing.preprocessor import clean_data

logger = logging.getLogger(__name__)


def run_data_pipeline(start_date, end_date):
    """
    Runs the full data pipeline: collects raw data, parses, saves to DB,
    loads, cleans, and engineers features.
    """
    logger.info("Starting data collection...")
    collect_races(start_date, end_date)
    collect_profiles()
    logger.info("Data collection complete.")

    logger.info("Loading data from database...")
    df = load_data_from_db()
    logger.info(f"Raw data loaded. Shape: {df.shape}")

    logger.info("Cleaning data...")
    df = clean_data(df)
    logger.info(f"Data cleaned. Shape: {df.shape}")

    logger.info("Generating features...")
    df = generate_features(df)
    logger.info(f"Features generated. Shape: {df.shape}")

    logger.info("Data pipeline finished.")
    return df


def display_pipeline_overview(df: pl.DataFrame):
    """
    Displays an overview of the processed DataFrame,
    highlighting key columns and potential issues.
    """
    logger.info("\n--- Data Pipeline Overview ---")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info("Columns and their data types:")
    for col, dtype in df.schema.items():
        logger.info(f"- {col}: {dtype}")

    logger.info("\nMissing values per column:")
    for col in df.columns:
        # Need to get owner_id, trainer_id from horse profile page later
        # For now, these are None or extracted from parsing.
        # The parser doesn't extract owner/trainer/producer/breeding_farm/jockey
        # birth/debut dates yet.
        # This part will need further refinement once horse profile scraping/parsing
        # is implemented.
        # Dummy data for now, actual extraction will be done later
        missing_count = df.select(pl.col(col).is_null().sum()).item()
        if missing_count > 0:
            logger.info(f"- {col}: {missing_count} missing values")

    logger.info("\nFirst 5 rows of the processed DataFrame:")
    logger.info(df.head())
    logger.info("\n--- Overview Complete ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import date

    today = date.today()
    # Example: Collect data for the last 7 days
    seven_days_ago = today - datetime.timedelta(days=7)
    processed_df = run_data_pipeline(seven_days_ago, today)
    display_pipeline_overview(processed_df)
