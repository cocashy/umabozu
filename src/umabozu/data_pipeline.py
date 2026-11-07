import datetime
import logging

import polars as pl

from umabozu.data_collection.batch_collector import (  # TID252
    collect_profiles,
    collect_races,
)
from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.database.crud import get_training_data

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


def prepare_training_data(use_db_optimized: bool = True) -> pl.DataFrame:
    """
    Load, clean, and generate features for training data.
    This is a common pipeline used across multiple commands.

    Args:
        use_db_optimized: If True, use get_training_data() which may have
                         optimized queries. If False, use load_data_from_db().

    Returns:
        Processed DataFrame ready for training or prediction.
    """
    logger.info("Loading data from database...")
    df = get_training_data() if use_db_optimized else load_data_from_db()

    if df.is_empty():
        logger.warning("No data loaded from database.")
        return df

    logger.info(f"Raw data loaded. Shape: {df.shape}")

    logger.info("Cleaning data...")
    df = clean_data(df)
    logger.info(f"Data cleaned. Shape: {df.shape}")

    logger.info("Generating features...")
    df = generate_features(df)
    logger.info(f"Features generated. Shape: {df.shape}")

    return df


def split_train_test(
    df: pl.DataFrame, test_ratio: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split DataFrame by whole races in chronological order.
    Every horse from one race stays in the same split.

    Args:
        df: Input DataFrame sorted or unsorted by race_date
        test_ratio: Proportion of data to use for testing (default: 0.2)

    Returns:
        Tuple of (train_df, test_df)
    """
    if df.is_empty():
        return df, df

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0.")
    required_cols = {"race_id", "race_date"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Chronological race split requires columns: "
            + ", ".join(sorted(missing_cols))
        )

    df = df.sort(["race_date", "race_id"])
    races = df.select(["race_id", "race_date"]).unique().sort(
        ["race_date", "race_id"]
    )
    split_idx = int(len(races) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(races):
        raise ValueError("Not enough races to create non-empty train and test sets.")

    train_races = races.head(split_idx).select("race_id")
    test_races = races.tail(len(races) - split_idx).select("race_id")
    train_df = df.join(train_races, on="race_id", how="semi").sort(
        ["race_date", "race_id"]
    )
    test_df = df.join(test_races, on="race_id", how="semi").sort(
        ["race_date", "race_id"]
    )

    logger.info(
        f"Data split: {train_df.select('race_id').n_unique()} training races, "
        f"{test_df.select('race_id').n_unique()} test races "
        f"({len(train_df)} and {len(test_df)} samples)"
    )

    return train_df, test_df


def get_valid_feature_cols(df: pl.DataFrame) -> list[str]:
    """
    Get the list of feature columns that actually exist in the DataFrame.

    Args:
        df: DataFrame to check for feature columns

    Returns:
        List of feature column names that exist in df
    """
    return [col for col in get_feature_cols() if col in df.columns]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import date

    today = date.today()
    # Example: Collect data for the last 7 days
    seven_days_ago = today - datetime.timedelta(days=7)
    processed_df = run_data_pipeline(seven_days_ago, today)
    display_pipeline_overview(processed_df)
