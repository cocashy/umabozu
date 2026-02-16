import logging
import sys

import polars as pl

from umabozu.data_preprocessing.data_loader import load_data_from_db  # TID252
from umabozu.data_preprocessing.feature_engineer import generate_features
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.models.predictor import predict_probabilities
from umabozu.models.trainer import train_model

logger = logging.getLogger(__name__)


def calculate_expected_value(
    probas_df: pl.DataFrame, original_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculates the expected value for single win bets based on
    predicted probabilities and odds.
    """
    # Ensure all necessary columns are present
    if "odds_win" not in original_df.columns or "proba_0" not in probas_df.columns:
        logger.error("Missing 'odds_win' in original_df or 'proba_0' in probas_df.")
        return pl.DataFrame()

    # Join probabilities back to the original DataFrame
    # Assuming the order is preserved or can be joined by an index/key
    # For now, we'll assume order is preserved (simple concatenation)
    df_with_probas = original_df.with_columns(probas_df)

    # Calculate Expected Value (EV) for a single win bet
    # EV = P(Win) * Odds_Win
    return df_with_probas.with_columns(
        (pl.col("proba_0") * pl.col("odds_win")).alias("expected_value"),
        pl.col("proba_0").alias("win_probability"),  # Rename for clarity
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Test EV Calculation Pipeline ---
    # 1. Load Data
    logger.info("Loading data for expected value calculation test...")
    full_df = load_data_from_db()

    # Create a dummy 'rank' column if it doesn't exist for testing purposes
    if "rank" not in full_df.columns:
        full_df = full_df.with_columns(pl.lit(0).alias("rank"))

    # For testing, we'll split off a small prediction set from the end
    split_idx = int(len(full_df) * 0.9)
    train_df = full_df.head(split_idx)
    test_df = full_df.tail(len(full_df) - split_idx)

    # 2. Clean and Feature Engineer
    logger.info("Cleaning and feature engineering training data...")
    train_df_processed = clean_data(train_df)
    train_df_processed = generate_features(train_df_processed)

    logger.info("Cleaning and feature engineering test data...")
    test_df_processed = clean_data(test_df)
    test_df_processed = generate_features(test_df_processed)

    # 3. Train a dummy model (or load a pre-trained one)
    logger.info("Training a model for expected value calculation test...")
    if "rank" not in train_df_processed.columns:
        logger.error(
            "Error: 'rank' column not found in featured_df. Cannot train model."
        )
        sys.exit(1)

    from umabozu.data_preprocessing.feature_engineer import get_feature_cols

    feature_cols = [
        col for col in get_feature_cols() if col in train_df_processed.columns
    ]
    model = train_model(train_df_processed, optimize=False)

    # 4. Predict Probabilities
    logger.info("Predicting probabilities for test data...")
    probas_df = predict_probabilities(model, test_df_processed, feature_cols)

    # 5. Calculate Expected Value
    logger.info("Calculating expected values...")
    df_with_expected_value = calculate_expected_value(probas_df, test_df_processed)

    logger.info("\n--- DataFrame with Expected Values Head ---")
    logger.info(df_with_expected_value.head())
    logger.info(
        f"\nDataFrame with Expected Values shape: {df_with_expected_value.shape}"
    )

    # Verify some expected values
    if not df_with_expected_value.is_empty():
        sample = df_with_expected_value.sample(
            n=min(5, len(df_with_expected_value)), seed=42
        )
        logger.info("\nSample of calculated Expected Values:")
        logger.info(sample.select(["odds_win", "proba_0", "expected_value"]))
