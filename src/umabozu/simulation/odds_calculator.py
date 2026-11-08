import logging
import sys

import polars as pl

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
    from umabozu.data_pipeline import prepare_training_data, split_train_test
    from umabozu.data_preprocessing.feature_engineer import get_feature_cols

    # --- Test EV Calculation Pipeline ---
    logger.info("Loading and preparing data for expected-value calculation...")
    full_df = prepare_training_data()
    train_df_processed, test_df_processed = split_train_test(full_df, 0.1)

    # 3. Train a dummy model (or load a pre-trained one)
    logger.info("Training a model for expected value calculation test...")
    if "rank" not in train_df_processed.columns:
        logger.error(
            "Error: 'rank' column not found in featured_df. Cannot train model."
        )
        sys.exit(1)

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
