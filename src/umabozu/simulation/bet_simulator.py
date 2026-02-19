import logging
import sys
from typing import Any

import polars as pl

from umabozu.data_preprocessing.data_loader import load_data_from_db  # TID252
from umabozu.data_preprocessing.feature_engineer import generate_features
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.models.predictor import load_model, predict_probabilities
from umabozu.simulation.odds_calculator import calculate_expected_value

logger = logging.getLogger(__name__)


def simulate_betting(
    df_with_expected_value: pl.DataFrame,
    ev_threshold: float = 1.0,
    min_probability: float = 0.0,
) -> pl.DataFrame:
    """
    Simulates betting by selecting horses whose expected value exceeds a
    given threshold. Also allows filtering by a minimum win probability to
    avoid noisy longshots.
    """
    if df_with_expected_value.is_empty():
        return pl.DataFrame()

    # Filter for positive expected value (EV > threshold) and minimum probability
    selected_bets = df_with_expected_value.filter(
        (pl.col("expected_value") > ev_threshold)
        & (pl.col("win_probability") >= min_probability)
    )

    # For actual results, we need the original rank
    actual_results = df_with_expected_value.select(["race_id", "horse_id", "rank"])

    # Join selected bets with actual results to determine wins
    return selected_bets.join(
        actual_results.select(["race_id", "horse_id", "rank"]),
        on=["race_id", "horse_id"],
        how="left",
    )


def calculate_payout(simulated_bets: pl.DataFrame) -> tuple[float, float, float]:
    """
    Calculates total investment, total payout, and recovery rate from simulated bets.
    """
    if simulated_bets.is_empty():
        return 0.0, 0.0, 0.0

    total_investment = len(simulated_bets) * 100  # Assume 100 yen per bet
    total_payout = 0.0

    for row in simulated_bets.iter_rows(named=True):
        if row["rank"] == 1:
            total_payout += row["odds_win"] * 100

    recovery_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    return total_investment, total_payout, recovery_rate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Test Betting Simulation Pipeline ---
    # 1. Load Data
    logger.info("Loading data for betting simulation test...")
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
    logger.info("Training a model for betting simulation test...")
    if "rank" not in train_df_processed.columns:
        logger.error(
            "Error: 'rank' column not found in featured_df. Cannot train model."
        )
        sys.exit(1)

    from umabozu.data_preprocessing.feature_engineer import get_feature_cols
    from umabozu.models.trainer import train_model

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

    # --- Task: Simulate Betting and Calculate Payout ---
    logger.info("\nSimulating betting...")
    # Using a high EV threshold for demonstration,
    # as mock data might not produce high EV
    selected_bets = simulate_betting(df_with_expected_value, ev_threshold=0.5)

    investment, payout, recovery_rate = calculate_payout(selected_bets)

    logger.info("\n--- Betting Simulation Results ---")
    logger.info(f"Total Bets: {len(selected_bets)}")
    logger.info(f"Total Investment: {investment:.2f} JPY")
    logger.info(f"Total Payout: {payout:.2f} JPY")
    logger.info(f"回収率: {recovery_rate:.2f}%")
