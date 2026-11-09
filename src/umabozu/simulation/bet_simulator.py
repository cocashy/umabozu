import logging
import sys
from typing import Any

import polars as pl

from umabozu.config import DEFAULT_MIN_WIN_PROBABILITY
from umabozu.models.predictor import (
    attach_win_probability,
    load_model,
    predict_probabilities,
)

logger = logging.getLogger(__name__)


def select_bets_by_probability(
    predictions: pl.DataFrame,
    min_probability: float = DEFAULT_MIN_WIN_PROBABILITY,
    max_bets_per_race: int = 1,
) -> pl.DataFrame:
    """Select at most the highest-probability horses per race, without odds."""
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be between 0.0 and 1.0.")
    if max_bets_per_race < 1:
        raise ValueError("max_bets_per_race must be at least 1.")
    if predictions.is_empty():
        return predictions

    required_cols = {"race_id", "win_probability"}
    missing_cols = required_cols - set(predictions.columns)
    if missing_cols:
        raise ValueError(
            "Probability selection requires columns: " + ", ".join(sorted(missing_cols))
        )

    eligible = predictions.filter(
        pl.col("win_probability").is_not_null()
        & (pl.col("win_probability") >= min_probability)
    )
    if eligible.is_empty():
        return eligible

    sort_cols = ["race_id", "win_probability"]
    descending = [False, True]
    if "horse_id" in eligible.columns:
        sort_cols.append("horse_id")
        descending.append(False)

    return (
        eligible.sort(sort_cols, descending=descending)
        .group_by("race_id", maintain_order=True)
        .head(max_bets_per_race)
    )


def simulate_betting(
    df_with_expected_value: pl.DataFrame,
    ev_threshold: float = 1.0,
    min_probability: float = 0.0,
) -> pl.DataFrame:
    """
    Legacy EV-based selector retained for compatibility with older analysis.
    Forecasts, current backtests, and strategy tools use
    :func:`select_bets_by_probability` instead.

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
            # Historical final odds are used only to settle already-selected
            # bets, never to decide which horses to select.
            total_payout += row["odds_win"] * 100

    recovery_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    return total_investment, total_payout, recovery_rate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from umabozu.data_pipeline import prepare_training_data, split_train_test
    from umabozu.data_preprocessing.feature_engineer import get_feature_cols
    from umabozu.models.trainer import train_model

    # --- Test Betting Simulation Pipeline ---
    logger.info("Loading and preparing data for betting simulation...")
    full_df = prepare_training_data()
    train_df_processed, test_df_processed = split_train_test(full_df, 0.1)

    # 3. Train a dummy model (or load a pre-trained one)
    logger.info("Training a model for betting simulation test...")
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

    # 5. Attach the win probability and select without using odds.
    predictions = attach_win_probability(test_df_processed, probas_df)

    # --- Task: Simulate Betting and Calculate Payout ---
    logger.info("\nSimulating betting...")
    selected_bets = select_bets_by_probability(
        predictions,
        min_probability=DEFAULT_MIN_WIN_PROBABILITY,
        max_bets_per_race=1,
    )

    investment, payout, recovery_rate = calculate_payout(selected_bets)

    logger.info("\n--- Betting Simulation Results ---")
    logger.info(f"Total Bets: {len(selected_bets)}")
    logger.info(f"Total Investment: {investment:.2f} JPY")
    logger.info(f"Total Payout: {payout:.2f} JPY")
    logger.info(f"回収率: {recovery_rate:.2f}%")
