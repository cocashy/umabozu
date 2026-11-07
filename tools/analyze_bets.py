import logging
import os
import sys
from typing import Any

import joblib
import polars as pl
from sklearn.base import BaseEstimator

from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.models.predictor import attach_win_probability, predict_probabilities
from umabozu.simulation.bet_simulator import (
    calculate_payout,
    select_bets_by_probability,
)

# Set up logging
logger = logging.getLogger(__name__)


def analyze_model_behavior(
    model: BaseEstimator,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    model_name: str = "Model",
):
    """Analyzes the betting behavior and performance of a given model."""
    logger.info(f"\n--- Analyzing {model_name} Behavior ---")

    # Predict probabilities
    probas_df = predict_probabilities(model, test_df, feature_cols)

    predictions = attach_win_probability(test_df, probas_df)

    # Use the same cutoff as live forecasts; odds are not a selection input.
    min_probability = float(os.getenv("UMABOZU_MIN_WIN_PROBABILITY", "0.13"))
    simulated_bets = select_bets_by_probability(
        predictions, min_probability=min_probability, max_bets_per_race=1
    )

    investment, payout, roi = calculate_payout(simulated_bets)

    logger.info(f"Total Bets: {len(simulated_bets)}")
    logger.info(f"Total Investment: {investment:.2f} JPY")
    logger.info(f"Total Payout: {payout:.2f} JPY")
    logger.info(f"ROI: {roi:.2f}%")

    if not simulated_bets.is_empty():
        # Analyze types of horses bet on
        avg_odds = simulated_bets["odds_win"].mean()
        avg_popularity = simulated_bets["popularity"].mean()
        logger.info(f"Average Odds of Bets: {avg_odds:.2f}")
        logger.info(f"Average Popularity of Bets: {avg_popularity:.2f}")
    else:
        logger.info("No bets placed for this model with current thresholds.")

    logger.info(f"--- {model_name} Analysis Complete ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    full_df = load_data_from_db()
    full_df = clean_data(full_df)
    full_df = generate_features(full_df)

    if full_df.is_empty():
        logger.error("No data available for analysis. Exiting.")
        sys.exit(1)

    # Split into train and test
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.head(split_idx)
    test_df = full_df.tail(len(full_df) - split_idx)

    # Get feature columns
    feature_cols = [col for col in get_feature_cols() if col in full_df.columns]

    # Load models
    default_model_path = "trained_models/lgbm_model.pkl"
    # Assume this exists
    optimized_model_path = "trained_models/lgbm_model_optimized.pkl"

    if os.path.exists(default_model_path):
        default_model = joblib.load(default_model_path)
        analyze_model_behavior(default_model, test_df, feature_cols, "DEFAULT MODEL")
    else:
        logger.warning(
            f"Default model not found at {default_model_path}. Skipping analysis."
        )

    if os.path.exists(optimized_model_path):
        optimized_model = joblib.load(optimized_model_path)
        # Line break for E501
        analyze_model_behavior(
            optimized_model, test_df, feature_cols, "OPTIMIZED MODEL"
        )
    else:
        logger.warning(
            f"Optimized model not found at {optimized_model_path}. Skipping analysis."
        )
