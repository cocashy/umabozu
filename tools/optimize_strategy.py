import logging
import os
import sys
from typing import Any

import joblib
import optuna
import polars as pl
from sklearn.base import BaseEstimator

from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.models.predictor import predict_probabilities
from umabozu.simulation.bet_simulator import calculate_payout, simulate_betting
from umabozu.simulation.odds_calculator import calculate_expected_value

# Set up logging
logger = logging.getLogger(__name__)


def optimize_betting_strategy(
    model: BaseEstimator, df: pl.DataFrame, feature_cols: list[str], n_trials: int = 100
) -> tuple[float, float]:
    """
    Optimizes betting strategy thresholds (EV and min_probability)
    using Optuna to maximize ROI.
    """
    logger.info("Starting Optuna optimization for betting strategy...")

    probas_df = predict_probabilities(model, df, feature_cols)
    df_with_ev = calculate_expected_value(probas_df, df)

    def objective(trial):
        ev_threshold = trial.suggest_float("ev_threshold", 1.0, 3.0)
        min_probability = trial.suggest_float("min_probability", 0.0, 0.5)

        simulated_bets = simulate_betting(
            df_with_ev, ev_threshold=ev_threshold, min_probability=min_probability
        )
        _, _, roi = calculate_payout(simulated_bets)
        return roi

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna optimization for betting strategy finished.")
    logger.info(f"Best trial ROI: {study.best_trial.value:.2f}%")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params["ev_threshold"], study.best_params["min_probability"]


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

    # Load the trained model (assuming it's already trained and saved)
    model_path = "trained_models/lgbm_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train a model first.")
        sys.exit(1)
    model = joblib.load(model_path)

    # Get feature columns
    feature_cols = [col for col in get_feature_cols() if col in full_df.columns]

    # Optimize betting strategy on the test set
    best_ev, best_min_prob = optimize_betting_strategy(
        model, test_df, feature_cols, n_trials=50
    )

    logger.info(
        f"\nOptimal Betting Strategy: EV Threshold = {best_ev:.2f},"
        f" Min Probability = {best_min_prob:.2f}"
    )

    # Verify optimal strategy performance
    probas_df = predict_probabilities(model, test_df, feature_cols)
    df_with_ev = calculate_expected_value(probas_df, test_df)
    final_bets = simulate_betting(
        df_with_ev, ev_threshold=best_ev, min_probability=best_min_prob
    )
    investment, payout, roi = calculate_payout(final_bets)

    logger.info("\n--- Final Strategy Performance on Test Set ---")
    logger.info(f"Total Bets: {len(final_bets)}")
    logger.info(f"Total Investment: {investment:.2f} JPY")
    logger.info(f"Total Payout: {payout:.2f} JPY")
    logger.info(f"ROI: {roi:.2f}%")
