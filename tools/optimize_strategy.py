import logging
import sys
from typing import Any

import optuna
import polars as pl

from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.models.predictor import attach_win_probability, predict_probabilities
from umabozu.models.trainer import train_model
from umabozu.simulation.bet_simulator import (
    calculate_payout,
    select_bets_by_probability,
)

logger = logging.getLogger(__name__)


def optimize_betting_strategy(
    model: Any,
    validation_df: pl.DataFrame,
    feature_cols: list[str],
    n_trials: int = 100,
) -> tuple[float, float]:
    """Tune only a minimum win-probability threshold on validation data.

    Final odds are used by ``calculate_payout`` only to score the completed
    historical bets. They are never passed to the selector.
    """
    logger.info("Tuning the minimum win-probability threshold...")

    probabilities = predict_probabilities(model, validation_df, feature_cols)
    predictions = attach_win_probability(validation_df, probabilities)

    def objective(trial):
        min_probability = trial.suggest_float("min_probability", 0.0, 0.5)
        selected_bets = select_bets_by_probability(
            predictions,
            min_probability=min_probability,
            max_bets_per_race=1,
        )
        if selected_bets.is_empty():
            return -100.0
        _, _, roi = calculate_payout(selected_bets)
        return roi

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_threshold = float(study.best_params["min_probability"])
    best_validation_roi = float(study.best_trial.value)
    logger.info(
        "Threshold tuning finished: minimum win probability %.3f; "
        "validation ROI %.2f%%",
        best_threshold,
        best_validation_roi,
    )
    return best_threshold, best_validation_roi


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading and preprocessing data...")
    full_df = load_data_from_db()
    full_df = generate_features(clean_data(full_df))
    if full_df.is_empty() or "race_date" not in full_df.columns:
        logger.error("No usable race data available. Exiting.")
        sys.exit(1)

    full_df = full_df.sort("race_date")
    race_order = full_df.select(["race_id", "race_date"]).unique().sort(
        ["race_date", "race_id"]
    )
    train_end = int(len(race_order) * 0.6)
    validation_end = int(len(race_order) * 0.8)
    if (
        train_end == 0
        or validation_end <= train_end
        or validation_end >= len(race_order)
    ):
        logger.error("Not enough data for chronological train/validation/test splits.")
        sys.exit(1)

    train_race_ids = race_order.head(train_end).select("race_id")
    validation_race_ids = race_order.slice(
        train_end, validation_end - train_end
    ).select("race_id")
    test_race_ids = race_order.tail(len(race_order) - validation_end).select("race_id")
    train_df = full_df.join(train_race_ids, on="race_id", how="semi")
    validation_df = full_df.join(validation_race_ids, on="race_id", how="semi")
    test_df = full_df.join(test_race_ids, on="race_id", how="semi")

    logger.info("Training the model on the earliest 60% of races...")
    model = train_model(train_df, optimize=False)
    feature_cols = [col for col in get_feature_cols() if col in train_df.columns]

    best_min_probability, validation_roi = optimize_betting_strategy(
        model, validation_df, feature_cols, n_trials=50
    )

    test_probabilities = predict_probabilities(model, test_df, feature_cols)
    test_predictions = attach_win_probability(test_df, test_probabilities)
    test_bets = select_bets_by_probability(
        test_predictions,
        min_probability=best_min_probability,
        max_bets_per_race=1,
    )
    investment, payout, test_roi = calculate_payout(test_bets)

    logger.info("\n--- Probability-only strategy on held-out test data ---")
    logger.info("Minimum win probability: %.3f", best_min_probability)
    logger.info(
        "To use this cutoff for live forecasts, set "
        "UMABOZU_MIN_WIN_PROBABILITY=%.3f",
        best_min_probability,
    )
    logger.info("Validation ROI: %.2f%%", validation_roi)
    logger.info("Bets placed: %s", len(test_bets))
    logger.info("Total investment: %.2f JPY", investment)
    logger.info("Total payout: %.2f JPY", payout)
    logger.info("Test ROI: %.2f%%", test_roi)
