import logging
import sys
from typing import Any

import joblib
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator

# Set up logging
logger = logging.getLogger(__name__)


def predict_probabilities(
    model: Any, df: pl.DataFrame, feature_cols: list[str]
) -> pl.DataFrame:
    """
    Predicts probabilities for each class (rank) using the trained model.
    """
    if df.is_empty():
        return pl.DataFrame(
            {
                "proba_0": [],
                "proba_1": [],
                "proba_2": [],
                "proba_3": [],
            }
        )

    # Ensure all feature columns are present in the DataFrame
    # and handle potential missing columns by filling with 0
    for col in feature_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    x_pred = df.select(feature_cols).to_numpy()
    x_pred = pd.DataFrame(x_pred, columns=feature_cols)
    probabilities = model.predict_proba(x_pred)

    return pl.DataFrame(
        probabilities,
        schema={f"proba_{i}": pl.Float64 for i in range(probabilities.shape[1])},
    )


def attach_win_probability(
    df: pl.DataFrame, probabilities_df: pl.DataFrame
) -> pl.DataFrame:
    """Attach model probabilities and expose the first-place probability.

    The trainer maps first place to target class 0, so ``proba_0`` is the
    predicted win probability. This helper does not read odds.
    """
    if df.height != probabilities_df.height:
        raise ValueError(
            "The prediction row count must match the source DataFrame row count."
        )
    if "proba_0" not in probabilities_df.columns:
        raise ValueError("Prediction output is missing the first-place class proba_0.")

    return df.with_columns(probabilities_df).with_columns(
        pl.col("proba_0").alias("win_probability")
    )


def load_model(model_path: str) -> BaseEstimator:
    """Loads a trained model from the specified path."""
    return joblib.load(model_path)


def save_model(model: BaseEstimator, model_path: str):
    """Saves a trained model to the specified path."""
    joblib.dump(model, model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from umabozu.data_pipeline import prepare_training_data, split_train_test
    from umabozu.data_preprocessing.feature_engineer import get_feature_cols
    from umabozu.models.trainer import train_model

    # --- Test Prediction Pipeline ---
    logger.info("Loading and preparing data for prediction example...")
    full_df = prepare_training_data()
    train_df_processed, predict_df_processed = split_train_test(full_df, 0.1)

    # 3. Train a dummy model (should be replaced by actual trained model)
    logger.info("Training a dummy model for prediction test...")
    if "rank" not in train_df_processed.columns:
        logger.error(
            "Error: 'rank' column not found in training data. Cannot train model."
        )
        sys.exit(1)

    feature_cols = [
        col for col in get_feature_cols() if col in train_df_processed.columns
    ]
    model = train_model(train_df_processed, optimize=False)

    # 4. Predict
    logger.info("Predicting probabilities...")
    probas_df = predict_probabilities(model, predict_df_processed, feature_cols)

    logger.info("\n--- Prediction Results Head ---")
    logger.info(probas_df.head())
    logger.info(f"\nPrediction Results shape: {probas_df.shape}")
