import logging
import sys

import joblib
import polars as pl
from sklearn.base import BaseEstimator

from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import generate_features
from umabozu.data_preprocessing.preprocessor import clean_data

# Set up logging
logger = logging.getLogger(__name__)


def predict_probabilities(
    model: BaseEstimator, df: pl.DataFrame, feature_cols: list[str]
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
    probabilities = model.predict_proba(x_pred)

    return pl.DataFrame(
        probabilities,
        schema={f"proba_{i}": pl.Float64 for i in range(probabilities.shape[1])},
    )


def load_model(model_path: str) -> BaseEstimator:
    """Loads a trained model from the specified path."""
    return joblib.load(model_path)


def save_model(model: BaseEstimator, model_path: str):
    """Saves a trained model to the specified path."""
    joblib.dump(model, model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Test Prediction Pipeline ---
    # 1. Load Data
    logger.info("Loading data for prediction test...")
    full_df = load_data_from_db()

    # Create a dummy 'rank' column if it doesn't exist for testing purposes
    if "rank" not in full_df.columns:
        full_df = full_df.with_columns(pl.lit(0).alias("rank"))

    # For testing, we'll split off a small prediction set from the end
    # In a real scenario, this would be new, unseen data.
    split_idx = int(len(full_df) * 0.9)
    train_df = full_df.head(split_idx)
    predict_df = full_df.tail(len(full_df) - split_idx)

    # 2. Clean and Feature Engineer
    logger.info("Cleaning and feature engineering training data...")
    train_df_processed = clean_data(train_df)
    train_df_processed = generate_features(train_df_processed)

    logger.info("Cleaning and feature engineering prediction data...")
    predict_df_processed = clean_data(predict_df)
    predict_df_processed = generate_features(predict_df_processed)

    # 3. Train a dummy model (should be replaced by actual trained model)
    logger.info("Training a dummy model for prediction test...")
    if "rank" not in train_df_processed.columns:
        logger.error(
            "Error: 'rank' column not found in training data. Cannot train model."
        )
        sys.exit(1)

    # Get feature columns from the processed training data
    from umabozu.data_preprocessing.feature_engineer import get_feature_cols

    feature_cols = [
        col for col in get_feature_cols() if col in train_df_processed.columns
    ]
    x_train = train_df_processed.select(feature_cols).to_numpy()
    y_train = train_df_processed.select("rank").to_numpy().flatten()

    # Use a simple LightGBM model for demonstration
    import lightgbm as lgb

    dummy_model = lgb.LGBMClassifier(random_state=42)
    dummy_model.fit(x_train, y_train)

    # 4. Predict
    logger.info("Predicting probabilities...")
    probas_df = predict_probabilities(dummy_model, predict_df_processed, feature_cols)

    logger.info("\n--- Prediction Results Head ---")
    logger.info(probas_df.head())
    logger.info(f"\nPrediction Results shape: {probas_df.shape}")
