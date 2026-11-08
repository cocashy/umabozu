import logging
import os
import sys
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

# Set up logging
logger = logging.getLogger(__name__)


def train_model(df: pl.DataFrame, optimize: bool = True, n_trials: int = 50) -> Any:
    """Train a LightGBM model to estimate each runner's win probability."""
    if "race_date" in df.columns:
        sort_cols = ["race_date"]
        sort_cols.extend(
            col for col in ["race_id", "horse_number"] if col in df.columns
        )
        df = df.sort(sort_cols, maintain_order=True)

    # Class 0 is a winner; class 1 is every other finisher. Keeping the win
    # event as the direct target aligns optimization with the forecast output.
    df = df.with_columns(
        pl.when(pl.col("rank") == 1)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("target")
    )

    # Define features and target
    from umabozu.data_preprocessing.feature_engineer import (
        FEATURE_SCHEMA_VERSION,
        get_feature_cols,
    )

    feature_cols = [col for col in get_feature_cols() if col in df.columns]
    x_data = df.select(feature_cols).to_numpy()
    x_data = pd.DataFrame(x_data, columns=feature_cols)
    y_data = df.select("target").to_numpy().flatten()

    if optimize:
        logger.info("Starting Optuna optimization for LightGBM...")

        if "race_date" not in df.columns:
            raise ValueError("Time-series optimization requires a race_date column.")
        if df.get_column("race_date").null_count():
            raise ValueError("Time-series optimization requires non-null race dates.")

        ordered_dates = (
            df.select("race_date").unique().sort("race_date").to_series().to_list()
        )
        row_dates = df.get_column("race_date").to_list()
        date_splits = []
        # Split the ordered date list so all runners and races on a date remain
        # in the same fold.
        for train_date_idx, val_date_idx in TimeSeriesSplit(n_splits=5).split(
            ordered_dates
        ):
            train_dates = {ordered_dates[i] for i in train_date_idx}
            val_dates = {ordered_dates[i] for i in val_date_idx}
            train_idx = [i for i, value in enumerate(row_dates) if value in train_dates]
            val_idx = [i for i, value in enumerate(row_dates) if value in val_dates]
            date_splits.append((train_idx, val_idx))

        def objective(trial):
            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "boosting_type": "gbdt",
            }

            logloss_scores = []

            for train_idx, val_idx in date_splits:
                x_train, x_val = x_data.iloc[train_idx], x_data.iloc[val_idx]
                y_train, y_val = y_data[train_idx], y_data[val_idx]

                from typing import Any

                model = lgb.LGBMClassifier(**(param))  # type: ignore
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                preds = model.predict_proba(x_val)
                logloss_scores.append(log_loss(y_val, preds, labels=[0, 1]))

            return sum(logloss_scores) / len(logloss_scores)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info("Optuna optimization finished.")
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_params}")

        best_params = study.best_params
        model = lgb.LGBMClassifier(  # type: ignore
            objective="binary",
            metric="binary_logloss",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            boosting_type="gbdt",
            **best_params,
        )
    else:
        logger.info("Training LightGBM model with default parameters...")
        model = lgb.LGBMClassifier(  # type: ignore
            objective="binary",
            metric="binary_logloss",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            boosting_type="gbdt",
        )

    model.fit(x_data, y_data)
    model._umabozu_feature_schema_version = FEATURE_SCHEMA_VERSION
    model._umabozu_feature_cols = feature_cols
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load data for testing
    # Add project root to sys.path for relative imports
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )
    from umabozu.data_preprocessing.data_loader import load_data_from_db
    from umabozu.data_preprocessing.feature_engineer import generate_features
    from umabozu.data_preprocessing.preprocessor import clean_data

    logger.info("Loading data for training test...")
    df = load_data_from_db()
    df = clean_data(df)
    df = generate_features(df)
    df = df.sort("race_date")

    if df.is_empty():
        logger.error("No data available for training. Exiting.")
        sys.exit(1)

    logger.info(f"Data loaded and preprocessed. Shape: {df.shape}")

    # Train without optimization
    logger.info("\n--- Training without Optuna Optimization ---")
    model_default = train_model(df, optimize=False)
    logger.info("Default model training complete.")

    # Train with optimization
    logger.info("\n--- Training with Optuna Optimization ---")
    model_optimized = train_model(df, optimize=True, n_trials=10)
    logger.info("Optimized model training complete.")
