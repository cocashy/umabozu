import logging
import os
import sys

import lightgbm as lgb
import optuna
import polars as pl
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Set up logging
logger = logging.getLogger(__name__)


def train_model(
    df: pl.DataFrame, optimize: bool = True, n_trials: int = 50
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM multi-class classification model,
    optionally with Optuna optimization.
    """
    # 1. Create Target (0: 1st, 1: 2nd, 3: 4th+)
    df = df.with_columns(
        (pl.col("rank") - 1).alias("target")
    )  # 1st becomes 0, 2nd becomes 1, etc.
    df = df.with_columns(
        pl.when(pl.col("target") >= 3)
        .then(pl.lit(3))
        .otherwise(pl.col("target"))
        .alias("target")
    )  # Map 4th+ to 3

    # Define features and target
    from umabozu.data_preprocessing.feature_engineer import get_feature_cols

    feature_cols = [col for col in get_feature_cols() if col in df.columns]
    x_data = df.select(feature_cols).to_numpy()
    y_data = df.select("target").to_numpy().flatten()

    if optimize:
        logger.info("Starting Optuna optimization for LightGBM...")

        def objective(trial):
            param = {
                "objective": "multiclass",
                "num_class": 4,  # 0, 1, 2, 3
                "metric": "multi_logloss",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "boosting_type": "gbdt",
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            logloss_scores = []

            for train_idx, val_idx in skf.split(x_data, y_data):
                x_train, x_val = x_data[train_idx], x_data[val_idx]
                y_train, y_val = y_data[train_idx], y_data[val_idx]

                model = lgb.LGBMClassifier(**param)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=50,
                    callbacks=[
                        optuna.integration.LightGBMPruningCallback(
                            trial, "multi_logloss"
                        )
                    ],
                )
                preds = model.predict_proba(x_val)
                logloss_scores.append(log_loss(y_val, preds))

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
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            metric="multi_logloss",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            boosting_type="gbdt",
            **best_params,
        )
    else:
        logger.info("Training LightGBM model with default parameters...")
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            metric="multi_logloss",
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
