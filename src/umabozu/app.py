import glob
import logging
import os
import sys
from contextlib import suppress
from datetime import date, timedelta

import polars as pl

# TID252: Prefer absolute imports
from umabozu.data_collection.batch_collector import collect_profiles, collect_races
from umabozu.data_collection.html_parser import (
    parse_horse_profile_html,
    parse_jockey_profile_html,
    parse_shutsubahyou_html,
)
from umabozu.data_collection.scraper import (
    fetch_and_save_horse_html,
    fetch_and_save_jockey_html,
    fetch_and_save_race_html,
    fetch_and_save_shutsubahyou_html,
)
from umabozu.data_pipeline import (
    get_valid_feature_cols,
    prepare_training_data,
    split_train_test,
)
from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.database.crud import (
    get_db_connection,
    get_db_status,
    get_training_data,
    insert_horse,
    insert_jockey,
)
from umabozu.models.predictor import load_model, predict_probabilities, save_model
from umabozu.models.trainer import train_model
from umabozu.simulation.bet_simulator import simulate_betting
from umabozu.simulation.odds_calculator import calculate_expected_value

logger = logging.getLogger(__name__)


class UmabozuApp:
    def __init__(self):
        self.ev_threshold = 1.1
        self.min_probability = 0.13

    def show_db_status(self, local=False):
        """Display database statistics and detect missing data points."""
        try:
            status = get_db_status()

            def emit(message=""):
                print(message)

            emit("Database status")
            emit(
                f"Race date range: {status['race_date_min']} to "
                f"{status['race_date_max']}"
            )
            emit(f"Total races: {status['race_count']:,}")
            emit(f"Total results: {status['result_count']:,}")
            emit(f"Total horses: {status['horse_count']:,}")
            emit(f"Total jockeys: {status['jockey_count']:,}")

            if not status["recorded_dates"]:
                return

            recorded_set = set(status["recorded_dates"])
            start_date = date.fromisoformat(status["race_date_min"])
            end_date = date.fromisoformat(status["race_date_max"])

            missing_days = []
            curr = start_date
            while curr <= end_date:
                d_str = curr.isoformat()
                if local:
                    # Check every single day
                    if d_str not in recorded_set:
                        missing_days.append(curr)
                else:
                    # JRA mode: Only check Sat (5) and Sun (6)
                    if curr.weekday() in [5, 6] and d_str not in recorded_set:
                        missing_days.append(curr)
                curr += timedelta(days=1)

            if missing_days:
                mode_str = "Daily (Local)" if local else "Weekend (JRA)"
                emit("")
                emit(f"Missing data ({mode_str} mode):")

                # Group consecutive missing dates into ranges
                ranges = []
                if missing_days:
                    range_start = missing_days[0]
                    for i in range(1, len(missing_days)):
                        # For JRA mode, "consecutive" means next expected weekend day
                        # For simplicity, if diff > 1 in local mode OR diff > 7
                        # in JRA mode, start new range
                        diff = (missing_days[i] - missing_days[i - 1]).days
                        threshold = 1 if local else 7
                        if diff > threshold:
                            ranges.append((range_start, missing_days[i - 1]))
                            range_start = missing_days[i]
                    ranges.append((range_start, missing_days[-1]))

                for s, e in ranges:
                    if s == e:
                        emit(f"- Missing: {s} ({s.strftime('%a')})")
                    else:
                        days_count = (e - s).days + 1
                        emit(f"- Gap: {s} to {e} ({days_count} days)")
            else:
                emit("")
                emit("No data gaps detected in the specified mode.")

            # Check for incomplete file transfers
            temp_files = glob.glob("data/raw/html/**/*.tmp", recursive=True)
            if temp_files:
                emit("")
                emit("Incomplete file transfers:")
                for tmp in temp_files:
                    emit(f"- {tmp}")
                emit("Run 'update' again to retry, or remove these files")

            # Check for races stuck in staging
            from umabozu.database.crud import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM staged_race_ids
                WHERE race_id NOT IN (SELECT race_id FROM races)
            """
            )
            stuck_count = cursor.fetchone()[0]
            conn.close()

            if stuck_count > 0:
                emit("")
                emit(f"Staging backlog: {stuck_count} race(s)")
                emit("Downloaded but not yet fully processed.")
                emit("Run 'update' again to retry processing")

            # Check for incomplete profiles
            from umabozu.database.crud import (
                get_missing_horse_ids,
                get_missing_jockey_ids,
            )

            missing_horses = get_missing_horse_ids()
            missing_jockeys = get_missing_jockey_ids()

            if missing_horses or missing_jockeys:
                emit("")
                emit("Incomplete profiles:")
                if missing_horses:
                    emit(f"- Missing horse profiles: {len(missing_horses)}")
                if missing_jockeys:
                    emit(f"- Missing jockey profiles: {len(missing_jockeys)}")

        except Exception as e:
            logger.error(f"Error retrieving database status: {e}")
            logger.exception(e)

    def _retry_incomplete_html_downloads(self) -> tuple[int, int]:
        temp_files = glob.glob("data/raw/html/**/*.tmp", recursive=True)
        if not temp_files:
            return 0, 0

        retried = 0
        failed = 0
        for tmp_path in temp_files:
            normalized = os.path.normpath(tmp_path)
            parts = normalized.split(os.sep)
            filename = os.path.basename(normalized)
            if not filename.endswith(".html.tmp"):
                continue
            item_id = filename[: -len(".html.tmp")]

            with suppress(OSError):
                os.remove(tmp_path)

            is_race = (
                "races" in parts
                and "race_lists" not in parts
                and "upcoming_races" not in parts
            )
            if is_race:
                html_content = fetch_and_save_race_html(item_id)
            elif "horses" in parts:
                html_content = fetch_and_save_horse_html(item_id)
            elif "jockeys" in parts:
                html_content = fetch_and_save_jockey_html(item_id)
            else:
                continue

            if html_content:
                retried += 1
            else:
                failed += 1

        return retried, failed

    def retry_incomplete_html_downloads(self) -> tuple[int, int]:
        retried, failed = self._retry_incomplete_html_downloads()
        if retried or failed:
            logger.info(
                "Retrying incomplete HTML downloads: %s succeeded, %s failed",
                retried,
                failed,
            )
        else:
            logger.info("No incomplete HTML downloads found.")
        return retried, failed

    def update_data(
        self,
        days_back=None,
        start_date=None,
        end_date=None,
        retry_tmp_only=False,
    ):
        """Sync database with netkeiba data for a specific period."""
        self.retry_incomplete_html_downloads()
        if retry_tmp_only:
            return
        if start_date and end_date:
            s_date = date.fromisoformat(start_date)
            e_date = date.fromisoformat(end_date)
        elif days_back:
            e_date = date.today()
            s_date = e_date - timedelta(days=days_back)
        else:
            e_date = date.today()
            s_date = e_date - timedelta(days=7)
        logger.info(f"Updating data from {s_date} to {e_date}...")
        collect_races(s_date, e_date)
        collect_profiles()
        logger.info("Data update complete.")

    def train_latest_model(self, optimize=False, n_trials=20, test_ratio=None):
        """Retrain the model using all data in the database.

        Args:
            optimize: If True, use Optuna to optimize hyperparameters
            n_trials: Number of Optuna trials (only used if optimize=True)
            test_ratio: If provided (e.g., 0.2), split data for evaluation.
                       If None, use all data for training (production mode).
        """
        logger.info("Preparing training data...")
        df = prepare_training_data(use_db_optimized=True)

        if df.is_empty():
            logger.error("No data available for training.")
            return

        if test_ratio is not None:
            train_df, test_df = split_train_test(df, test_ratio)
            logger.info(f"Training model on {len(train_df)} samples...")
            model = train_model(train_df, optimize=optimize, n_trials=n_trials)

            # Optional: Evaluate on test set
            if not test_df.is_empty():
                logger.info(f"Test set contains {len(test_df)} samples.")
        else:
            logger.info(f"Training model on {len(df)} samples (all data)...")
            model = train_model(df, optimize=optimize, n_trials=n_trials)

        save_model(model, "trained_models/lgbm_model.pkl")
        logger.info("Model training complete and saved.")

    def forecast_race(self, race_id):
        """Analyze and predict a future race using shutsubahyou."""
        logger.info(f"Fetching upcoming race card for {race_id}...")

        html_path = fetch_and_save_shutsubahyou_html(race_id)
        if not html_path:
            return

        content = None
        for enc in ["utf-8", "euc-jp", "cp932"]:
            with suppress(UnicodeDecodeError), open(html_path, encoding=enc) as f:
                content = f.read()
            if content is not None:
                break

        if content is None:
            logger.error(f"Error: Could not decode HTML file {html_path}")
            return
        race_info, runners = parse_shutsubahyou_html(race_id, content)
        if not runners:
            logger.warning("Error: No runners found in race card.")
            return

        # --- Sync Missing Profiles ---
        conn = get_db_connection()
        cursor = conn.cursor()
        missing_h = []
        missing_j = []
        for r in runners:
            cursor.execute(
                "SELECT horse_name FROM horses WHERE horse_id = ?", (r["horse_id"],)
            )
            row = cursor.fetchone()
            if not row or row[0] in ("UNKNOWN", ""):
                missing_h.append(r["horse_id"])

            cursor.execute(
                "SELECT jockey_name FROM jockeys WHERE jockey_id = ?", (r["jockey_id"],)
            )
            row = cursor.fetchone()
            if not row or row[0] in ("UNKNOWN", ""):
                missing_j.append(r["jockey_id"])
        conn.close()

        if missing_h or missing_j:
            logger.info(
                f"Syncing {len(missing_h)} horses and {len(missing_j)} jockeys..."
            )
            for h_id in missing_h:
                p = fetch_and_save_horse_html(h_id)
                if p:
                    with suppress(Exception), open(p, "rb") as f_horse:  # SIM117
                        insert_horse(parse_horse_profile_html(h_id, f_horse.read()))
            for j_id in missing_j:
                p = fetch_and_save_jockey_html(j_id)
                if p:
                    with suppress(Exception), open(p, "rb") as f_jockey:  # SIM117
                        insert_jockey(parse_jockey_profile_html(j_id, f_jockey.read()))

        # --- Prepare DataFrame ---
        # Ensure each runner dictionary has all necessary basic keys before creating DF
        # This will be used in upcoming_df.select later, so ensure existence.
        # Initialize with None/default to prevent KeyError if parse fails
        for r in runners:
            r["race_id"] = race_id  # Ensure race_id is in each runner dict
            r["course_type"] = race_info.get("course_type", "芝")
            r["distance"] = race_info.get("distance", 1600)
            r["course_direction"] = race_info.get("course_direction", "右")
            r["track_condition"] = race_info.get("track_condition", "良")
            r["weather"] = race_info.get("weather", "晴")
            r["race_date"] = date.today().strftime("%Y-%m-%d")

        upcoming_df = pl.DataFrame(runners)

        # Use common pipeline for historical data
        full_training_data = prepare_training_data(use_db_optimized=True)

        # Align schema
        for col in upcoming_df.columns:
            if col in full_training_data.columns:
                upcoming_df = upcoming_df.with_columns(
                    pl.col(col).cast(full_training_data.schema[col])
                )

        # Combine and generate features for the target race
        all_data = pl.concat([full_training_data, upcoming_df], how="diagonal")
        all_data = clean_data(all_data)
        all_data = generate_features(all_data)

        target_df = all_data.filter(pl.col("race_id") == race_id)
        model = load_model("trained_models/lgbm_model.pkl")
        feature_cols = get_valid_feature_cols(target_df)
        probas = predict_probabilities(model, target_df, feature_cols)
        df_ev = calculate_expected_value(probas, target_df)
        recommendations = simulate_betting(
            df_ev, ev_threshold=self.ev_threshold, min_probability=self.min_probability
        )

        r_name = race_info.get("race_name", "Unknown Race")
        r_num = race_info.get("race_number", "")
        logger.info(f"\n--- Forecast: {r_num} {r_name} ({race_id}) ---")

        if recommendations.is_empty():
            logger.info("No high-confidence bets found.")
        else:
            logger.info(
                f"Showing recommendations (EV > {self.ev_threshold}"
                f" and Win Prob > {self.min_probability}):"
            )

            display_df = (
                recommendations.join(
                    df_ev.select(
                        ["race_id", "horse_id", "expected_value", "win_probability"]
                    ),
                    on=["race_id", "horse_id"],
                    how="inner",
                )
                .join(
                    upcoming_df.select(
                        [
                            c
                            for c in [
                                "race_id",
                                "horse_id",
                                "horse_name",
                                "horse_number",
                                "popularity",
                                "odds_win",
                            ]
                            if c in upcoming_df.columns
                        ]
                    ),
                    on=["race_id", "horse_id"],
                    how="inner",
                )
                .unique(subset=["horse_id"])
            )

            display_df = display_df.filter(
                pl.col("expected_value") >= self.ev_threshold
            ).sort("expected_value", descending=True)

            for row in display_df.to_dicts():
                h_num = row.get("horse_number", "??")
                h_name = row.get("horse_name", "Unknown")
                h_pop = row.get("popularity", "?")
                ev = row.get("expected_value", 0.0)
                odds = row.get("odds_win", 0.0)
                prob = row.get("win_probability", 0.0)
                logger.info(
                    f"馬番: {h_num:>2} | {h_name} (人気: {h_pop}) | "
                    f"期待値: {ev:.2f} | 単勝: {odds} | 勝率: {prob * 100:.1f}%"
                )

    def simulate_backtest(
        self,
        test_ratio=0.2,
        model_path="trained_models/lgbm_model.pkl",
        retrain=False,
        optimize=False,
        n_trials=20,
    ):
        """
        Run a backtesting simulation using train/test split and ROI calculation.

        To prevent leakage, simulate always retrains on the training split and
        evaluates on the test split. Existing models are not used.

        Args:
            test_ratio: Proportion of data to use for testing (default: 0.2)
            model_path: Ignored. simulate always retrains to avoid leakage.
            retrain: Ignored. simulate always retrains to avoid leakage.
            optimize: If True and retrain=True, use Optuna optimization
            n_trials: Number of Optuna trials
                     (only used if retrain=True and optimize=True)
        """
        logger.info("--- Starting Backtest Simulation ---")

        # Load and prepare data
        logger.info("Preparing data...")
        df = prepare_training_data(use_db_optimized=True)

        if df.is_empty():
            logger.error("No data available for backtesting.")
            return

        # Split data
        train_df, test_df = split_train_test(df, test_ratio)

        if test_df.is_empty():
            logger.error("Test set is empty. Cannot run backtest.")
            return

        if not retrain:
            logger.info(
                "simulate always retrains on the training split to avoid leakage; "
                "ignoring existing model."
            )

        logger.info(
            f"Training new model (optimize={optimize}, n_trials={n_trials})..."
        )
        model = train_model(train_df, optimize=optimize, n_trials=n_trials)
        logger.info("Model training complete.")
        # Predict on test set
        logger.info("Predicting probabilities on test set...")
        feature_cols = get_valid_feature_cols(test_df)
        predicted_probas_df = predict_probabilities(model, test_df, feature_cols)

        # Calculate expected value
        logger.info("Calculating expected values...")
        df_with_ev = calculate_expected_value(predicted_probas_df, test_df)

        # Simulate betting
        logger.info(
            f"Simulating betting (EV > {self.ev_threshold}, "
            f"Prob > {self.min_probability})..."
        )
        selected_bets = simulate_betting(
            df_with_ev,
            ev_threshold=self.ev_threshold,
            min_probability=self.min_probability,
        )

        # Calculate payout and ROI
        from umabozu.simulation.bet_simulator import calculate_payout

        investment, total_payout, roi = calculate_payout(selected_bets)

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)

        if not test_df.is_empty() and "race_date" in test_df.columns:
            test_start = test_df.select("race_date").min().item()
            test_end = test_df.select("race_date").max().item()
            logger.info(f"Test period: {test_start} to {test_end}")

        logger.info(f"Total races: {test_df.select('race_id').n_unique():,}")
        logger.info(f"Total horses evaluated: {len(test_df):,}")
        logger.info(f"Horses meeting criteria: {len(df_with_ev):,}")
        logger.info(f"Bets placed: {len(selected_bets):,}")
        logger.info("-" * 60)

        if not selected_bets.is_empty():
            logger.info(f"Total investment: {investment:,.0f} JPY")
            logger.info(f"Total payout: {total_payout:,.0f} JPY")
            logger.info(f"Net profit: {total_payout - investment:,.0f} JPY")
            logger.info(f"Return on Investment (ROI): {roi:.2f}%")

            # Additional statistics
            wins = selected_bets.filter(pl.col("rank") == 1)
            win_count = len(wins)
            hit_rate = (
                (win_count / len(selected_bets) * 100) if len(selected_bets) > 0 else 0
            )
            logger.info(f"Winning bets: {win_count} / {len(selected_bets)}")
            logger.info(f"Hit rate: {hit_rate:.2f}%")
        else:
            logger.info("No bets were placed with the current criteria.")

        logger.info("=" * 60)
        logger.info("--- Backtest Simulation Complete ---\n")
