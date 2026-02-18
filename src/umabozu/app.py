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

    def train_latest_model(self):
        """Retrain the model using all data in the database."""
        logger.info("Loading data for training...")
        df = get_training_data()
        df = clean_data(df)
        df = generate_features(df)
        logger.info(f"Training model on {len(df)} samples...")
        model = train_model(df, optimize=False)
        save_model(model, "trained_models/lgbm_model.pkl")
        logger.info("Model training complete and saved.")

    def predict_upcoming(self, target_date=None):
        """Predict and show recommended bets for a specific date (from DB)."""
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")
        logger.info(f"Analyzing races for {target_date}...")
        try:
            model = load_model("trained_models/lgbm_model.pkl")
        except Exception:  # E722
            logger.error("Error: Model not found. Please run 'train' first.")
            return

        df = load_data_from_db()  # F821
        df = df.filter(pl.col("race_date") == str(target_date))
        if df.is_empty():
            logger.info(
                f"No race data found for {target_date} in DB. Run 'update' first."
            )
            return

        df = clean_data(df)
        df = generate_features(df)
        feature_cols = [col for col in get_feature_cols() if col in df.columns]

        probas = predict_probabilities(model, df, feature_cols)
        df_ev = calculate_expected_value(probas, df)
        recommendations = simulate_betting(
            df_ev, ev_threshold=self.ev_threshold, min_probability=self.min_probability
        )

        logger.info(
            f"\n--- Backtest Recommendations for {target_date} ---"  # E501
        )

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
                    df.select(["race_id", "horse_id", "horse_name", "popularity"]),
                    on=["race_id", "horse_id"],
                    how="inner",
                )
                .unique(subset=["horse_id"])
            )
            display_df = display_df.filter(
                pl.col("expected_value") >= self.ev_threshold
            ).sort("expected_value", descending=True)

            for row in display_df.to_dicts():
                logger.info(
                    f"Race ID: {row.get('race_id', 'Unknown')} | "
                    f"馬名: {row.get('horse_name', 'Unknown')} | "
                    f"人気: {row.get('popularity', '?')} | "
                    f"単勝オッズ: {row.get('odds_win', 0.0)} | "
                    f"期待値: {row.get('expected_value', 0.0):.2f}"
                )

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

        full_training_data = get_training_data()
        for col in upcoming_df.columns:
            if col in full_training_data.columns:
                upcoming_df = upcoming_df.with_columns(
                    pl.col(col).cast(full_training_data.schema[col])
                )

        all_data = pl.concat([full_training_data, upcoming_df], how="diagonal")
        all_data = clean_data(all_data)
        all_data = generate_features(all_data)

        target_df = all_data.filter(pl.col("race_id") == race_id)
        model = load_model("trained_models/lgbm_model.pkl")
        feature_cols = [col for col in get_feature_cols() if col in target_df.columns]
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
