import polars as pl

FEATURE_SCHEMA_VERSION = 3


def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates advanced features including past performance stats
    from the combined DataFrame.
    """
    if df.is_empty():
        return df

    # Feature 1: 馬齢 (age)
    # Note: birth_date is YYYY-MM-DD string, race_date is YYYY-MM-DD string
    df = df.with_columns(
        (pl.col("race_date") - pl.col("horse_birth_date"))
        .dt.total_days()
        .alias("age_at_race_days")
    )
    df = df.with_columns(
        (pl.col("age_at_race_days") / 365.25).alias("age_at_race_years")
    )

    # Feature 2: 騎手の経験年数 (jockey_experience_years)
    df = df.with_columns(
        (pl.col("race_date").dt.year() - pl.col("debut_date").dt.year()).alias(
            "jockey_experience_years"
        )
    )

    # Convert necessary columns to numeric, coercing errors to null
    numeric_cols = [
        "horse_weight",
        "horse_weight_diff",
        "weight_carried",
        "distance_diff",
        "last_3f_time",
        "age_at_race_days",
        "age_at_race_years",
        "jockey_experience_years",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Preserve the result-page observations separately. The model feature
    # columns below use the last known value from a prior race, which is also
    # available when forecasting before the current race's weigh-in.
    for col in ["horse_weight", "horse_weight_diff"]:
        observed_col = f"{col}_observed"
        if col in df.columns and observed_col not in df.columns:
            df = df.with_columns(pl.col(col).alias(observed_col))

    # Missing IDs must not be grouped together: that would mix different horses'
    # race outcomes into one history. Give each such row an isolated key.
    row_index_col = "__feature_row_index"
    history_key_col = "__horse_history_key"
    df = df.with_row_index(row_index_col)

    if "horse_id" in df.columns:
        horse_id = pl.col("horse_id").cast(pl.String, strict=False).str.strip_chars()
        valid_horse_id = horse_id.is_not_null() & ~horse_id.str.to_uppercase().is_in(
            ["", "0", "UNKNOWN", "NONE", "NULL"]
        )
        df = df.with_columns(
            pl.when(valid_horse_id)
            .then(pl.concat_str([pl.lit("horse:"), horse_id]))
            .otherwise(
                pl.concat_str([pl.lit("row:"), pl.col(row_index_col).cast(pl.String)])
            )
            .alias(history_key_col)
        )
    else:
        # Without a stable horse ID, no historical result is safe to reuse.
        df = df.with_columns(
            pl.concat_str(
                [pl.lit("row:"), pl.col(row_index_col).cast(pl.String)]
            ).alias(history_key_col)
        )

    # A horse cannot run multiple races on one date. Refuse ambiguous ordering
    # rather than allowing one same-day result to become another race's feature.
    duplicate_horse_days = (
        df.filter(pl.col(history_key_col).str.starts_with("horse:"))
        .group_by([history_key_col, "race_date"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_horse_days.height:
        raise ValueError(
            "Cannot generate past-performance features: a horse has multiple "
            "records on the same race_date."
        )

    sort_cols = [history_key_col, "race_date"]
    sort_cols.extend(col for col in ["race_id", "horse_number"] if col in df.columns)
    df = df.sort(sort_cols, maintain_order=True)

    for col in ["horse_weight", "horse_weight_diff"]:
        observed_col = f"{col}_observed"
        if observed_col in df.columns:
            df = df.with_columns(
                pl.col(observed_col).shift(1).over(history_key_col).alias(col)
            )

    # Feature 3: 過去の成績 (rolling statistics)
    # Window functions need to be applied carefully to avoid data leakage
    # Shift by 1 to only use past data
    rolling_windows = [3, 5, 10]
    for window in rolling_windows:
        for source_col, feature_col in [
            ("rank", f"horse_rolling_rank_mean_{window}"),
            ("last_3f_time", f"horse_rolling_last3f_mean_{window}"),
        ]:
            if source_col in df.columns:
                expression = (
                    pl.col(source_col)
                    .shift(1)
                    .rolling_mean(window_size=window)
                    .over(history_key_col)
                )
            else:
                expression = pl.lit(0.0)
            df = df.with_columns(expression.alias(feature_col))

    if "rank" in df.columns:
        df = df.with_columns(
            (pl.col("rank") == 1).cast(pl.Int64).alias("__history_win"),
            (pl.col("rank") <= 3).cast(pl.Int64).alias("__history_top3"),
        )
        df = df.with_columns(
            (pl.col("rank").cum_count().over(history_key_col) - 1)
            .cast(pl.Float64)
            .alias("horse_prior_starts"),
            (
                pl.col("__history_win").cum_sum().over(history_key_col)
                - pl.col("__history_win")
            ).alias("__horse_prior_wins"),
            (
                pl.col("__history_top3").cum_sum().over(history_key_col)
                - pl.col("__history_top3")
            ).alias("__horse_prior_top3"),
            (pl.col("rank").cum_sum().over(history_key_col) - pl.col("rank")).alias(
                "__horse_prior_rank_sum"
            ),
            pl.col("rank")
            .shift(1)
            .over(history_key_col)
            .cast(pl.Float64)
            .alias("horse_last_rank"),
        )
        df = df.with_columns(
            (
                (pl.col("__horse_prior_wins") + 1.0)
                / (pl.col("horse_prior_starts") + 12.0)
            ).alias("horse_prior_win_rate"),
            (
                (pl.col("__horse_prior_top3") + 4.0)
                / (pl.col("horse_prior_starts") + 12.0)
            ).alias("horse_prior_top3_rate"),
            pl.when(pl.col("horse_prior_starts") > 0)
            .then(pl.col("__horse_prior_rank_sum") / pl.col("horse_prior_starts"))
            .otherwise(8.0)
            .alias("horse_prior_avg_rank"),
        )

        if "race_date" in df.columns:
            df = df.with_columns(
                pl.col("race_date")
                .diff()
                .over(history_key_col)
                .dt.total_days()
                .cast(pl.Float64)
                .alias("horse_days_since_last")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("horse_days_since_last"))

        if "distance" in df.columns:
            df = df.with_columns(
                (pl.col("distance") - pl.col("distance").shift(1).over(history_key_col))
                .cast(pl.Float64)
                .alias("horse_distance_change")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("horse_distance_change"))

        # Aggregate jockey results by date, then shift the cumulative totals.
        # This prevents an earlier race on the same date from influencing a
        # later race's prediction when the whole card is predicted in advance.
        jockey_feature_cols = [
            "jockey_prior_starts",
            "jockey_prior_win_rate",
            "jockey_prior_top3_rate",
        ]
        existing_jockey_features = [
            col for col in jockey_feature_cols if col in df.columns
        ]
        if existing_jockey_features:
            df = df.drop(existing_jockey_features)

        if "jockey_id" in df.columns and "race_date" in df.columns:
            jockey_id = (
                pl.col("jockey_id").cast(pl.String, strict=False).str.strip_chars()
            )
            valid_jockey_id = (
                jockey_id.is_not_null()
                & ~jockey_id.str.to_uppercase().is_in(
                    ["", "0", "UNKNOWN", "NONE", "NULL"]
                )
            )
            jockey_key_col = "__jockey_history_key"
            df = df.with_columns(
                pl.when(valid_jockey_id)
                .then(pl.concat_str([pl.lit("jockey:"), jockey_id]))
                .otherwise(
                    pl.concat_str(
                        [pl.lit("row:"), pl.col(row_index_col).cast(pl.String)]
                    )
                )
                .alias(jockey_key_col)
            )

            jockey_daily = (
                df.group_by([jockey_key_col, "race_date"])
                .agg(
                    pl.len().alias("__jockey_day_starts"),
                    pl.col("__history_win").sum().alias("__jockey_day_wins"),
                    pl.col("__history_top3").sum().alias("__jockey_day_top3"),
                )
                .sort([jockey_key_col, "race_date"], maintain_order=True)
                .with_columns(
                    (
                        pl.col("__jockey_day_starts").cum_sum().over(jockey_key_col)
                        - pl.col("__jockey_day_starts")
                    ).alias("jockey_prior_starts"),
                    (
                        pl.col("__jockey_day_wins").cum_sum().over(jockey_key_col)
                        - pl.col("__jockey_day_wins")
                    ).alias("__jockey_prior_wins"),
                    (
                        pl.col("__jockey_day_top3").cum_sum().over(jockey_key_col)
                        - pl.col("__jockey_day_top3")
                    ).alias("__jockey_prior_top3"),
                )
                .with_columns(
                    (
                        (pl.col("__jockey_prior_wins") + 1.0)
                        / (pl.col("jockey_prior_starts") + 12.0)
                    ).alias("jockey_prior_win_rate"),
                    (
                        (pl.col("__jockey_prior_top3") + 4.0)
                        / (pl.col("jockey_prior_starts") + 12.0)
                    ).alias("jockey_prior_top3_rate"),
                )
                .select(
                    [
                        jockey_key_col,
                        "race_date",
                        "jockey_prior_starts",
                        "jockey_prior_win_rate",
                        "jockey_prior_top3_rate",
                    ]
                )
            )
            df = df.join(
                jockey_daily,
                on=[jockey_key_col, "race_date"],
                how="left",
            )
            df = df.drop(jockey_key_col)
        else:
            df = df.with_columns(
                pl.lit(0.0).alias("jockey_prior_starts"),
                pl.lit(1.0 / 12.0).alias("jockey_prior_win_rate"),
                pl.lit(1.0 / 3.0).alias("jockey_prior_top3_rate"),
            )

        df = df.drop(
            [
                "__history_win",
                "__history_top3",
                "__horse_prior_wins",
                "__horse_prior_top3",
                "__horse_prior_rank_sum",
            ]
        )

    # Fill any NaN values that result from rolling calculations
    # (e.g., for first few races)
    return df.drop([row_index_col, history_key_col]).fill_null(0)


def get_feature_cols() -> list[str]:
    """Returns a list of feature columns used for the model."""
    return [
        "horse_weight",
        "horse_weight_diff",
        "weight_carried",
        "distance",  # From race data
        "horse_number",
        "num_of_runners",
        "race_number",
        "age_at_race_days",
        "jockey_experience_years",
        # Rolling features
        "horse_rolling_rank_mean_3",
        "horse_rolling_rank_mean_5",
        "horse_rolling_rank_mean_10",
        "horse_rolling_last3f_mean_3",
        "horse_rolling_last3f_mean_5",
        "horse_rolling_last3f_mean_10",
        # Pre-race horse and jockey history
        "horse_prior_starts",
        "horse_prior_win_rate",
        "horse_prior_top3_rate",
        "horse_prior_avg_rank",
        "horse_last_rank",
        "horse_days_since_last",
        "horse_distance_change",
        "jockey_prior_starts",
        "jockey_prior_win_rate",
        "jockey_prior_top3_rate",
    ]
