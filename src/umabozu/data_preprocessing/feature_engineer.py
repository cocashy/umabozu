import polars as pl

FEATURE_SCHEMA_VERSION = 2


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
        "age_at_race_days",
        "jockey_experience_years",
        # Rolling features
        "horse_rolling_rank_mean_3",
        "horse_rolling_rank_mean_5",
        "horse_rolling_rank_mean_10",
        "horse_rolling_last3f_mean_3",
        "horse_rolling_last3f_mean_5",
        "horse_rolling_last3f_mean_10",
    ]
