import polars as pl


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

    # Sort data for window functions to prevent data leakage
    df = df.sort(by=["horse_name", "race_date"])

    # Feature 3: 過去の成績 (rolling statistics)
    # Window functions need to be applied carefully to avoid data leakage
    # Shift by 1 to only use past data
    rolling_windows = [3, 5, 10]
    for window in rolling_windows:
        df = df.with_columns(
            pl.col("rank")
            .shift(1)
            .rolling_mean(window_size=window)
            .over("horse_name")
            .alias(f"horse_rolling_rank_mean_{window}")
        )
        df = df.with_columns(
            pl.col("last_3f_time")
            .shift(1)
            .rolling_mean(window_size=window)
            .over("horse_name")
            .alias(f"horse_rolling_last3f_mean_{window}")
        )

    # Fill any NaN values that result from rolling calculations
    # (e.g., for first few races)
    return df.fill_null(0)


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
