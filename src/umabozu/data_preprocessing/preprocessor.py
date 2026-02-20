from typing import Any

import polars as pl


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans the raw Polars DataFrame.
    - Handles missing values (filling with 0 or mean/median).
    - Converts data types.
    - Encodes categorical features.
    """
    if df.is_empty():
        return df

    # Convert date strings to Date type
    # For race_date, birth_date, debut_date, ensure they are in YYYY-MM-DD format
    # then convert to Date. Errors as null.
    # Only convert if the column is String type (skip if already Date type)
    date_cols = [
        "race_date",
        "horse_birth_date",
        "jockey_birth_date",
        "debut_date",
    ]
    date_conversions = []
    for col in date_cols:
        if col in df.columns:
            col_type = df.schema.get(col)
            # Only convert if it's a String type (not already Date)
            if col_type == pl.String:
                date_conversions.append(pl.col(col).str.to_date("%Y-%m-%d", strict=False))
    
    if date_conversions:
        df = df.with_columns(*date_conversions)

    # Convert numeric columns, coercing errors to null
    numeric_cols = [
        "distance",
        "odds_win",
        "popularity",
        "horse_weight",
        "horse_weight_diff",
        "weight_carried",
        "distance_diff",
        "last_3f_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0)
            )

    # Fill remaining nulls in numeric columns with 0
    # (after casting, so that new nulls from coercion are also filled)
    df = df.fill_null(0)

    # Convert categorical features to numeric using one-hot encoding or similar
    # For simplicity, we'll just convert some common ones to category type.
    # LightGBM can handle categorical features if specified.
    categorical_cols = [
        "course_type",
        "course_direction",
        "track_condition",
        "weather",
        "sex",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

    # One-hot encode categorical columns
    # Polars has get_dummies for one-hot encoding
    df = df.to_dummies(columns=categorical_cols, separator="_")

    # Ensure target column ('rank') is integer
    if "rank" in df.columns:
        df = df.with_columns(pl.col("rank").cast(pl.Int64))

    return df
