"""
Tests for data preprocessing module.

Following t-wada pragmatic testing approach:
- Focus on real data transformations that matter for model quality
- Test boundary conditions: empty DataFrames, missing columns, malformed data
- Verify data types and structure post-processing
"""

import polars as pl
import pytest

from umabozu.data_preprocessing.preprocessor import clean_data


class TestCleanData:
    """Test suite for the clean_data function."""

    def test_clean_data_empty_dataframe(self):
        """Empty DataFrame should be returned as-is."""
        df = pl.DataFrame()
        result = clean_data(df)
        assert result.is_empty()

    def test_clean_data_preserves_columns(self):
        """Verify important columns are preserved after cleaning."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02"],
                "horse_birth_date": ["2020-01-01", "2020-01-02"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02"],
                "debut_date": ["2021-01-01", "2021-01-02"],
                "distance": [1000, 1200],
                "odds_win": ["1.5", "2.0"],
                "rank": [1, 2],
            }
        )
        result = clean_data(df)
        assert "distance" in result.columns
        assert "rank" in result.columns
        assert "odds_win" in result.columns

    def test_clean_data_handles_malformatted_dates(self):
        """Invalid dates should be converted to null, then handled."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "invalid-date", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        assert "race_date" in result.columns
        # Invalid dates become null, filled with 0 by fill_null
        assert len(result) == 3

    def test_clean_data_converts_numeric_strings(self):
        """String representations of numbers should be converted."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "distance": ["1000", "1200", "invalid"],
                "odds_win": ["1.5", "2.0", "3.5"],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        # All should be Float64 after cleaning (invalid becomes 0)
        assert result["distance"].dtype == pl.Float64
        assert result["odds_win"].dtype == pl.Float64

    def test_clean_data_fills_nulls_with_zero(self):
        """Missing numeric values should be filled with 0."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "distance": [1000.0, None, 1200.0],
                "odds_win": [1.5, 2.0, None],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        assert not result["distance"].is_null().any()
        assert not result["odds_win"].is_null().any()
        # Verify nulls are replaced with 0
        assert result["distance"][1] == 0.0
        assert result["odds_win"][2] == 0.0

    def test_clean_data_encodes_categorical_features(self):
        """Categorical columns should be converted and one-hot encoded."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "course_type": ["芝", "ダ", "芝"],
                "weather": ["晴", "晴", "雨"],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        # After one-hot encoding, original categorical columns are expanded
        assert any("course_type_" in col for col in result.columns)
        # Original columns might still exist or be replaced with dummies
        assert len(result) == 3

    def test_clean_data_converts_rank_to_int64(self):
        """'rank' column should be Int64 type."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "rank": [1, 2, 3],
                "distance": [1000, 1200, 1400],
            }
        )
        result = clean_data(df)
        assert result["rank"].dtype == pl.Int64

    def test_clean_data_with_weight_columns(self):
        """Weight columns should be handled as Float64."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "horse_weight": [450.0, 480.0, 470.0],
                "horse_weight_diff": [5.0, -10.0, 0.0],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        assert result["horse_weight"].dtype == pl.Float64
        assert result["horse_weight_diff"].dtype == pl.Float64

    def test_clean_data_missing_optional_cols_no_error(self):
        """Missing optional columns should not cause errors."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01"],
                "horse_birth_date": ["2020-01-01"],
                "jockey_birth_date": ["1990-01-01"],
                "debut_date": ["2021-01-01"],
                "rank": [1],
                # Missing: distance, odds_win, etc. - these are optional
            }
        )
        result = clean_data(df)
        # Should succeed without errors
        assert len(result) == 1

    def test_clean_data_all_null_numeric_column(self):
        """Column with all nulls should be filled with 0."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "odds_win": [None, None, None],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)
        assert (result["odds_win"] == 0.0).all()

    def test_clean_data_preserves_data_count(self):
        """clean_data should not change the number of rows."""
        df = pl.DataFrame(
            {
                "race_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "horse_birth_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "rank": [1, 2, 3],
            }
        )
        initial_count = len(df)
        result = clean_data(df)
        assert len(result) == initial_count


class TestCleanDataIntegration:
    """Integration tests with realistic data."""

    def test_clean_realistic_race_data(self):
        """Test with realistic race result data structure."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101", "202301010102", "202301010103"],
                "race_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
                "horse_id": ["2020101234", "2021102345", "2022103456"],
                "horse_birth_date": ["2020-06-01", "2021-03-15", "2021-09-20"],
                "jockey_birth_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
                "debut_date": ["2021-08-01", "2022-05-10", "2022-11-15"],
                "distance": [1600, 2000, 1200],
                "course_type": ["\u829d", "\u30c0", "\u829d"],
                "odds_win": [2.5, 3.2, 1.8],
                "popularity": [3, 5, 2],
                "horse_weight": [450, 480, 470],
                "horse_weight_diff": [5, -10, 0],
                "weight_carried": [55.0, 56.0, 56.5],
                "rank": [1, 2, 3],
            }
        )
        result = clean_data(df)

        # Verify all rows are preserved
        assert len(result) == 3

        # Verify key data types for numeric columns
        assert result["distance"].dtype == pl.Float64

        # Verify no nulls in important numeric columns
        numeric_cols = ["distance", "odds_win", "horse_weight"]
        for col in numeric_cols:
            if col in result.columns:
                assert not result[col].is_null().any()
