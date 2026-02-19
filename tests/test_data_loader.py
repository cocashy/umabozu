"""
Tests for data_loader module.

Following t-wada pragmatic testing approach:
- Test core functionality: loading data from database
- Test error handling: database not found, empty database
- Test data integrity: returned data structure
"""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from umabozu.data_preprocessing.data_loader import load_data_from_db


class TestLoadDataFromDb:
    """Test suite for load_data_from_db function."""

    def test_load_data_from_db_returns_dataframe(self):
        """Should return a Polars DataFrame."""
        # Mock get_training_data to return a sample DataFrame
        sample_df = pl.DataFrame(
            {
                "race_id": ["202301010101", "202301010102"],
                "horse_id": ["2020101234", "2021102345"],
                "rank": [1, 2],
            }
        )

        with patch("umabozu.data_preprocessing.data_loader.get_training_data") as mock:
            mock.return_value = sample_df
            result = load_data_from_db()

            assert isinstance(result, pl.DataFrame)
            assert len(result) == 2

    def test_load_data_from_db_preserves_columns(self):
        """Returned DataFrame should maintain expected columns."""
        sample_df = pl.DataFrame(
            {
                "race_id": ["202301010101"],
                "horse_id": ["2020101234"],
                "jockey_id": ["00123"],
                "rank": [1],
                "odds_win": [2.5],
            }
        )

        with patch("umabozu.data_preprocessing.data_loader.get_training_data") as mock:
            mock.return_value = sample_df
            result = load_data_from_db()

            assert "race_id" in result.columns
            assert "horse_id" in result.columns
            assert "rank" in result.columns

    def test_load_data_from_db_handles_empty_result(self):
        """Should handle case when database returns empty DataFrame."""
        empty_df = pl.DataFrame()

        with patch("umabozu.data_preprocessing.data_loader.get_training_data") as mock:
            mock.return_value = empty_df
            result = load_data_from_db()

            assert result.is_empty()
            assert isinstance(result, pl.DataFrame)

    def test_load_data_from_db_integrates_with_pipeline(self):
        """Data should be compatible with preprocessing pipeline."""
        from umabozu.data_preprocessing.preprocessor import clean_data

        sample_df = pl.DataFrame(
            {
                "race_id": ["202301010101"],
                "horse_id": ["2020101234"],
                "race_date": ["2023-01-01"],
                "horse_birth_date": ["2020-06-01"],
                "jockey_birth_date": ["1990-01-01"],
                "debut_date": ["2021-08-01"],
                "rank": [1],
                "odds_win": [2.5],
                "distance": [1600],
            }
        )

        with patch("umabozu.data_preprocessing.data_loader.get_training_data") as mock:
            mock.return_value = sample_df
            loaded = load_data_from_db()

            # Should be compatible with preprocessor
            cleaned = clean_data(loaded)
            assert isinstance(cleaned, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
