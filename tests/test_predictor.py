"""
Tests for predictor module.

Following t-wada pragmatic testing approach:
- Focus on core prediction logic: probability inference
- Test edge cases: empty data, missing features, model loading/saving
- Verify output structure and value ranges (probabilities 0-1)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import polars as pl
import pytest
from sklearn.base import BaseEstimator

from umabozu.models.predictor import (
    load_model,
    predict_probabilities,
    save_model,
)


class MockLGBMClassifier(BaseEstimator):
    """Mock LightGBM classifier for testing."""

    def predict_proba(self, x):
        """Return mock probabilities."""
        import numpy as np

        n_samples = len(x)
        # Return 4 classes (0-3 for rank 1-4+)
        return np.random.dirichlet([1, 1, 1, 1], n_samples)


class TestPredictProbabilities:
    """Test suite for predict_probabilities function."""

    def test_predict_probabilities_returns_dataframe(self):
        """Should return a Polars DataFrame."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0],
                "popularity": [3, 5],
            }
        )

        mock_model = MockLGBMClassifier()
        feature_cols = ["odds_win", "popularity"]

        result = predict_probabilities(mock_model, df, feature_cols)

        assert isinstance(result, pl.DataFrame)

    def test_predict_probabilities_empty_dataframe(self):
        """Empty input should return DataFrame with empty probability columns."""
        df = pl.DataFrame()
        mock_model = MockLGBMClassifier()

        result = predict_probabilities(mock_model, df, [])

        assert result.is_empty()
        assert "proba_0" in result.columns
        assert "proba_1" in result.columns
        assert "proba_2" in result.columns
        assert "proba_3" in result.columns

    def test_predict_probabilities_has_required_columns(self):
        """Output should have probability columns for all classes."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0],
                "popularity": [3],
            }
        )

        mock_model = MockLGBMClassifier()
        feature_cols = ["odds_win", "popularity"]

        result = predict_probabilities(mock_model, df, feature_cols)

        assert "proba_0" in result.columns
        assert "proba_1" in result.columns
        assert "proba_2" in result.columns
        assert "proba_3" in result.columns

    def test_predict_probabilities_adds_missing_features(self):
        """Missing feature columns should be added with value 0."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0],
                "popularity": [3],
            }
        )

        mock_model = MockLGBMClassifier()
        feature_cols = [
            "odds_win",
            "popularity",
            "missing_feature_1",
            "missing_feature_2",
        ]

        result = predict_probabilities(mock_model, df, feature_cols)

        # Should not raise error, and should have probability output
        assert isinstance(result, pl.DataFrame)
        assert "proba_0" in result.columns

    def test_predict_probabilities_values_in_valid_range(self):
        """Probability values should be between 0 and 1."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0, 5.0],
                "popularity": [3, 5, 8],
            }
        )

        mock_model = MockLGBMClassifier()
        feature_cols = ["odds_win", "popularity"]

        result = predict_probabilities(mock_model, df, feature_cols)

        for col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
            assert (result[col] >= 0.0).all()
            assert (result[col] <= 1.0).all()

    def test_predict_probabilities_sum_to_one(self):
        """Sum of probabilities across classes should equal 1 (approximately)."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0],
            }
        )

        mock_model = MockLGBMClassifier()

        result = predict_probabilities(mock_model, df, ["odds_win"])

        # Sum probabilities
        total = (
            result["proba_0"][0]
            + result["proba_1"][0]
            + result["proba_2"][0]
            + result["proba_3"][0]
        )

        assert total == pytest.approx(1.0, abs=0.01)

    def test_predict_probabilities_multiple_samples(self):
        """Should handle multiple horse predictions."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0, 4.0],
                "popularity": [3, 5, 8],
            }
        )

        mock_model = MockLGBMClassifier()

        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        assert len(result) == 3

    def test_predict_probabilities_with_realistic_features(self):
        """Test with realistic feature set."""
        df = pl.DataFrame(
            {
                "odds_win": [2.5, 3.2, 5.0],
                "popularity": [3, 7, 12],
                "horse_weight": [450, 480, 470],
                "age_at_race_years": [3.0, 2.5, 4.0],
                "jockey_experience_years": [5, 8, 3],
            }
        )

        feature_cols = [
            "odds_win",
            "popularity",
            "horse_weight",
            "age_at_race_years",
            "jockey_experience_years",
        ]

        mock_model = MockLGBMClassifier()

        result = predict_probabilities(mock_model, df, feature_cols)

        assert len(result) == 3
        # Verify probability columns exist and have valid values
        for col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
            assert (result[col] > 0.0).any()  # At least some positive probability


class TestLoadModel:
    """Test suite for load_model function."""

    def test_load_model_returns_object(self):
        """Should return the loaded model object."""
        mock_model = MockLGBMClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            joblib.dump(mock_model, str(model_path))

            loaded = load_model(str(model_path))

            assert isinstance(loaded, MockLGBMClassifier)


class TestSaveModel:
    """Test suite for save_model function."""

    def test_save_model_creates_file(self):
        """Should create a file at specified path."""
        mock_model = MockLGBMClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            save_model(mock_model, str(model_path))

            assert model_path.exists()

    def test_save_and_load_roundtrip(self):
        """Saved and reloaded model should be equivalent."""
        mock_model = MockLGBMClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            save_model(mock_model, str(model_path))
            loaded = load_model(str(model_path))

            assert isinstance(loaded, MockLGBMClassifier)
            assert hasattr(loaded, "predict_proba")


class TestPredictorIntegration:
    """Integration tests for prediction pipeline."""

    def test_prediction_pipeline_full_workflow(self):
        """Test complete prediction workflow."""
        # Create sample data
        df = pl.DataFrame(
            {
                "race_id": ["202301010101", "202301010101"],
                "horse_id": ["H001", "H002"],
                "odds_win": [2.5, 3.2],
                "popularity": [3, 7],
                "horse_weight": [450, 480],
                "age_at_race_years": [3.0, 2.5],
                "jockey_experience_years": [5, 8],
            }
        )

        feature_cols = [
            "odds_win",
            "popularity",
            "horse_weight",
            "age_at_race_years",
            "jockey_experience_years",
        ]

        # Create and save model
        mock_model = MockLGBMClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            save_model(mock_model, str(model_path))

            # Load model and make predictions
            loaded_model = load_model(str(model_path))
            probas = predict_probabilities(loaded_model, df, feature_cols)

            # Verify results
            assert len(probas) == 2
            assert "proba_0" in probas.columns
            # Verify probabilities are valid
            for prob_col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
                assert (probas[prob_col] >= 0.0).all()
                assert (probas[prob_col] <= 1.0).all()


class TestPredictProbabilitiesAdvanced:
    """Additional tests for edge cases and advanced scenarios."""

    def test_predict_probabilities_single_horse(self):
        """Should handle prediction for single horse."""
        df = pl.DataFrame(
            {
                "odds_win": [5.0],
                "popularity": [10],
            }
        )

        mock_model = MockLGBMClassifier()
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        assert len(result) == 1
        total_prob = (
            result["proba_0"][0]
            + result["proba_1"][0]
            + result["proba_2"][0]
            + result["proba_3"][0]
        )
        assert total_prob == pytest.approx(1.0, abs=0.01)

    def test_predict_probabilities_large_dataset(self):
        """Should handle large dataset efficiently."""
        import numpy as np

        np.random.seed(42)
        df = pl.DataFrame(
            {
                "odds_win": np.random.uniform(1.5, 30.0, 1000),
                "popularity": np.random.randint(1, 20, 1000),
            }
        )

        mock_model = MockLGBMClassifier()
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        assert len(result) == 1000
        # All rows should have valid probabilities
        for col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
            assert (result[col] >= 0.0).all()
            assert (result[col] <= 1.0).all()

    def test_predict_probabilities_with_float_feature_values(self):
        """Should handle float-type feature values correctly."""
        df = pl.DataFrame(
            {
                "odds_win": [2.5, 3.75, 4.25],
                "popularity": [3.0, 5.0, 7.0],
            }
        )

        mock_model = MockLGBMClassifier()
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        assert len(result) == 3
        assert isinstance(result, pl.DataFrame)

    def test_predict_probabilities_repeats_race_horses(self):
        """Should handle multiple horses from same race."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 8,
                "horse_id": [f"H{i:03d}" for i in range(1, 9)],
                "odds_win": [2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0],
                "popularity": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        mock_model = MockLGBMClassifier()
        feature_cols = ["odds_win", "popularity"]
        result = predict_probabilities(mock_model, df, feature_cols)

        assert len(result) == 8
        # Check that all have valid probabilities
        for col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
            assert result[col].is_null().sum() == 0  # No NaN values

    def test_predict_probabilities_with_pandas_compatible_output(self):
        """Output should be Polars DataFrame for compatibility."""
        df = pl.DataFrame({"odds_win": [2.0], "popularity": [3]})

        mock_model = MockLGBMClassifier()
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        # Must be Polars DataFrame, not Pandas
        assert isinstance(result, pl.DataFrame)
        assert not isinstance(result, type(None))

    def test_save_model_creates_readable_file(self):
        """Saved model file should be readable and loadable."""
        mock_model = MockLGBMClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            save_model(mock_model, str(model_path))

            # File should exist and have content
            assert model_path.exists()
            assert model_path.stat().st_size > 0

    def test_load_model_nonexistent_path_raises_error(self):
        """Loading nonexistent model should raise error."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")

    def test_predict_probabilities_numeric_conversion(self):
        """Should convert data to numeric for model prediction."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0],
                "popularity": [3, 5],
            }
        )

        mock_model = MockLGBMClassifier()

        # Should not raise type error
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        assert len(result) == 2

    def test_predict_probabilities_with_decimal_odds(self):
        """Should handle decimal odds format correctly."""
        df = pl.DataFrame(
            {
                "odds_win": [1.5, 2.25, 3.75],
                "popularity": [1, 3, 5],
            }
        )

        mock_model = MockLGBMClassifier()
        result = predict_probabilities(mock_model, df, ["odds_win", "popularity"])

        # Should have valid probabilities despite decimal odds
        for col in ["proba_0", "proba_1", "proba_2", "proba_3"]:
            assert (result[col] >= 0.0).all()
