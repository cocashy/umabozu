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
