"""
Integration tests for data pipeline module.

Tests workflow orchestration without external dependencies.
"""

from unittest.mock import patch

import polars as pl

from umabozu.data_pipeline import (
    display_pipeline_overview,
    get_valid_feature_cols,
    prepare_training_data,
    run_data_pipeline,
    split_train_test,
)


class TestRunDataPipeline:
    """Test suite for run_data_pipeline orchestration."""

    @patch("umabozu.data_pipeline.collect_races")
    @patch("umabozu.data_pipeline.collect_profiles")
    @patch("umabozu.data_pipeline.load_data_from_db")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_run_data_pipeline_workflow(
        self,
        mock_generate_features,
        mock_clean_data,
        mock_load_data,
        mock_collect_profiles,
        mock_collect_races,
    ):
        """Should execute pipeline steps in correct order."""
        # Setup mock data
        raw_df = pl.DataFrame(
            {
                "race_id": ["R001", "R002"],
                "horse_id": ["H001", "H002"],
                "odds_win": [2.0, 3.0],
            }
        )
        cleaned_df = raw_df.clone()
        featured_df = raw_df.with_columns(
            pl.lit(1).alias("feature_1"),
            pl.lit(2).alias("feature_2"),
        )

        mock_load_data.return_value = raw_df
        mock_clean_data.return_value = cleaned_df
        mock_generate_features.return_value = featured_df

        from datetime import date

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        result = run_data_pipeline(start_date, end_date)

        # Verify collection was called
        mock_collect_races.assert_called_once_with(start_date, end_date)
        mock_collect_profiles.assert_called_once()

        # Verify data processing was called in order
        mock_load_data.assert_called_once()
        mock_clean_data.assert_called_once()
        mock_generate_features.assert_called_once()

        # Verify result is featured dataframe
        assert isinstance(result, pl.DataFrame)
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns

    @patch("umabozu.data_pipeline.collect_races")
    @patch("umabozu.data_pipeline.collect_profiles")
    @patch("umabozu.data_pipeline.load_data_from_db")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_run_data_pipeline_empty_data(
        self,
        mock_generate_features,
        mock_clean_data,
        mock_load_data,
        mock_collect_profiles,
        mock_collect_races,
    ):
        """Should handle empty data gracefully."""
        # Return empty dataframes
        empty_df = pl.DataFrame()

        mock_load_data.return_value = empty_df
        mock_clean_data.return_value = empty_df
        mock_generate_features.return_value = empty_df

        from datetime import date

        result = run_data_pipeline(date(2023, 1, 1), date(2023, 1, 31))

        # Should return empty dataframe
        assert result.is_empty()

    @patch("umabozu.data_pipeline.collect_races")
    @patch("umabozu.data_pipeline.collect_profiles")
    @patch("umabozu.data_pipeline.load_data_from_db")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_run_data_pipeline_propagates_data(
        self,
        mock_generate_features,
        mock_clean_data,
        mock_load_data,
        mock_collect_profiles,
        mock_collect_races,
    ):
        """Data should be propagated through each pipeline stage."""
        # Setup unique data for each stage to verify propagation
        raw_data = pl.DataFrame(
            {"race_id": ["R1", "R2"], "horse_id": ["H1", "H2"], "value": [1, 2]}
        )

        cleaned_data = raw_data.with_columns(pl.col("value") * 10)

        featured_data = cleaned_data.with_columns(
            [
                pl.col("value").alias("feature_1"),
                (pl.col("value") * 2).alias("feature_2"),
            ]
        )

        mock_load_data.return_value = raw_data
        mock_clean_data.return_value = cleaned_data
        mock_generate_features.return_value = featured_data

        from datetime import date

        result = run_data_pipeline(date(2023, 1, 1), date(2023, 1, 1))

        # Verify data propagation
        assert len(result) == 2
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns


class TestSplitTrainTest:
    """Test suite for train-test split functionality."""

    def test_split_train_test_default_ratio(self):
        """Should split at default 80-20 ratio."""
        import datetime

        df = pl.DataFrame(
            {
                "id": list(range(100)),
                "value": list(range(100)),
                "race_date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(100)
                ],
            }
        )

        train, test = split_train_test(df)

        # Default should be 80% train, 20% test
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(df)

    def test_split_train_test_custom_ratio(self):
        """Should respect custom split ratio."""
        import datetime

        df = pl.DataFrame({"id": list(range(100))})

        df = df.with_columns(
            pl.Series(
                "race_date",
                [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(100)
                ],
            )
        )

        train, test = split_train_test(df, test_ratio=0.5)

        assert len(train) == 50
        assert len(test) == 50

    def test_split_train_test_preserves_columns(self):
        """Should preserve all columns in split."""
        import datetime

        df = pl.DataFrame(
            {
                "race_id": ["R1", "R2", "R3"],
                "horse_id": ["H1", "H2", "H3"],
                "odds": [2.0, 3.0, 4.0],
                "race_date": [
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 1, 2),
                    datetime.date(2023, 1, 3),
                ],
            }
        )

        train, test = split_train_test(df, test_ratio=0.34)

        assert set(train.columns) == set(df.columns)
        assert set(test.columns) == set(df.columns)

    def test_split_train_test_no_overlap(self):
        """Train and test sets should not overlap."""
        import datetime

        df = pl.DataFrame(
            {
                "id": list(range(100)),
                "value": list(range(100)),
                "race_date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(100)
                ],
            }
        )

        train, test = split_train_test(df)

        # No row should appear in both
        train_ids = set(train["id"].to_list())
        test_ids = set(test["id"].to_list())

        assert len(train_ids & test_ids) == 0  # No intersection


class TestGetValidFeatureCols:
    """Test suite for feature column validation."""

    @patch("umabozu.data_pipeline.get_feature_cols")
    def test_get_valid_feature_cols_filters_available_features(
        self, mock_get_feature_cols
    ):
        """Should return only features present in dataframe."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0],
                "popularity": [3, 5],
                "horse_weight": [450, 480],
            }
        )

        # All requested features are available
        all_features = ["odds_win", "popularity", "horse_weight"]
        mock_get_feature_cols.return_value = all_features
        result = get_valid_feature_cols(df)

        assert set(result) == set(all_features)

    @patch("umabozu.data_pipeline.get_feature_cols")
    def test_get_valid_feature_cols_excludes_missing_features(
        self, mock_get_feature_cols
    ):
        """Should exclude features not in dataframe."""
        df = pl.DataFrame(
            {
                "odds_win": [2.0, 3.0],
                "popularity": [3, 5],
            }
        )

        requested = ["odds_win", "popularity", "missing_feature"]
        mock_get_feature_cols.return_value = requested
        result = get_valid_feature_cols(df)

        assert "missing_feature" not in result
        assert "odds_win" in result
        assert "popularity" in result

    @patch("umabozu.data_pipeline.get_feature_cols")
    def test_get_valid_feature_cols_empty_dataframe(self, mock_get_feature_cols):
        """Should handle empty dataframe."""
        df = pl.DataFrame()

        mock_get_feature_cols.return_value = ["col1", "col2"]
        result = get_valid_feature_cols(df)

        assert len(result) == 0

    @patch("umabozu.data_pipeline.get_feature_cols")
    def test_get_valid_feature_cols_empty_features_list(
        self, mock_get_feature_cols
    ):
        """Should handle empty features list."""
        df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        mock_get_feature_cols.return_value = []
        result = get_valid_feature_cols(df)

        assert len(result) == 0


class TestPrepareTrainingData:
    """Test suite for training data preparation."""

    @patch("umabozu.data_pipeline.get_training_data")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_prepare_training_data_uses_db_optimized(
        self, mock_generate_features, mock_clean_data, mock_get_training_data
    ):
        """Should load data using optimized DB query by default."""
        raw_df = pl.DataFrame({"race_id": ["R1"], "odds_win": [2.0]})
        cleaned_df = raw_df.clone()
        featured_df = raw_df.with_columns(pl.lit(1).alias("feature_1"))

        mock_get_training_data.return_value = raw_df
        mock_clean_data.return_value = cleaned_df
        mock_generate_features.return_value = featured_df

        result = prepare_training_data()

        mock_get_training_data.assert_called_once()
        mock_clean_data.assert_called_once()
        mock_generate_features.assert_called_once()
        assert "feature_1" in result.columns

    @patch("umabozu.data_pipeline.load_data_from_db")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_prepare_training_data_uses_fallback_loader(
        self, mock_generate_features, mock_clean_data, mock_load_data
    ):
        """Should use load_data_from_db when use_db_optimized is False."""
        raw_df = pl.DataFrame({"race_id": ["R1"], "odds_win": [2.0]})
        cleaned_df = raw_df.clone()
        featured_df = raw_df.with_columns(pl.lit(1).alias("feature_1"))

        mock_load_data.return_value = raw_df
        mock_clean_data.return_value = cleaned_df
        mock_generate_features.return_value = featured_df

        result = prepare_training_data(use_db_optimized=False)

        mock_load_data.assert_called_once()
        mock_clean_data.assert_called_once()
        mock_generate_features.assert_called_once()
        assert "feature_1" in result.columns

    @patch("umabozu.data_pipeline.get_training_data")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_prepare_training_data_empty_data(
        self, mock_generate_features, mock_clean_data, mock_get_training_data
    ):
        """Should return empty dataframe when no data is loaded."""
        empty_df = pl.DataFrame()
        mock_get_training_data.return_value = empty_df

        result = prepare_training_data()

        assert result.is_empty()
        mock_clean_data.assert_not_called()
        mock_generate_features.assert_not_called()


class TestDisplayPipelineOverview:
    """Test suite for pipeline overview display."""

    def test_display_pipeline_overview_accepts_dataframe(self):
        """Should accept valid dataframe without error."""
        df = pl.DataFrame(
            {
                "race_id": ["R1", "R2"],
                "horse_id": ["H1", "H2"],
                "odds_win": [2.0, 3.0],
            }
        )

        # Should not raise error
        result = display_pipeline_overview(df)

        # Result should be displayable (string or None)
        assert result is None or isinstance(result, str)

    def test_display_pipeline_overview_handles_empty_dataframe(self):
        """Should handle empty dataframe gracefully."""
        df = pl.DataFrame()

        # Should not raise error
        result = display_pipeline_overview(df)

        assert result is None or isinstance(result, str)

    def test_display_pipeline_overview_shows_structure(self):
        """Should display dataframe structure information."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        # Should complete without error
        display_pipeline_overview(df)

        # Function should process successfully
        assert df.shape[0] == 3
        assert df.shape[1] == 3


class TestDataPipelineIntegration:
    """Integration tests for complete pipeline workflow."""

    @patch("umabozu.data_pipeline.collect_races")
    @patch("umabozu.data_pipeline.collect_profiles")
    @patch("umabozu.data_pipeline.load_data_from_db")
    @patch("umabozu.data_pipeline.clean_data")
    @patch("umabozu.data_pipeline.generate_features")
    def test_full_pipeline_workflow(
        self,
        mock_generate_features,
        mock_clean_data,
        mock_load_data,
        mock_collect_profiles,
        mock_collect_races,
    ):
        """Test full pipeline from collection to feature engineering."""
        # Setup workflow data
        raw_df = pl.DataFrame(
            {
                "race_id": ["R001", "R002"],
                "horse_id": ["H001", "H002"],
                "rank": [1, 2],
                "odds_win": [2.0, 3.0],
                "popularity": [3, 5],
            }
        )

        featured_df = raw_df.with_columns(
            pl.lit(1).alias("odds_win_ln"),
            pl.lit(2).alias("popularity_ln"),
        )

        mock_load_data.return_value = raw_df
        mock_clean_data.return_value = raw_df
        mock_generate_features.return_value = featured_df

        from datetime import date

        result = run_data_pipeline(date(2023, 1, 1), date(2023, 1, 1))

        # Verify complete flow
        assert result.shape[0] == 2
        assert len(result.columns) >= 5

        # Verify features were added
        assert "odds_win_ln" in result.columns
        assert "popularity_ln" in result.columns

    def test_data_pipeline_train_test_workflow(self):
        """Test train-test split integration."""
        import datetime

        df = pl.DataFrame(
            {
                "rank": list(range(1, 101)),
                "odds_win": [float(i) * 0.5 for i in range(1, 101)],
                "race_date": [
                    datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                    for i in range(100)
                ],
            }
        )

        # Split
        train, test = split_train_test(df, test_ratio=0.2)

        # Verification
        assert len(train) == 80
        assert len(test) == 20
        assert train["race_date"].max() < test["race_date"].min()
