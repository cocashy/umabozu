"""
Integration tests for batch collector module.

Tests data collection workflow using mocked external dependencies.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from umabozu.data_collection.batch_collector import collect_profiles, collect_races


class TestCollectRaces:
    """Test suite for collect_races function."""

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_empty_date_range(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Should handle date range with no races."""
        # No races found for any day
        mock_get_ids.return_value = []
        mock_get_staged.return_value = []

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        # Should not raise error
        collect_races(start_date, end_date)

        # Should have tried to get race IDs for each day
        assert mock_get_ids.call_count == 3

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_with_race_ids(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Should collect races found on target dates."""
        # First call returns races, second/third return empty
        race_ids = ["202301010101", "202301010102"]
        mock_get_ids.side_effect = [race_ids, [], []]
        mock_get_staged.return_value = race_ids
        mock_fetch.return_value = "<html>dummy</html>"

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        collect_races(start_date, end_date)

        # Should have inserted staged race IDs
        mock_insert_staged.assert_called()

        # Should have fetched race HTML
        assert mock_fetch.call_count == 2

        # Should have deleted staged IDs after processing
        assert mock_delete.call_count == 2

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_handles_fetch_failure(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Should handle fetch failures gracefully."""
        race_ids = ["202301010101"]
        mock_get_ids.return_value = race_ids
        mock_get_staged.return_value = race_ids
        # Fetch returns None to simulate failure
        mock_fetch.return_value = None

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 1)

        # Should not raise error
        collect_races(start_date, end_date)

        # Should not have saved data if fetch failed
        mock_save_parsed.assert_not_called()

        # Should not delete staged ID when fetch fails
        mock_delete.assert_not_called()

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_single_day(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Should handle single day collection."""
        race_ids = ["202301010101", "202301010102", "202301010103"]
        mock_get_ids.return_value = race_ids
        mock_get_staged.return_value = race_ids
        mock_fetch.return_value = "<html>race data</html>"

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 1)

        collect_races(start_date, end_date)

        # Should get race IDs for one day
        assert mock_get_ids.call_count == 1

        # Should insert all found race IDs
        mock_insert_staged.assert_called_with(race_ids)

        # Should process all races
        assert mock_fetch.call_count == 3


class TestCollectProfiles:
    """Test suite for collect_profiles function."""

    @patch("umabozu.data_collection.batch_collector.get_db_connection")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_horse_html")
    @patch("umabozu.data_collection.batch_collector.parse_horse_profile_html")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_jockey_html")
    @patch("umabozu.data_collection.batch_collector.parse_jockey_profile_html")
    @patch("umabozu.data_collection.batch_collector.insert_horses")
    @patch("umabozu.data_collection.batch_collector.insert_jockeys")
    def test_collect_profiles_workflow(
        self,
        mock_insert_jockeys,
        mock_insert_horses,
        mock_parse_jockey,
        mock_fetch_jockey,
        mock_parse_horse,
        mock_fetch_horse,
        mock_get_staged,
        mock_get_conn,
    ):
        """Should collect horse and jockey profiles."""
        mock_get_conn.return_value = MagicMock()

        # Mock database queries to return race data
        mock_get_staged.return_value = []

        # Should not raise error
        collect_profiles()

        # Should have attempted connection
        mock_get_conn.assert_called()


class TestDataCollectionIntegration:
    """Integration tests for complete data collection workflow."""

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_multi_day_workflow(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Test realistic multi-day race collection."""
        # Day 1: 2 races
        # Day 2: 3 races
        # Day 3: 0 races
        race_ids_day1 = ["20230101_001", "20230101_002"]
        race_ids_day2 = ["20230102_001", "20230102_002", "20230102_003"]

        mock_get_ids.side_effect = [race_ids_day1, race_ids_day2, []]
        mock_get_staged.return_value = race_ids_day1 + race_ids_day2
        mock_fetch.return_value = "<html>race</html>"

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 3)

        collect_races(start_date, end_date)

        # Should insert staged IDs twice (days with races)
        assert mock_insert_staged.call_count == 2

        # Should have processed 5 races total
        assert mock_fetch.call_count == 5
        assert mock_delete.call_count == 5

    @patch("umabozu.data_collection.batch_collector.get_race_ids_by_day")
    @patch("umabozu.data_collection.batch_collector.insert_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.get_staged_race_ids")
    @patch("umabozu.data_collection.batch_collector.fetch_and_save_race_html")
    @patch("umabozu.data_collection.batch_collector.save_parsed_data_to_db")
    @patch("umabozu.data_collection.batch_collector.delete_staged_race_id")
    def test_collect_races_partial_failures(
        self,
        mock_delete,
        mock_save_parsed,
        mock_fetch,
        mock_get_staged,
        mock_insert_staged,
        mock_get_ids,
    ):
        """Should continue processing despite some fetch failures."""
        race_ids = ["race1", "race2", "race3"]
        mock_get_ids.return_value = race_ids
        mock_get_staged.return_value = race_ids

        # Some fetches fail (return None), some succeed
        mock_fetch.side_effect = [
            "<html>race1</html>",
            None,  # Failure
            "<html>race3</html>",
        ]

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 1)

        collect_races(start_date, end_date)

        # Should have attempted to save only successful races
        assert mock_save_parsed.call_count == 2

        # Should clean up only successful races
        assert mock_delete.call_count == 2
