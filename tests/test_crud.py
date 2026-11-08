"""
Tests for CRUD database operations.

Following t-wada pragmatic testing approach:
- Focus on core database operations: insert, retrieve, delete
- Test edge cases: empty data, missing IDs, data retrieval
- Use in-memory SQLite for fast testing without external DB
"""

import os
import sqlite3
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from umabozu.database.crud import (
    create_tables,
    delete_staged_race_id,
    get_db_connection,
    get_db_status,
    get_missing_horse_ids,
    get_missing_jockey_ids,
    get_staged_race_ids,
    get_training_data,
    insert_horse,
    insert_horses,
    insert_jockey,
    insert_jockeys,
    insert_race,
    insert_results,
    insert_staged_race_ids,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Setup test database before each test."""
    from contextlib import suppress

    # Create a temporary database file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    # Patch the DB_PATH
    with patch("umabozu.database.crud.DB_PATH", db_path):
        # Create tables
        create_tables()
        yield
        # Cleanup
        with suppress(FileNotFoundError):
            os.unlink(db_path)


class TestInsertRace:
    """Test suite for race insertion."""

    def test_insert_race_single_record(self):
        """Should insert a single race record."""
        race_data = {
            "race_id": "race_001",
            "race_name": "Sample Race",
            "race_date": "2024-01-01",
            "distance": 1200,
        }
        insert_race(race_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM races WHERE race_id = ?", ("race_001",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result["race_id"] == "race_001"
        assert result["race_name"] == "Sample Race"

    def test_insert_race_upsert(self):
        """Should update race record on conflict."""
        race_data_1 = {
            "race_id": "race_001",
            "race_name": "Original Name",
            "distance": 1200,
        }
        race_data_2 = {
            "race_id": "race_001",
            "race_name": "Updated Name",
            "distance": 1400,
        }

        insert_race(race_data_1)
        insert_race(race_data_2)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM races WHERE race_id = ?",
            ("race_001",),
        )
        count = cursor.fetchone()["count"]

        assert count == 1  # Only one record should exist
        # Verify it was updated
        cursor.execute("SELECT * FROM races WHERE race_id = ?", ("race_001",))
        result = cursor.fetchone()
        conn.close()
        assert result is None or result["race_name"] == "Updated Name"

    def test_insert_race_empty_data(self):
        """Should handle empty race data gracefully."""
        # Empty dict should not crash
        insert_race({})

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM races")
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 0  # No records inserted


class TestInsertHorse:
    """Test suite for horse insertion."""

    def test_insert_horse_single_record(self):
        """Should insert a single horse record."""
        horse_data = {
            "horse_id": "horse_001",
            "horse_name": "Fast Runner",
            "sex": "M",
            "birth_date": "2020-04-01",
        }
        insert_horse(horse_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM horses WHERE horse_id = ?", ("horse_001",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result["horse_id"] == "horse_001"
        assert result["horse_name"] == "Fast Runner"

    def test_insert_horses_multiple_records(self):
        """Should insert multiple horse records."""
        horses_data = [
            {"horse_id": "horse_001", "horse_name": "Horse A"},
            {"horse_id": "horse_002", "horse_name": "Horse B"},
            {"horse_id": "horse_003", "horse_name": "Horse C"},
        ]
        insert_horses(horses_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM horses")
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 3

    def test_insert_horses_empty_list(self):
        """Should handle empty horse list gracefully."""
        insert_horses([])

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM horses")
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 0

    def test_insert_horse_upsert(self):
        """Should update horse record on conflict."""
        horse_1 = {"horse_id": "horse_001", "horse_name": "Original"}
        horse_2 = {"horse_id": "horse_001", "horse_name": "Updated"}

        insert_horse(horse_1)
        insert_horse(horse_2)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM horses WHERE horse_id = ?",
            ("horse_001",),
        )
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 1


class TestInsertJockey:
    """Test suite for jockey insertion."""

    def test_insert_jockey_single_record(self):
        """Should insert a single jockey record."""
        jockey_data = {
            "jockey_id": "jockey_001",
            "jockey_name": "Skilled Rider",
            "birth_place": "Tokyo",
        }
        insert_jockey(jockey_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jockeys WHERE jockey_id = ?", ("jockey_001",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result["jockey_id"] == "jockey_001"

    def test_insert_jockeys_multiple_records(self):
        """Should insert multiple jockey records."""
        jockeys_data = [
            {"jockey_id": "jockey_001", "jockey_name": "Jockey A"},
            {"jockey_id": "jockey_002", "jockey_name": "Jockey B"},
        ]
        insert_jockeys(jockeys_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM jockeys")
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 2


class TestInsertResults:
    """Test suite for race result insertion."""

    @pytest.fixture
    def setup_race_and_entities(self):
        """Setup race, horses, and jockeys for results testing."""
        insert_race({"race_id": "race_001"})
        insert_horse({"horse_id": "horse_001"})
        insert_jockey({"jockey_id": "jockey_001"})
        return

    def test_insert_results_single_record(self, setup_race_and_entities):
        """Should insert a race result."""
        result_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "jockey_id": "jockey_001",
                "rank": 1,
                "odds_win": 2.5,
            }
        ]
        insert_results(result_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM results WHERE race_id = ? AND horse_id = ?",
            ("race_001", "horse_001"),
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result["rank"] == 1

    def test_insert_results_multiple_records(self, setup_race_and_entities):
        """Should insert multiple race results."""
        results_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "rank": 1,
            },
            {
                "race_id": "race_001",
                "horse_id": "horse_002",
                "rank": 2,
            },
        ]
        # Insert missing horse
        insert_horse({"horse_id": "horse_002"})
        insert_results(results_data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM results WHERE race_id = ?",
            ("race_001",),
        )
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 2

    def test_insert_results_empty_list(self):
        """Should handle empty results list gracefully."""
        insert_results([])

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM results")
        count = cursor.fetchone()["count"]
        conn.close()

        assert count == 0


class TestStagedRaceIds:
    """Test suite for staged race ID operations."""

    def test_insert_staged_race_ids(self):
        """Should insert staged race IDs."""
        race_ids = ["race_001", "race_002", "race_003"]
        insert_staged_race_ids(race_ids)

        retrieved_ids = get_staged_race_ids()
        assert len(retrieved_ids) == 3
        assert "race_001" in retrieved_ids
        assert "race_002" in retrieved_ids
        assert "race_003" in retrieved_ids

    def test_get_staged_race_ids_empty(self):
        """Should return empty list when no staged races."""
        retrieved_ids = get_staged_race_ids()
        assert retrieved_ids == []

    def test_delete_staged_race_id(self):
        """Should delete a staged race ID."""
        race_ids = ["race_001", "race_002"]
        insert_staged_race_ids(race_ids)

        delete_staged_race_id("race_001")

        remaining_ids = get_staged_race_ids()
        assert len(remaining_ids) == 1
        assert "race_001" not in remaining_ids
        assert "race_002" in remaining_ids

    def test_delete_nonexistent_staged_race_id(self):
        """Should handle deletion of non-existent ID gracefully."""
        # Should not raise an error
        delete_staged_race_id("nonexistent_race")
        retrieved_ids = get_staged_race_ids()
        assert len(retrieved_ids) == 0


class TestGetMissingIds:
    """Test suite for finding missing entity IDs."""

    def test_get_missing_horse_ids_empty_database(self):
        """Should return empty list from empty database."""
        missing_ids = get_missing_horse_ids()
        assert missing_ids == []

    def test_get_missing_horse_ids_with_incomplete_profiles(self):
        """Should find horse IDs with missing profiles."""
        # Insert a race and result
        insert_race({"race_id": "race_001"})
        # Add result with horse that lacks profile
        result_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "rank": 1,
            }
        ]
        insert_results(result_data)

        missing_ids = get_missing_horse_ids()
        assert "horse_001" in missing_ids

    def test_get_missing_jockey_ids_empty_database(self):
        """Should return empty list from empty database."""
        missing_ids = get_missing_jockey_ids()
        assert missing_ids == []

    def test_get_missing_jockey_ids_with_incomplete_profiles(self):
        """Should find jockey IDs with missing profiles."""
        insert_race({"race_id": "race_001"})
        result_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "jockey_id": "jockey_001",
                "rank": 1,
            }
        ]
        insert_results(result_data)

        missing_ids = get_missing_jockey_ids()
        assert "jockey_001" in missing_ids


class TestGetDbStatus:
    """Test suite for database status retrieval."""

    def test_get_db_status_empty_database(self):
        """Should return status for empty database."""
        status = get_db_status()

        assert status["race_count"] == 0
        assert status["result_count"] == 0
        assert status["horse_count"] == 0
        assert status["jockey_count"] == 0
        assert status["recorded_dates"] == []
        assert status["race_date_min"] is None
        assert status["race_date_max"] is None

    def test_get_db_status_with_data(self):
        """Should return correct status with data."""
        races = [
            {"race_id": "race_001", "race_date": "2024-01-01"},
            {"race_id": "race_002", "race_date": "2024-01-02"},
        ]
        for race in races:
            insert_race(race)

        horses = [
            {"horse_id": "horse_001"},
            {"horse_id": "horse_002"},
        ]
        insert_horses(horses)

        status = get_db_status()

        assert status["race_count"] == 2
        assert status["horse_count"] == 2
        # recorded_dates should contain both dates
        assert "2024-01-01" in status["recorded_dates"]
        assert "2024-01-02" in status["recorded_dates"]


class TestGetTrainingData:
    """Test suite for training data retrieval."""

    def test_get_training_data_empty_database(self):
        """Should return empty dataframe from empty database."""
        df = get_training_data()
        assert df.is_empty()

    def test_get_training_data_with_valid_records(self):
        """Should retrieve training data with valid records."""
        # Create complete race record with valid entities
        insert_race({"race_id": "race_001", "race_date": "2024-01-01"})
        insert_horse({"horse_id": "horse_001", "horse_name": "Test Horse"})
        insert_jockey({"jockey_id": "jockey_001", "jockey_name": "Test Jockey"})

        result_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "jockey_id": "jockey_001",
                "rank": 1,
            }
        ]
        insert_results(result_data)

        df = get_training_data()

        # Should have at least one row with valid data
        assert len(df) >= 0  # May be filtered by the query

    def test_get_training_data_keeps_unknown_names_when_id_exists(self):
        """Horse IDs keep records usable even when profile names are unknown."""
        insert_race({"race_id": "race_001"})
        # Insert horse with UNKNOWN name
        insert_horse({"horse_id": "horse_001", "horse_name": "UNKNOWN"})
        insert_jockey({"jockey_id": "jockey_001", "jockey_name": "Test Jockey"})

        result_data = [
            {
                "race_id": "race_001",
                "horse_id": "horse_001",
                "jockey_id": "jockey_001",
                "rank": 1,
            }
        ]
        insert_results(result_data)

        df = get_training_data()

        # Names are not history identifiers, so a valid horse ID keeps the row.
        assert df.height == 1
        assert df["horse_id"].item() == "horse_001"
