import os
from collections.abc import Callable
from datetime import date
from typing import Any
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest
import requests

from umabozu.data_collection.scraper import (
    fetch_and_save_horse_html,
    fetch_and_save_jockey_html,
    fetch_and_save_race_html,
    fetch_and_save_shutsubahyou_html,
    get_race_ids_by_day,
)


# Mock the requests.get to prevent actual HTTP requests
class MockResponse:
    def __init__(self, text, status_code, encoding="utf-8", content=b""):
        self.text = text
        self.status_code = status_code
        self.encoding = encoding
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            # Create a mock requests.Response object for the HTTPError
            # This is a minimal mock to satisfy type checkers
            mock_req_response = MagicMock(spec=requests.Response)
            mock_req_response.status_code = self.status_code
            # Add any other attributes HTTPError might use
            mock_req_response.reason = "Mock Reason"

            raise requests.exceptions.HTTPError(
                f"HTTP Error: {self.status_code}", response=mock_req_response
            )


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_wait_random_time():
    with patch("umabozu.data_collection.scraper._wait_random_time") as mock_wait:
        yield mock_wait


@pytest.fixture
def mock_os_makedirs():
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mock_open_file():
    with patch("builtins.open", mock_open()) as mock_file:
        yield mock_file


# Test for fetch_and_save_shutsubahyou_html
def test_fetch_and_save_shutsubahyou_html_success(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("<html>Shutsubahyou HTML</html>", 200)
    race_id = "123456789012"
    save_dir = "test_data/upcoming_races"

    result = fetch_and_save_shutsubahyou_html(race_id, save_dir)

    expected_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    mock_requests_get.assert_called_once_with(expected_url, headers=ANY, timeout=10)
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_called_once_with(
        os.path.join(save_dir, f"{race_id}.html"), "w", encoding="utf-8"
    )
    mock_open_file().write.assert_called_once_with("<html>Shutsubahyou HTML</html>")
    mock_wait_random_time.assert_called_once()
    assert result == os.path.join(save_dir, f"{race_id}.html")


def test_fetch_and_save_shutsubahyou_html_already_exists(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    race_id = "123456789012"
    save_dir = "test_data/upcoming_races"
    expected_path = os.path.join(save_dir, f"{race_id}.html")

    with patch("os.path.exists", return_value=True):
        result = fetch_and_save_shutsubahyou_html(race_id, save_dir)
        mock_requests_get.assert_not_called()
        mock_wait_random_time.assert_not_called()
        mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_open_file.assert_not_called()
        assert result == expected_path


def test_fetch_and_save_shutsubahyou_html_fetch_fails(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 500)
    mock_requests_get.return_value.raise_for_status: Callable[..., Any] = MagicMock(
        side_effect=requests.exceptions.RequestException("Test error")
    )

    race_id = "123456789012"
    save_dir = "test_data/upcoming_races"

    result = fetch_and_save_shutsubahyou_html(race_id, save_dir)

    assert mock_requests_get.call_count == 3  # Retry mechanism
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_not_called()
    assert mock_wait_random_time.call_count == 3
    assert result is None


# Test for get_race_ids_by_day
def test_get_race_ids_by_day_success(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    # Mock HTML content with race IDs
    dummy_html = """
    <html>
        <body>
            <a href="/race/202301010101">Race 1</a>
            <a href="/race/202301010102">Race 2</a>
            <a href="/other/link">Other Link</a>
            <a href="/race/202301010101">Duplicate Race 1</a>
        </body>
    </html>
    """
    mock_requests_get.return_value = MockResponse(dummy_html, 200, encoding="euc-jp")
    mock_requests_get.return_value.content = dummy_html.encode("euc-jp")

    target_date = date(2023, 1, 1)
    date_str = target_date.strftime("%Y%m%d")
    save_dir = os.path.join("data", "raw", "html", "race_lists")
    save_path = os.path.join(save_dir, f"{date_str}.html")

    result = get_race_ids_by_day(target_date)

    expected_url = f"https://db.netkeiba.com/race/list/{date_str}/"
    mock_requests_get.assert_called_once_with(expected_url, headers=ANY, timeout=10)
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_called_once_with(save_path, "w", encoding="utf-8")
    mock_open_file().write.assert_called_once_with(dummy_html)
    mock_wait_random_time.assert_called_once()
    assert sorted(result) == sorted(["202301010101", "202301010102"])


def test_get_race_ids_by_day_from_cache(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    target_date = date(2023, 1, 1)
    date_str = target_date.strftime("%Y%m%d")
    save_dir = os.path.join("data", "raw", "html", "race_lists")
    save_path = os.path.join(save_dir, f"{date_str}.html")

    cached_html = """
    <html>
        <body>
            <a href="/race/202301010103">Race 3</a>
            <a href="/race/202301010104">Race 4</a>
        </body>
    </html>
    """

    # Mock os.path.exists to return True, indicating a cached file
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cached_html)) as mock_open_for_read,
    ):
        result = get_race_ids_by_day(target_date)

        mock_requests_get.assert_not_called()
        mock_wait_random_time.assert_not_called()
        mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_open_for_read.assert_called_once_with(save_path, encoding="utf-8")
        assert sorted(result) == sorted(["202301010103", "202301010104"])


def test_get_race_ids_by_day_fetch_fails(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 500)
    mock_requests_get.return_value.raise_for_status: Callable[..., Any] = MagicMock(
        side_effect=requests.exceptions.RequestException("Test error")
    )

    target_date = date(2023, 1, 1)
    result = get_race_ids_by_day(target_date)

    assert mock_requests_get.call_count == 3  # Retry mechanism
    mock_os_makedirs.assert_called_once()
    mock_open_file.assert_not_called()
    assert result == []


# Test for fetch_and_save_race_html
def test_fetch_and_save_race_html_success(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 200, content=b"Race Result HTML")
    race_id = "123456789012"
    save_dir = "test_data/races"

    result = fetch_and_save_race_html(race_id, save_dir)

    expected_url = f"https://db.netkeiba.com/race/{race_id}"
    mock_requests_get.assert_called_once_with(expected_url, headers=ANY, timeout=10)
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_called_once_with(
        os.path.join(save_dir, f"{race_id}.html"), "wb"
    )
    mock_open_file().write.assert_called_once_with(b"Race Result HTML")
    mock_wait_random_time.assert_called_once()
    assert result == b"Race Result HTML"


def test_fetch_and_save_race_html_already_exists(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    race_id = "123456789012"
    save_dir = "test_data/races"
    expected_path = os.path.join(save_dir, f"{race_id}.html")

    with (
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open", mock_open(read_data=b"Cached Race Result HTML")
        ) as mock_open_for_read,
    ):
        result = fetch_and_save_race_html(race_id, save_dir)
        mock_requests_get.assert_not_called()
        mock_wait_random_time.assert_not_called()
        mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_open_for_read.assert_called_once_with(expected_path, "rb")
        assert result == b"Cached Race Result HTML"


def test_fetch_and_save_race_html_fetch_fails(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 500)
    mock_requests_get.return_value.raise_for_status: Callable[..., Any] = MagicMock(
        side_effect=requests.exceptions.RequestException("Test error")
    )

    race_id = "123456789012"
    save_dir = "test_data/races"

    result = fetch_and_save_race_html(race_id, save_dir)

    assert mock_requests_get.call_count == 1
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_not_called()
    assert result is None


# Test for fetch_and_save_horse_html
def test_fetch_and_save_horse_html_success(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse(
        "", 200, content=b"Horse Profile HTML"
    )
    horse_id = "1234567890"
    save_dir = "test_data/horses"

    result = fetch_and_save_horse_html(horse_id, save_dir)

    expected_url = f"https://db.netkeiba.com/horse/{horse_id}"
    mock_requests_get.assert_called_once_with(expected_url, headers=ANY, timeout=10)
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_called_once_with(
        os.path.join(save_dir, f"{horse_id}.html"), "wb"
    )
    mock_open_file().write.assert_called_once_with(b"Horse Profile HTML")
    mock_wait_random_time.assert_called_once()
    assert result == b"Horse Profile HTML"


def test_fetch_and_save_horse_html_already_exists(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    horse_id = "1234567890"
    save_dir = "test_data/horses"
    expected_path = os.path.join(save_dir, f"{horse_id}.html")

    with (
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open", mock_open(read_data=b"Cached Horse Profile HTML")
        ) as mock_open_for_read,
    ):
        result = fetch_and_save_horse_html(horse_id, save_dir)
        mock_requests_get.assert_not_called()
        mock_wait_random_time.assert_not_called()
        mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_open_for_read.assert_called_once_with(expected_path, "rb")
        assert result == b"Cached Horse Profile HTML"


def test_fetch_and_save_horse_html_fetch_fails(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 500)
    mock_requests_get.return_value.raise_for_status: Callable[..., Any] = MagicMock(
        side_effect=requests.exceptions.RequestException("Test error")
    )

    horse_id = "1234567890"
    save_dir = "test_data/horses"

    result = fetch_and_save_horse_html(horse_id, save_dir)

    assert mock_requests_get.call_count == 1
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_not_called()
    assert result is None


# Test for fetch_and_save_jockey_html
def test_fetch_and_save_jockey_html_success(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse(
        "", 200, content=b"Jockey Profile HTML"
    )
    jockey_id = "00001"
    save_dir = "test_data/jockeys"

    result = fetch_and_save_jockey_html(jockey_id, save_dir)

    expected_url = f"https://db.netkeiba.com/jockey/{jockey_id}"
    mock_requests_get.assert_called_once_with(expected_url, headers=ANY, timeout=10)
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_called_once_with(
        os.path.join(save_dir, f"{jockey_id}.html"), "wb"
    )
    mock_open_file().write.assert_called_once_with(b"Jockey Profile HTML")
    mock_wait_random_time.assert_called_once()
    assert result == b"Jockey Profile HTML"


def test_fetch_and_save_jockey_html_already_exists(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    jockey_id = "00001"
    save_dir = "test_data/jockeys"
    expected_path = os.path.join(save_dir, f"{jockey_id}.html")

    with (
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open", mock_open(read_data=b"Cached Jockey Profile HTML")
        ) as mock_open_for_read,
    ):
        result = fetch_and_save_jockey_html(jockey_id, save_dir)
        mock_requests_get.assert_not_called()
        mock_wait_random_time.assert_not_called()
        mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_open_for_read.assert_called_once_with(expected_path, "rb")
        assert result == b"Cached Jockey Profile HTML"


def test_fetch_and_save_jockey_html_fetch_fails(
    mock_requests_get, mock_wait_random_time, mock_os_makedirs, mock_open_file
):
    mock_requests_get.return_value = MockResponse("", 500)
    mock_requests_get.return_value.raise_for_status: Callable[..., Any] = MagicMock(
        side_effect=requests.exceptions.RequestException("Test error")
    )

    jockey_id = "00001"
    save_dir = "test_data/jockeys"

    result = fetch_and_save_jockey_html(jockey_id, save_dir)

    assert mock_requests_get.call_count == 1
    mock_os_makedirs.assert_called_once_with(save_dir, exist_ok=True)
    mock_open_file.assert_not_called()
    assert result is None
