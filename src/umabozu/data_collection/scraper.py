import logging
import os
import random
import re
import time
from datetime import date, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Base URLs
NETKEIBA_BASE_URL = "https://db.netkeiba.com"
RACE_RESULT_URL_FORMAT = f"{NETKEIBA_BASE_URL}/race/" + "{race_id}"
HORSE_PROFILE_URL_FORMAT = f"{NETKEIBA_BASE_URL}/horse/" + "{horse_id}"
JOCKEY_PROFILE_URL_FORMAT = f"{NETKEIBA_BASE_URL}/jockey/" + "{jockey_id}"

# Note: shutsubahyou.html seems to be the current one for upcoming races
# not shutuba.html or others.
RACE_SHUTSUBAHYOU_URL_FORMAT = (
    "https://race.netkeiba.com/race/shutuba.html"
    "?race_id={race_id}"
)

# Headers to mimic a browser visit
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def _wait_random_time():
    """Waits for a random time between 1.0 and 2.0 seconds."""
    time.sleep(random.uniform(1.0, 2.0))


def _fetch_html(url: str, encoding: str = "utf-8") -> str | None:
    """Fetches HTML content from a URL with retry and politeness."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            # Raise HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()
            response.encoding = encoding
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1} failed to fetch {url}: {e}"
            )
            _wait_random_time()
    logger.error(f"Failed to fetch {url} after {max_retries} attempts.")
    return None


def fetch_and_save_shutsubahyou_html(
    race_id: str, save_dir: str | None = None
) -> str | None:
    """Fetches shutsubahyou (race card) HTML for future races."""
    if save_dir is None:
        save_dir = os.path.join("data", "raw", "html", "upcoming_races")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{race_id}.html")

    if os.path.exists(save_path):
        logger.info(
            f"HTML for Shutsubahyou ID {race_id} already exists. Skipping fetch."
        )
        return save_path

    url = RACE_SHUTSUBAHYOU_URL_FORMAT.format(race_id=race_id)
    html_content = _fetch_html(url, encoding="utf-8")  # shutsubahyou is utf-8
    if html_content:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Successfully fetched and saved shutsubahyou for {race_id}.")
        _wait_random_time()
        return save_path
    return None


def get_race_ids_by_day(target_date: date) -> list[str]:
    """Fetches race IDs for all races on a given date from Netkeiba's race list."""
    url = (
        f"{NETKEIBA_BASE_URL}/?pid=race_list&id=c"
        f"&mode=all&start_date={target_date.strftime('%Y%m%d')}"
    )
    logger.info(f"Fetching race list from: {url}")
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    race_id_links = soup.find_all("a", href=re.compile(r"^/race/\d+$"))
    race_ids = []
    for link in race_id_links:
        match = re.search(r"/race/(\d+)", link["href"])
        if match:
            race_ids.append(match.group(1))
    return list(set(race_ids))  # Return unique race IDs


def fetch_and_save_race_html(race_id: str, save_dir: str | None = None) -> bytes | None:
    """Fetches and saves a single race result HTML."""
    if save_dir is None:
        save_dir = os.path.join("data", "raw", "html", "races")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{race_id}.html")

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return f.read()

    url = RACE_RESULT_URL_FORMAT.format(race_id=race_id)
    logger.info(f"Fetching race HTML for {race_id} from {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        # Race result pages are euc-jp/cp932
        html_content = response.content
        with open(save_path, "wb") as f:
            f.write(html_content)
        logger.info(f"Successfully fetched and saved race HTML for {race_id}.")
        _wait_random_time()
        return html_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch race HTML for {race_id}: {e}")
        return None


def fetch_and_save_horse_html(
    horse_id: str, save_dir: str | None = None
) -> bytes | None:
    """Fetches and saves a horse profile HTML."""
    if save_dir is None:
        save_dir = os.path.join("data", "raw", "html", "horses")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{horse_id}.html")

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return f.read()

    url = HORSE_PROFILE_URL_FORMAT.format(horse_id=horse_id)
    logger.info(f"Fetching horse HTML for {horse_id} from {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        # Horse profile pages are euc-jp/cp932
        html_content = response.content
        with open(save_path, "wb") as f:
            f.write(html_content)
        logger.info(f"Successfully fetched and saved horse HTML for {horse_id}.")
        _wait_random_time()
        return html_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch horse HTML for {horse_id}: {e}")
        return None


def fetch_and_save_jockey_html(
    jockey_id: str, save_dir: str | None = None
) -> bytes | None:
    """Fetches jockey profile HTML."""
    if save_dir is None:
        save_dir = os.path.join("data", "raw", "html", "jockeys")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{jockey_id}.html")

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return f.read()

    url = JOCKEY_PROFILE_URL_FORMAT.format(jockey_id=jockey_id)
    logger.info(f"Fetching jockey HTML for {jockey_id} from {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        # Jockey profile pages are euc-jp/cp932
        html_content = response.content
        with open(save_path, "wb") as f:
            f.write(html_content)
        logger.info(f"Successfully fetched and saved jockey HTML for {jockey_id}.")
        _wait_random_time()
        return html_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch jockey HTML for {jockey_id}: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Test fetching race IDs
    race_ids_today = get_race_ids_by_day(today)
    logger.info(f"Race IDs for {today}: {race_ids_today}")

    # Test fetching and saving shutsubahyou (using a mock future race ID if possible)
    # Replace with an actual upcoming race ID for live testing
    mock_upcoming_race_id = "202305040811"  # Example ID
    fetch_and_save_shutsubahyou_html(mock_upcoming_race_id)

    # Test fetching and saving a race result (use a past race ID)
    # Replace with an actual past race ID for live testing
    mock_past_race_id = "202305040811"  # Example ID
    fetch_and_save_race_html(mock_past_race_id)

    # Test fetching and saving horse profile
    mock_horse_id = "2019104085"  # Example ID
    fetch_and_save_horse_html(mock_horse_id)

    # Test fetching and saving jockey profile
    mock_jockey_id = "01180"  # Example ID
    fetch_and_save_jockey_html(mock_jockey_id)
