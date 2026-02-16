import datetime
import logging
import time

from tqdm import tqdm

from umabozu.data_collection.html_parser import (
    parse_horse_profile_html,
    parse_jockey_profile_html,
    save_parsed_data_to_db,
)
from umabozu.data_collection.scraper import (
    fetch_and_save_horse_html,
    fetch_and_save_jockey_html,
    fetch_and_save_race_html,
    get_race_ids_by_day,
)
from umabozu.database.crud import (
    delete_staged_race_id,
    get_db_connection,
    get_staged_race_ids,
    insert_staged_race_id,
)

logger = logging.getLogger(__name__)


def collect_races(start_date: datetime.date, end_date: datetime.date):
    """Collects race data for a given date range."""
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Collecting races for {current_date}...")
        race_ids = get_race_ids_by_day(current_date)
        for race_id in tqdm(race_ids, desc=f"Races on {current_date}"):
            insert_staged_race_id(race_id)
        current_date += datetime.timedelta(days=1)
        time.sleep(1)  # Be polite

    staged_race_ids = get_staged_race_ids()
    for race_id in tqdm(staged_race_ids, desc="Fetching race details"):
        html_content = fetch_and_save_race_html(race_id)
        if html_content:
            save_parsed_data_to_db(race_id, html_content)
            delete_staged_race_id(race_id)


def collect_profiles():
    """Collects missing horse and jockey profiles."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Collect missing horse profiles
    cursor.execute(
        "SELECT DISTINCT horse_id FROM results WHERE horse_id NOT IN "
        "(SELECT horse_id FROM horses WHERE horse_name != 'UNKNOWN')"
    )
    missing_horse_ids = [row[0] for row in cursor.fetchall()]
    for horse_id in tqdm(missing_horse_ids, desc="Fetching horse profiles"):
        html_path = fetch_and_save_horse_html(horse_id)
        if html_path:
            with open(html_path, "rb") as f:
                horse_data = parse_horse_profile_html(horse_id, f.read())
                # insert_horse(horse_data) # This is already handled by app.py
        time.sleep(1)

    # Collect missing jockey profiles
    cursor.execute(
        "SELECT DISTINCT jockey_id FROM results WHERE jockey_id NOT IN "
        "(SELECT jockey_id FROM jockeys WHERE jockey_name != 'UNKNOWN')"
    )
    missing_jockey_ids = [row[0] for row in cursor.fetchall()]
    for jockey_id in tqdm(missing_jockey_ids, desc="Fetching jockey profiles"):
        html_path = fetch_and_save_jockey_html(jockey_id)
        if html_path:
            with open(html_path, "rb") as f:
                jockey_data = parse_jockey_profile_html(jockey_id, f.read())
                # insert_jockey(jockey_data) # This is already handled by app.py
        time.sleep(1)

    conn.close()
