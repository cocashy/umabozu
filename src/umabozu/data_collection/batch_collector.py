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
    insert_horses,
    insert_jockeys,
    insert_staged_race_ids,
)

logger = logging.getLogger(__name__)


def collect_races(start_date: datetime.date, end_date: datetime.date):
    """Collects race data for a given date range."""
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Collecting races for {current_date}...")
        race_ids = get_race_ids_by_day(current_date)
        logger.info(f"Found {len(race_ids)} races for {current_date}.")
        if race_ids:
            insert_staged_race_ids(race_ids)
        current_date += datetime.timedelta(days=1)

    staged_race_ids = get_staged_race_ids()
    for i, race_id in enumerate(tqdm(staged_race_ids, desc="Fetching race details")):
        html_content = fetch_and_save_race_html(race_id)
        if html_content:
            save_parsed_data_to_db(race_id, html_content)
            delete_staged_race_id(race_id)
        if i % 5 == 0:  # Be polite every few requests
            time.sleep(1)


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
    horses_to_insert = []
    for i, horse_id in enumerate(
        tqdm(missing_horse_ids, desc="Fetching horse profiles")
    ):
        html_content = fetch_and_save_horse_html(horse_id)
        if html_content:
            horse_data = parse_horse_profile_html(horse_id, html_content)
            horses_to_insert.append(horse_data)

        if len(horses_to_insert) >= 20:
            insert_horses(horses_to_insert)
            horses_to_insert = []

        if i % 10 == 0:
            time.sleep(1)

    if horses_to_insert:
        insert_horses(horses_to_insert)

    # Collect missing jockey profiles
    cursor.execute(
        "SELECT DISTINCT jockey_id FROM results WHERE jockey_id NOT IN "
        "(SELECT jockey_id FROM jockeys WHERE jockey_name != 'UNKNOWN')"
    )
    missing_jockey_ids = [row[0] for row in cursor.fetchall()]
    jockeys_to_insert = []
    for i, jockey_id in enumerate(
        tqdm(missing_jockey_ids, desc="Fetching jockey profiles")
    ):
        html_content = fetch_and_save_jockey_html(jockey_id)
        if html_content:
            jockey_data = parse_jockey_profile_html(jockey_id, html_content)
            jockeys_to_insert.append(jockey_data)

        if len(jockeys_to_insert) >= 20:
            insert_jockeys(jockeys_to_insert)
            jockeys_to_insert = []

        if i % 10 == 0:
            time.sleep(1)

    if jockeys_to_insert:
        insert_jockeys(jockeys_to_insert)

    conn.close()
