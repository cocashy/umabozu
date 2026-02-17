import logging
import os
import sys

# Add the parent directory of umabozu to the system path
_script_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "src"))
sys.path.insert(0, _project_root)

from umabozu.data_collection.html_parser import parse_horse_profile_html  # noqa: E402

logger = logging.getLogger(__name__)


def check_horse():
    path = "data/raw/html/horses/2020106601.html"
    if not os.path.exists(path):
        logger.error("File not found")
        return

    with open(path, "rb") as f:
        content = f.read()

    horse_data = parse_horse_profile_html("2020106601", content)

    logger.info("--- Horse Profile ---")
    logger.info(f"Name: {horse_data.get('horse_name', 'N/A')}")
    logger.info(f"Birth Date: {horse_data.get('birth_date', 'N/A')}")
    logger.info(f"Trainer ID: {horse_data.get('trainer_id', 'N/A')}")
    logger.info(f"Owner ID: {horse_data.get('owner_id', 'N/A')}")
    logger.info(f"Producer ID: {horse_data.get('producer_id', 'N/A')}")
    logger.info(f"Sire ID: {horse_data.get('sire_id', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_horse()
