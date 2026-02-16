import logging
import os
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def check_horse():
    path = "data/raw/html/horses/2020106601.html"
    if not os.path.exists(path):
        logger.error("File not found")
        return

    with open(path, "rb") as f:
        content = f.read()

    text = ""
    try:
        text = content.decode("euc-jp")
    except UnicodeDecodeError:
        text = content.decode("cp932", errors="replace")
    except Exception as e:
        logger.error(f"Error decoding horse HTML: {e}")
        return

    soup = BeautifulSoup(text, "html.parser")

    logger.info("--- Horse Profile ---")
    h1 = soup.find("h1")
    logger.info(f"Name: {h1.get_text(strip=True) if h1 else 'N/A'}")

    # Profile table
    table = soup.find("table", class_="db_prof_table")
    if table:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td:
                logger.info(f"{th.get_text(strip=True)}: {td.get_text(strip=True)}")

    # Bloodline table
    blood_table = soup.find("table", class_="blood_table")
    if blood_table:
        logger.info("\n--- Bloodline ---")
        # Sire, Dam, DamSire are usually in specific cells
        tds = blood_table.find_all("td")
        if len(tds) > 0:
            sire_a = tds[0].find("a")
            logger.info(
                f"Sire: {sire_a.get_text(strip=True) if sire_a else 'N/A'} "
                f"({sire_a['href'] if sire_a else 'N/A'})"
            )
        # Add more sophisticated parsing for dam and dam_sire if needed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_horse()
