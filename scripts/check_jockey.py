import logging
import os

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def check_jockey():
    path = "data/raw/html/jockeys/05445.html"
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
        logger.error(f"Error decoding jockey HTML: {e}")
        return

    soup = BeautifulSoup(text, "html.parser")

    logger.info("--- Jockey Profile ---")
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_jockey()
