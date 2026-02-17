import logging
import os
import re
from contextlib import suppress
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag

# Set up logging
logger = logging.getLogger(__name__)


def parse_shutsubahyou_html(
    race_id: str, html_content: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parses shutsubahyou (race card) HTML from race.netkeiba.com.
    Note: Future races have different HTML structure than results page.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Parse Race Info
    race_data: dict[str, Any] = {"race_id": race_id}

    # Extract race name and number
    race_name_tag = soup.find("div", class_="RaceName") or soup.find("h1")
    if race_name_tag:
        race_data["race_name"] = race_name_tag.get_text(strip=True)

    # Line break for E501
    race_num_tag = soup.find("dt", class_="RaceNum") or soup.find(
        "div", class_="RaceNum"
    )
    if race_num_tag:
        num_match = re.search(r"\d+", race_num_tag.get_text(strip=True))
        if num_match:
            race_data["race_number"] = int(num_match.group())  # Extract only digits

    race_data_01 = soup.find("div", class_="RaceData01")
    if race_data_01:
        meta_text = race_data_01.get_text(strip=True)
        dist_match = re.search(r"(芝|ダ|障)(\d+)m", meta_text)
        if dist_match:
            race_data["course_type"] = dist_match.group(1)
            race_data["distance"] = int(dist_match.group(2))
        dir_match = re.search(r"\((右|左|直線)\)", meta_text)
        if dir_match:
            race_data["course_direction"] = dir_match.group(1)
        else:
            race_data["course_direction"] = "None"
        # Extract track condition and weather from meta_text or default
        tc_match = re.search(r"馬場:\s*(\w+)", meta_text)  # RUF001: Corrected colon
        if tc_match:
            race_data["track_condition"] = tc_match.group(1)
        # RUF001: Corrected colon - changed from 文字化け to 天候
        weather_match = re.search(r"天候:\s*(\w+)", meta_text)
        if weather_match:
            race_data["weather"] = weather_match.group(1)

    # 2. Parse Runners (ShutsubaTable)
    runners = []
    # Try different table selectors
    table_candidate = soup.find("table", class_=re.compile(r"Shutsuba_?Table", re.I))
    rows = []
    if isinstance(table_candidate, Tag):
        rows = table_candidate.find_all("tr", class_=re.compile(r"HorseList", re.I))
    elif isinstance(soup, BeautifulSoup):
        rows = soup.find_all("tr", class_=re.compile(r"HorseList", re.I))


    if rows:
        for row in rows:
            runner = {"race_id": race_id}

            # Horse Number (Umaban)
            umaban_td = row.find("td", class_=re.compile(r"Umaban", re.I))
            if umaban_td:
                txt = umaban_td.get_text(strip=True)
                if txt.isdigit():
                    runner["horse_number"] = int(txt)

            # Horse Info
            # Line break for E501
            horse_td = row.find(
                "td", class_=re.compile(r"HorseInfo", re.I)
            ) or row.find("span", class_=re.compile(r"Horse_?Name", re.I))

            if horse_td:
                a_tag = horse_td.find("a", href=True)
                if a_tag:
                    runner["horse_name"] = a_tag.get_text(strip=True)
                    href = a_tag.get("href", "")
                    if isinstance(href, str):
                        h_id_match = re.search(r"/horse/(\d+)", href)
                        if h_id_match:
                            runner["horse_id"] = h_id_match.group(1)

            # Jockey Info
            jockey_td = row.find("td", class_=re.compile(r"Jockey", re.I))
            if jockey_td:
                a_tag = jockey_td.find("a", href=True)
                if a_tag:
                    runner["jockey_name"] = a_tag.get_text(strip=True)
                    href = a_tag.get("href", "")
                    if isinstance(href, str):
                        # Line break for E501
                        j_id_match = re.search(
                            r"/jockey/(?:result/recent/)?(\d+)", href
                        )
                        if j_id_match:
                            runner["jockey_id"] = j_id_match.group(1)

            # Weight Carried
            weight_td = row.find("td", string=re.compile(r"^\d+\.\d+$"))
            if not weight_td:
                for td in row.find_all("td"):
                    if re.match(r"^\d+\.\d+$", td.get_text(strip=True)):
                        weight_td = td
                        break

            if weight_td:
                with suppress(Exception):
                    weight_match = re.search(
                        r"\d+\.\d+", weight_td.get_text(strip=True)
                    )
                    if weight_match:
                        runner["weight_carried"] = float(weight_match.group())
            else:
                runner["weight_carried"] = 55.0  # Default value

            # Odds & Popularity (ensure these keys exist)
            runner["odds_win"] = 1.0
            runner["popularity"] = 0

            odds_td = row.find("td", class_=re.compile(r"Odds", re.I))
            if odds_td:
                odds_span = odds_td.find("span")
                if odds_span:
                    with suppress(Exception):
                        runner["odds_win"] = float(odds_span.get_text(strip=True))

            pop_td = row.find("td", class_=re.compile(r"Popular", re.I)) or row.find(
                "td", class_=re.compile(r"Fav", re.I)
            )
            if pop_td:
                txt = pop_td.get_text(strip=True)
                if txt.isdigit():
                    runner["popularity"] = int(txt)
            else:
                # Fallback: look at all TDs for a cell that looks like popularity
                tds = row.find_all("td")
                if len(tds) > 9:  # Popularity is often near the end
                    txt = tds[9].get_text(strip=True)
                    if txt.isdigit():
                        runner["popularity"] = int(txt)

            if "horse_id" in runner:
                runners.append(runner)

    return race_data, runners


def parse_race_result_html(
    race_id: str, html_content: bytes
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parses race result page HTML."""
    content_str = html_content.decode("euc-jp")
    soup = BeautifulSoup(content_str, "html.parser")

    race_data: dict[str, Any] = {"race_id": race_id}
    h1 = soup.find("h1")
    if h1:
        race_data["race_name"] = h1.get_text(strip=True)

    full_text = soup.get_text(" ", strip=True)
    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", full_text)
    if date_match:
        race_data["race_date"] = (
            f"{date_match.group(1)}-{int(date_match.group(2)):02d}"
            f"-{int(date_match.group(3)):02d}"
        )
    course_match = re.search(r"\d+回(\w+)\d+日目", full_text)
    if course_match:
        race_data["course_name"] = course_match.group(1)
    num_match = re.search(r"(\d+)R", full_text)
    if num_match:
        race_data["race_number"] = int(num_match.group(1))

    dist_match = re.search(r"([芝ダ障])([右左直][内外]?)(\d+)m", full_text)
    if dist_match:
        race_data["course_type"] = dist_match.group(1)
        race_data["course_direction"] = dist_match.group(2) or "直"
        race_data["distance"] = int(dist_match.group(3))

    weather_match = re.search(r"天候:\s*(\w+)", full_text)
    if weather_match:
        race_data["weather"] = weather_match.group(1)
    cond_match = re.search(r"(?:馬場|ダート):\s*(\w+)", full_text)
    if cond_match:
        race_data["track_condition"] = cond_match.group(1)
    start_match = re.search(r"発走:\s*(\d{2}:\d{2})", full_text)
    if start_match:
        race_data["start_time"] = start_match.group(1)

    race_data.setdefault("race_name", "UNKNOWN")
    race_data.setdefault("race_number", 0)
    race_data.setdefault("race_date", "1900-01-01")
    race_data.setdefault("start_time", "00:00")
    race_data.setdefault("course_name", "UNKNOWN")
    race_data.setdefault("distance", 0)
    race_data.setdefault("course_type", "UNKNOWN")
    race_data.setdefault("course_direction", "UNKNOWN")
    race_data.setdefault("track_condition", "UNKNOWN")
    race_data.setdefault("weather", "UNKNOWN")
    race_data.setdefault("race_grade", None)
    race_data.setdefault("num_of_runners", 0)

    results_data = []
    table = soup.find("table", class_="race_table_01")
    if not table:
        return race_data, []

    rows = []
    if isinstance(table, Tag):
        rows = table.find_all("tr")
    else:
        # Fallback in case table is not a Tag (e.g. NavigableString).
        # Although unlikely here.
        return race_data, []


    if len(rows) < 2:
        return race_data, []
    header_cols = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

    # B023 fix: Make cols an argument to get_txt or inline it
    idx = {
        "rank": _find_idx_in_headers(["着順", "着"], header_cols),
        "bracket": _find_idx_in_headers(["枠番", "枠"], header_cols),
        "horse_num": _find_idx_in_headers(["馬番", "馬"], header_cols),
        "horse_name": _find_idx_in_headers(["馬名"], header_cols),
        "sex_age": _find_idx_in_headers(["性齢"], header_cols),
        "weight": _find_idx_in_headers(["斤量"], header_cols),
        "jockey": _find_idx_in_headers(["騎手"], header_cols),
        "diff": _find_idx_in_headers(["着差"], header_cols),
        "last_3f": _find_idx_in_headers(["上り", "上り3F"], header_cols),
        "odds": _find_idx_in_headers(["単勝", "オッズ"], header_cols),
        "pop": _find_idx_in_headers(["人気"], header_cols),
        "horse_weight": _find_idx_in_headers(["馬体重"], header_cols),
        "trainer": _find_idx_in_headers(["調教師"], header_cols),
        "owner": _find_idx_in_headers(["馬主"], header_cols),
    }

    def _get_txt_from_cols(key, cols_list):  # New helper to pass cols
        i = idx.get(key, -1)
        # Line break for E501
        return (
            cols_list[i].get_text(strip=True) if i != -1 and i < len(cols_list) else ""
        )

    results_data = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 10:
            continue

        res = {"race_id": race_id}
        rank_txt = _get_txt_from_cols("rank", cols)
        if not rank_txt.isdigit():
            continue
        res["rank"] = int(rank_txt)
        res["bracket_number"] = (
            int(_get_txt_from_cols("bracket", cols))
            if _get_txt_from_cols("bracket", cols).isdigit()
            else 0
        )
        res["horse_number"] = (
            int(_get_txt_from_cols("horse_num", cols))
            if _get_txt_from_cols("horse_num", cols).isdigit()
            else 0
        )

        horse_a = cols[idx["horse_name"]].find("a") if idx["horse_name"] != -1 else None
        horse_href = horse_a.get("href") if horse_a else None
        if horse_a and isinstance(horse_href, str):
            res["horse_id"] = horse_href.split("/")[-2]
        else:
            res["horse_id"] = None
        res["horse_name"] = horse_a.get_text(strip=True) if horse_a else "UNKNOWN"

        jockey_a = cols[idx["jockey"]].find("a") if idx["jockey"] != -1 else None
        jockey_href = jockey_a.get("href") if jockey_a else None
        if jockey_a and isinstance(jockey_href, str):
            res["jockey_id"] = jockey_href.split("/")[-2]
        else:
            res["jockey_id"] = None
        res["jockey_name"] = jockey_a.get_text(strip=True) if jockey_a else "UNKNOWN"

        trainer_a = cols[idx["trainer"]].find("a") if idx["trainer"] != -1 else None
        trainer_href = trainer_a.get("href") if trainer_a else None
        if trainer_a and isinstance(trainer_href, str):
            res["trainer_id"] = trainer_href.split("/")[-2]
        else:
            res["trainer_id"] = None
        res["trainer_name"] = trainer_a.get_text(strip=True) if trainer_a else "UNKNOWN"

        owner_a = cols[idx["owner"]].find("a") if idx["owner"] != -1 else None
        owner_href = owner_a.get("href") if owner_a else None
        if owner_a and isinstance(owner_href, str):
            res["owner_id"] = owner_href.split("/")[-2]
        else:
            res["owner_id"] = None
        res["owner_name"] = owner_a.get_text(strip=True) if owner_a else "UNKNOWN"

        res["sex_age"] = _get_txt_from_cols("sex_age", cols)
        res["weight_carried"] = (
            float(_get_txt_from_cols("weight", cols))
            if _get_txt_from_cols("weight", cols).replace(".", "").isdigit()
            else 0.0
        )

        diff_txt = _get_txt_from_cols("diff", cols)
        if not diff_txt or diff_txt in ["--", "0", ""]:
            res["distance_diff"] = 0.0
        elif "ハナ" in diff_txt:
            res["distance_diff"] = 0.1
        elif "クビ" in diff_txt:
            res["distance_diff"] = 0.2
        else:
            res["distance_diff"] = 0.0
            with suppress(Exception):
                res["distance_diff"] = float(diff_txt)

        res["last_3f_time"] = (
            float(_get_txt_from_cols("last_3f", cols))
            if _get_txt_from_cols("last_3f", cols).replace(".", "").isdigit()
            else None
        )
        res["odds_win"] = (
            float(_get_txt_from_cols("odds", cols))
            if _get_txt_from_cols("odds", cols).replace(".", "").isdigit()
            else 0.0
        )
        res["popularity"] = (
            int(_get_txt_from_cols("pop", cols))
            if _get_txt_from_cols("pop", cols).isdigit()
            else 0
        )
        w_txt = _get_txt_from_cols("horse_weight", cols)
        w_match = re.search(r"(\d+)\(([-+]?\d+)\)", w_txt)
        if w_match:
            res["horse_weight"] = int(w_match.group(1))
            res["horse_weight_diff"] = int(w_match.group(2))
        else:
            res["horse_weight"] = int(w_txt) if w_txt.isdigit() else 0
            res["horse_weight_diff"] = 0
        results_data.append(res)

    return race_data, results_data


def parse_horse_profile_html(horse_id: str, html_content: bytes) -> dict[str, Any]:
    """Parses horse profile page HTML."""
    content_str = html_content.decode("euc-jp")
    soup = BeautifulSoup(content_str, "html.parser")

    horse_data: dict[str, Any] = {"horse_id": horse_id}
    h1 = soup.select_one("div.horse_title h1")
    if h1:
        horse_data["horse_name"] = h1.get_text(strip=True)

    prof_table = soup.find("table", class_="db_prof_table")
    if prof_table and isinstance(prof_table, Tag):
        for tr in prof_table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(strip=True)
            if "生年月日" in key:
                m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", val)
                if m:
                    horse_data["birth_date"] = (
                        f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                    )
            elif "調教師" in key:
                a = td.find("a")
                href = a.get("href") if a else None
                if a and isinstance(href, str):
                    horse_data["trainer_id"] = href.split("/")[-2]
            elif "馬主" in key:
                a = td.find("a")
                href = a.get("href") if a else None
                if a and isinstance(href, str):
                    horse_data["owner_id"] = href.split("/")[-2]
            elif "生産者" in key:
                a = td.find("a")
                href = a.get("href") if a else None
                if a and isinstance(href, str):
                    horse_data["producer_id"] = href.split("/")[-2]
            elif "産地" in key:
                pass

    # Bloodline
    blood_table = soup.find("table", class_="blood_table")
    if blood_table and isinstance(blood_table, Tag):
        tds = blood_table.find_all("td")
        if len(tds) > 0:
            a = tds[0].find("a")
            href = a.get("href") if a else None
            if a and isinstance(href, str):
                horse_data["sire_id"] = href.split("/")[-2]

    horse_data.setdefault("sex", "U")
    horse_data.setdefault("birth_date", "2000-01-01")
    return horse_data


def parse_jockey_profile_html(jockey_id: str, html_content: bytes) -> dict[str, Any]:
    """Parses jockey profile page HTML."""
    content_str = html_content.decode("euc-jp")
    soup = BeautifulSoup(content_str, "html.parser")

    jockey_data: dict[str, Any] = {"jockey_id": jockey_id}
    # Name is often in a specific div for jockeys
    name_elem = soup.select_one("div.db_head_name h1") or soup.find("h1")
    if name_elem:
        jockey_data["jockey_name"] = (
            name_elem.get_text(strip=True).split("(")[0].strip()
        )

    full_text = soup.get_text(" ", strip=True)
    birth_match = re.search(r"(\d{4}/\d{2}/\d{2})", full_text)
    if birth_match:
        jockey_data["birth_date"] = birth_match.group(1).replace("/", "-")

    debut_match = re.search(r"デビュー年 (\d{4})年", full_text)
    if debut_match:
        jockey_data["debut_date"] = f"{debut_match.group(1)}-01-01"

    if "中央" in full_text:
        jockey_data["belonging"] = "中央"
    elif "地方" in full_text:
        jockey_data["belonging"] = "地方"
    else:
        jockey_data["belonging"] = "不明"

    return jockey_data


def save_parsed_data_to_db(race_id: str, html_content: bytes):
    from umabozu.database.crud import insert_race, insert_results

    race_data, results_data = parse_race_result_html(race_id, html_content)
    if not results_data:
        return
    insert_race(race_data)

    results_to_insert = []
    for r in results_data:
        insert_data = {
            k: v
            for k, v in r.items()
            if k not in ["horse_name", "jockey_name", "trainer_name", "owner_name"]
        }
        results_to_insert.append(insert_data)

    insert_results(results_to_insert)
    logger.info(f"Race {race_id} ({race_data['race_name']}) saved.")


# Helper for _find_idx_in_headers (to fix B023)
def _find_idx_in_headers(names, header_cols):
    for name in names:
        if name in header_cols:
            return header_cols.index(name)
    return -1


if __name__ == "__main__":
    # Move sys.path.insert here for script execution
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )
    from umabozu.database.crud import create_tables, insert_horse, insert_jockey

    logging.basicConfig(level=logging.INFO)
    create_tables()

    # Test Race
    sample_race_id = "202336010301"
    race_path = os.path.join("data", "raw", "html", "races", f"{sample_race_id}.html")
    if os.path.exists(race_path):
        with open(race_path, "rb") as f:
            content = f.read()
        save_parsed_data_to_db(sample_race_id, content)

    # Test Horse
    sample_horse_id = "2020106601"
    horse_path = os.path.join(
        "data", "raw", "html", "horses", f"{sample_horse_id}.html"
    )
    if os.path.exists(horse_path):
        with open(horse_path, "rb") as f:
            content = f.read()
        horse_data = parse_horse_profile_html(sample_horse_id, content)
        insert_horse(horse_data)

    # Test Jockey
    sample_jockey_id = "05445"
    jockey_path = os.path.join(
        "data", "raw", "html", "jockeys", f"{sample_jockey_id}.html"
    )
    if os.path.exists(jockey_path):
        with open(jockey_path, "rb") as f:
            content = f.read()
        jockey_data = parse_jockey_profile_html(sample_jockey_id, content)
        insert_jockey(jockey_data)
