import sqlite3
from typing import Any

import polars as pl

DB_PATH = "data/umabozu.db"


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    """Creates necessary tables in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            race_name TEXT,
            race_date TEXT,
            start_time TEXT,
            course_name TEXT,
            distance INTEGER,
            course_type TEXT,
            course_direction TEXT,
            track_condition TEXT,
            weather TEXT,
            race_number INTEGER,
            race_grade TEXT,
            num_of_runners INTEGER,
            prize_money_1st REAL,
            prize_money_2nd REAL,
            prize_money_3rd REAL,
            prize_money_4th REAL,
            prize_money_5th REAL
        );"""
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS horses (
            horse_id TEXT PRIMARY KEY,
            horse_name TEXT DEFAULT 'UNKNOWN',
            sex TEXT,
            birth_date TEXT,
            sire_id TEXT,
            dam_id TEXT,
            dam_sire_id TEXT,
            owner_id TEXT,
            trainer_id TEXT,
            producer_id TEXT,
            breeding_farm_id TEXT
        );"""
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS jockeys (
            jockey_id TEXT PRIMARY KEY,
            jockey_name TEXT DEFAULT 'UNKNOWN',
            birth_date TEXT,
            birth_place TEXT,
            belonging TEXT,
            debut_date TEXT,
            license_type TEXT
        );"""
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            race_id TEXT,
            horse_id TEXT,
            jockey_id TEXT,
            trainer_id TEXT,
            owner_id TEXT,
            rank INTEGER,
            bracket_number INTEGER,
            horse_number INTEGER,
            sex_age TEXT,
            weight_carried REAL,
            distance_diff REAL,
            last_3f_time REAL,
            odds_win REAL,
            popularity INTEGER,
            horse_weight INTEGER,
            horse_weight_diff INTEGER,
            PRIMARY KEY (race_id, horse_id),
            FOREIGN KEY (race_id) REFERENCES races(race_id),
            FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
            FOREIGN KEY (jockey_id) REFERENCES jockeys(jockey_id)
        );"""
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS staged_race_ids (
            race_id TEXT PRIMARY KEY
        );"""
    )

    conn.commit()
    conn.close()


def _insert_data(table_name: str, data: dict[str, Any], upsert: bool = False):
    """Generic insert or upsert function."""
    if not data:
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    values = tuple(data.values())

    if upsert:
        update_clause = ", ".join([f"{col} = ?" for col in data])
        query = f"""
            INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
            ON CONFLICT(rowid) DO UPDATE SET {update_clause}
        """
        cursor.execute(query, values + values)  # Pass values twice for UPSERT
    else:
        query = (
            f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
        )
        cursor.execute(query, values)

    conn.commit()
    conn.close()


def insert_race(data: dict[str, Any]):
    _insert_data("races", data)


def insert_horse(data: dict[str, Any]):
    # Horses table uses PRIMARY KEY (horse_id), so ON CONFLICT (horse_id)
    # is more appropriate than ON CONFLICT (rowid).
    # Since horse data can be updated (e.g., owner changes), UPSERT is suitable.
    conn = get_db_connection()
    cursor = conn.cursor()
    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    values = tuple(data.values())
    update_clause = ", ".join([f"{col} = ?" for col in data])
    query = f"""
        INSERT INTO horses ({columns}) VALUES ({placeholders})
        ON CONFLICT(horse_id) DO UPDATE SET {update_clause}
    """
    cursor.execute(query, values + values)
    conn.commit()
    conn.close()


def insert_jockey(data: dict[str, Any]):
    # Jockeys table uses PRIMARY KEY (jockey_id), so ON CONFLICT (jockey_id)
    # is more appropriate than ON CONFLICT (rowid).
    # Jockey data can also be updated, so UPSERT is suitable.
    conn = get_db_connection()
    cursor = conn.cursor()
    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    values = tuple(data.values())
    update_clause = ", ".join([f"{col} = ?" for col in data])
    query = f"""
        INSERT INTO jockeys ({columns}) VALUES ({placeholders})
        ON CONFLICT(jockey_id) DO UPDATE SET {update_clause}
    """
    cursor.execute(query, values + values)
    conn.commit()
    conn.close()


def insert_trainer(data: dict[str, Any]):
    _insert_data("trainers", data, upsert=True)


def insert_owner(data: dict[str, Any]):
    _insert_data("owners", data, upsert=True)


def insert_result(data: dict[str, Any]):
    _insert_data("results", data)


def insert_staged_race_id(race_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO staged_race_ids (race_id) VALUES (?)",
        (race_id,),
    )
    conn.commit()
    conn.close()


def get_staged_race_ids() -> list[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT race_id FROM staged_race_ids")
    race_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return race_ids


def delete_staged_race_id(race_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM staged_race_ids WHERE race_id = ?", (race_id,))
    conn.commit()
    conn.close()


def get_missing_horse_ids() -> list[str]:
    """Finds horse_ids in results with incomplete horse profiles."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = (
        "SELECT DISTINCT T1.horse_id FROM results AS T1 "
        "LEFT JOIN horses AS T2 ON T1.horse_id = T2.horse_id "
        "WHERE T2.horse_id IS NULL OR T2.horse_name = 'UNKNOWN'"
    )
    cursor.execute(query)
    missing_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return missing_ids


def get_missing_jockey_ids() -> list[str]:
    """Finds jockey_ids in results with incomplete jockey profiles."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = (
        "SELECT DISTINCT T1.jockey_id FROM results AS T1 "
        "LEFT JOIN jockeys AS T2 ON T1.jockey_id = T2.jockey_id "
        "WHERE T2.jockey_id IS NULL OR T2.jockey_name = 'UNKNOWN'"
    )
    cursor.execute(query)
    missing_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return missing_ids


def get_training_data():
    """
    Retrieves all necessary data for training the model as a Polars DataFrame.
    This includes joining races, results, horses, and jockeys.
    """
    conn = get_db_connection()
    query = """
        SELECT
            r.*,
            res.horse_id, res.jockey_id,
            res.rank, res.bracket_number, res.horse_number, res.sex_age,
            res.weight_carried, res.distance_diff, res.last_3f_time,
            res.odds_win, res.popularity, res.horse_weight, res.horse_weight_diff,
            h.horse_name, h.sex, h.birth_date AS horse_birth_date,
            j.jockey_name, j.birth_date AS jockey_birth_date, j.debut_date
        FROM
            races AS r
        JOIN
            results AS res ON r.race_id = res.race_id
        LEFT JOIN
            horses AS h ON res.horse_id = h.horse_id
        LEFT JOIN
            jockeys AS j ON res.jockey_id = j.jockey_id
        WHERE
            h.horse_name IS NOT 'UNKNOWN' AND j.jockey_name IS NOT 'UNKNOWN'
    """
    df = pl.read_database(query, connection=conn)
    conn.close()
    return df
