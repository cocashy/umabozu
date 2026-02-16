# src/umabozu/database/schema.py

# Define SQL statements for table creation

CREATE_RACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    race_name TEXT NOT NULL,
    race_date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    course_name TEXT NOT NULL,
    distance INTEGER NOT NULL,
    course_type TEXT NOT NULL,
    course_direction TEXT NOT NULL,
    track_condition TEXT NOT NULL,
    weather TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_grade TEXT,
    num_of_runners INTEGER NOT NULL,
    prize_money_1st INTEGER,
    prize_money_2nd INTEGER,
    prize_money_3rd INTEGER,
    prize_money_4th INTEGER,
    prize_money_5th INTEGER
);
"""

CREATE_OWNERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS owners (
    owner_id TEXT PRIMARY KEY,
    owner_name TEXT NOT NULL
);
"""

CREATE_TRAINERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id TEXT PRIMARY KEY,
    trainer_name TEXT NOT NULL
);
"""

CREATE_PRODUCERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS producers (
    producer_id TEXT PRIMARY KEY,
    producer_name TEXT NOT NULL
);
"""

CREATE_BREEDING_FARMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS breeding_farms (
    breeding_farm_id TEXT PRIMARY KEY,
    breeding_farm_name TEXT NOT NULL
);
"""

CREATE_HORSES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS horses (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT NOT NULL,
    sex TEXT NOT NULL,
    birth_date TEXT NOT NULL,
    sire_id TEXT,
    dam_id TEXT,
    dam_sire_id TEXT,
    owner_id TEXT,
    trainer_id TEXT,
    producer_id TEXT,
    breeding_farm_id TEXT,
    FOREIGN KEY (sire_id) REFERENCES horses(horse_id),
    FOREIGN KEY (dam_id) REFERENCES horses(horse_id),
    FOREIGN KEY (dam_sire_id) REFERENCES horses(horse_id),
    FOREIGN KEY (owner_id) REFERENCES owners(owner_id),
    FOREIGN KEY (trainer_id) REFERENCES trainers(trainer_id),
    FOREIGN KEY (producer_id) REFERENCES producers(producer_id),
    FOREIGN KEY (breeding_farm_id) REFERENCES breeding_farms(breeding_farm_id)
);
"""

CREATE_JOCKEYS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id TEXT PRIMARY KEY,
    jockey_name TEXT NOT NULL,
    birth_date TEXT,
    birth_place TEXT,
    belonging TEXT,
    debut_date TEXT,
    license_type TEXT
);
"""

CREATE_RESULTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS results (
    race_id TEXT,
    horse_id TEXT,
    jockey_id TEXT,
    trainer_id TEXT,
    owner_id TEXT,
    rank INTEGER NOT NULL,
    bracket_number INTEGER NOT NULL,
    horse_number INTEGER NOT NULL,
    sex_age TEXT NOT NULL,
    weight_carried REAL NOT NULL,
    distance_diff REAL,
    last_3f_time REAL,
    odds_win REAL,
    popularity INTEGER,
    horse_weight INTEGER,
    horse_weight_diff INTEGER,
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
    FOREIGN KEY (jockey_id) REFERENCES jockeys(jockey_id),
    FOREIGN KEY (trainer_id) REFERENCES trainers(trainer_id),
    FOREIGN KEY (owner_id) REFERENCES owners(owner_id)
);
"""

CREATE_STAGED_RACE_IDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS staged_race_ids (
    race_id TEXT PRIMARY KEY,
    race_date TEXT NOT NULL
);
"""

ALL_TABLE_SQLS = [
    CREATE_RACES_TABLE_SQL,
    CREATE_OWNERS_TABLE_SQL,
    CREATE_TRAINERS_TABLE_SQL,
    CREATE_PRODUCERS_TABLE_SQL,
    CREATE_BREEDING_FARMS_TABLE_SQL,
    CREATE_HORSES_TABLE_SQL,
    CREATE_JOCKEYS_TABLE_SQL,
    CREATE_RESULTS_TABLE_SQL,
    CREATE_STAGED_RACE_IDS_TABLE_SQL,
]
