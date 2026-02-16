import datetime
import logging
import os
import random
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def generate_mock_data(
    num_races: int = 10, num_horses: int = 20, num_jockeys: int = 5
) -> dict[str, pl.DataFrame]:
    """
    Generate mock data for all tables based on the defined schema.
    """
    mock_data: dict[str, pl.DataFrame] = {}

    # --- Owners ---
    owner_ids = [f"O{i:05d}" for i in range(num_races)]
    owners_data = {
        "owner_id": owner_ids,
        "owner_name": [f"Owner_{i}" for i in range(num_races)],
    }
    mock_data["owners"] = pl.DataFrame(owners_data)

    # --- Trainers ---
    trainer_ids = [f"T{i:05d}" for i in range(num_races)]
    trainers_data = {
        "trainer_id": trainer_ids,
        "trainer_name": [f"Trainer_{i}" for i in range(num_races)],
    }
    mock_data["trainers"] = pl.DataFrame(trainers_data)

    # --- Producers ---
    producer_ids = [f"P{i:05d}" for i in range(num_races)]
    producers_data = {
        "producer_id": producer_ids,
        "producer_name": [f"Producer_{i}" for i in range(num_races)],
    }
    mock_data["producers"] = pl.DataFrame(producers_data)

    # --- Breeding Farms ---
    breeding_farm_ids = [f"B{i:05d}" for i in range(num_races)]
    breeding_farms_data = {
        "breeding_farm_id": breeding_farm_ids,
        "breeding_farm_name": [f"Farm_{i}" for i in range(num_races)],
    }
    mock_data["breeding_farms"] = pl.DataFrame(breeding_farms_data)

    # --- Jockeys ---
    jockey_ids = [f"J{i:05d}" for i in range(num_jockeys)]
    jockeys_data = {
        "jockey_id": jockey_ids,
        "jockey_name": [f"Jockey_{i}" for i in range(num_jockeys)],
        "birth_date": [
            (
                datetime.date(1980, 1, 1)
                + datetime.timedelta(days=random.randint(0, 365 * 10))
            ).strftime("%Y-%m-%d")
            for _ in range(num_jockeys)
        ],
        "birth_place": random.choices(
            ["北海道", "関東", "関西", "海外"], k=num_jockeys
        ),
        "belonging": random.choices(["中央", "地方", "不明"], k=num_jockeys),
        "debut_date": [
            (
                datetime.date(2000, 1, 1)
                + datetime.timedelta(days=random.randint(0, 365 * 10))
            ).strftime("%Y-%m-%d")
            for _ in range(num_jockeys)
        ],
        "license_type": random.choices(["平地", "障害", "見習"], k=num_jockeys),
    }
    mock_data["jockeys"] = pl.DataFrame(jockeys_data)

    # --- Horses ---
    horse_ids = [f"H{i:05d}" for i in range(num_horses)]
    sex_choices = ["牡", "牝", "セン"]
    birth_dates = [
        (
            datetime.date(2018, 1, 1)
            + datetime.timedelta(days=random.randint(0, 365 * 4))
        ).strftime("%Y-%m-%d")
        for _ in range(num_horses)
    ]
    # Sire, Dam, DamSire IDs - Self-referencing and existing horses
    sire_ids = random.choices(horse_ids, k=num_horses)
    dam_ids = random.choices(horse_ids, k=num_horses)
    dam_sire_ids = random.choices(horse_ids, k=num_horses)

    horses_data = {
        "horse_id": horse_ids,
        "horse_name": [f"Horse_{i}" for i in range(num_horses)],
        "sex": random.choices(sex_choices, k=num_horses),
        "birth_date": birth_dates,
        "sire_id": sire_ids,
        "dam_id": dam_ids,
        "dam_sire_id": dam_sire_ids,
        "owner_id": random.choices(
            mock_data["owners"]["owner_id"].to_list(), k=num_horses
        ),
        "trainer_id": random.choices(
            mock_data["trainers"]["trainer_id"].to_list(), k=num_horses
        ),
        "producer_id": random.choices(
            mock_data["producers"]["producer_id"].to_list(), k=num_horses
        ),
        "breeding_farm_id": random.choices(
            mock_data["breeding_farms"]["breeding_farm_id"].to_list(), k=num_horses
        ),
    }
    mock_data["horses"] = pl.DataFrame(horses_data)

    # --- Races ---
    race_ids = [f"R{i:05d}" for i in range(num_races)]
    race_names = [f"Race_{i}" for i in range(num_races)]
    race_dates = [
        (
            datetime.date(2023, 1, 1)
            + datetime.timedelta(days=i * random.randint(1, 10))
        ).strftime("%Y-%m-%d")
        for i in range(num_races)
    ]
    course_names = ["東京", "中山", "京都", "阪神"]
    course_types = ["芝", "ダート"]
    course_directions = ["右", "左", "直線"]
    track_conditions = ["良", "稍重", "重", "不良"]
    weathers = ["晴", "曇", "雨", "雪"]
    race_grades = [None, "G1", "G2", "G3"]

    races_data = {
        "race_id": race_ids,
        "race_name": race_names,
        "race_date": race_dates,
        "start_time": [
            f"{random.randint(9, 17):02d}:{random.randint(0, 59):02d}"
            for _ in range(num_races)
        ],
        "course_name": random.choices(course_names, k=num_races),
        "distance": random.choices(
            [1200, 1400, 1600, 1800, 2000, 2400, 3000], k=num_races
        ),
        "course_type": random.choices(course_types, k=num_races),
        "course_direction": random.choices(course_directions, k=num_races),
        "track_condition": random.choices(track_conditions, k=num_races),
        "weather": random.choices(weathers, k=num_races),
        "race_number": [i + 1 for i in range(num_races)],
        "race_grade": random.choices(race_grades, k=num_races),
        "num_of_runners": [random.randint(8, 18) for _ in range(num_races)],
        "prize_money_1st": [random.randint(1000, 10000) for _ in range(num_races)],
        "prize_money_2nd": [random.randint(500, 5000) for _ in range(num_races)],
        "prize_money_3rd": [random.randint(200, 2000) for _ in range(num_races)],
        "prize_money_4th": [random.randint(100, 1000) for _ in range(num_races)],
        "prize_money_5th": [random.randint(50, 500) for _ in range(num_races)],
    }
    mock_data["races"] = pl.DataFrame(races_data)

    # --- Results ---
    results_records: list[dict[str, Any]] = []
    for race_idx, race_id in enumerate(race_ids):
        num_runners = mock_data["races"]["num_of_runners"].item(race_idx)

        # Select unique horses for this race
        available_horses = random.sample(horse_ids, num_runners)
        # Select unique jockeys for this race
        available_jockeys = random.sample(
            mock_data["jockeys"]["jockey_id"].to_list(), num_runners
        )

        # Shuffle ranks to ensure uniqueness
        ranks = list(range(1, num_runners + 1))
        random.shuffle(ranks)

        for i in range(num_runners):
            horse_id = available_horses[i]
            jockey_id = available_jockeys[i]
            rank = ranks[i]

            # Get horse and jockey specific data for foreign keys
            horse_info = (
                mock_data["horses"]
                .filter(pl.col("horse_id") == horse_id)
                .row(0, named=True)
            )
            jockey_info = (
                mock_data["jockeys"]
                .filter(pl.col("jockey_id") == jockey_id)
                .row(0, named=True)
            )

            sex = horse_info["sex"]
            birth_date_str = horse_info["birth_date"]
            # Line break for E501
            race_date_str = (
                mock_data["races"]
                .filter(pl.col("race_id") == race_id)["race_date"]
                .item()
            )

            birth_date = datetime.datetime.strptime(birth_date_str, "%Y-%m-%d").date()
            race_date = datetime.datetime.strptime(race_date_str, "%Y-%m-%d").date()

            age = race_date.year - birth_date.year

            results_records.append(
                {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "jockey_id": jockey_id,
                    # Trainer from horse's current info
                    "trainer_id": horse_info["trainer_id"],
                    # Owner from horse's current info
                    "owner_id": horse_info["owner_id"],
                    "rank": rank,
                    "bracket_number": (i % 8) + 1,  # Simple bracket assignment
                    "horse_number": i + 1,
                    "sex_age": f"{sex}{age}",
                    "weight_carried": random.uniform(50.0, 58.0),
                    "distance_diff": random.uniform(-1.0, 1.0),
                    "last_3f_time": random.uniform(33.0, 38.0),
                    "odds_win": random.uniform(1.2, 50.0),
                    "popularity": random.randint(1, num_runners),
                    "horse_weight": random.randint(400, 550),
                    "horse_weight_diff": random.randint(-10, 10),
                }
            )
    mock_data["results"] = pl.DataFrame(results_records)

    return mock_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_dir = "data/mock"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating mock data...")
    mock_dfs = generate_mock_data(num_races=10, num_horses=50, num_jockeys=50)

    for table_name, df in mock_dfs.items():
        output_path = os.path.join(output_dir, f"{table_name}.csv")
        df.write_csv(output_path)
        logger.info(f"Generated {output_path} with {len(df)} rows.")
    logger.info("Mock data generation complete.")
