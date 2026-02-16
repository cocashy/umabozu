import os
import sys
from typing import Any  # F821

import polars as pl

try:
    from umabozu.database.crud import get_training_data
except (ImportError, ValueError):
    # Add project root to sys.path for relative imports
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )
    from src.umabozu.database.crud import get_training_data


def load_mock_data(data_dir: str = "data/mock") -> dict[str, pl.DataFrame]:
    """
    Loads mock data from CSV files into Polars DataFrames.
    """
    data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            table_name = filename.replace(".csv", "")
            file_path = os.path.join(data_dir, filename)
            data[table_name] = pl.read_csv(file_path)
    return data


def load_data_from_db() -> pl.DataFrame:
    """Loads data from the SQLite database."""
    return get_training_data()
