import polars as pl

from umabozu.database.crud import get_training_data


def load_data_from_db() -> pl.DataFrame:
    """Loads data from the SQLite database."""
    return get_training_data()
