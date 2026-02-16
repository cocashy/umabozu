import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def check_db():
    files = [
        "data/umabozu.db",
    ]

    for f in files:
        if os.path.exists(f):
            logger.info(f"Checking {f}...")
            conn = sqlite3.connect(f)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                r = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='races'"
                ).fetchone()
                logger.info(f"  Table 'races' exists: {r is not None}")
                if r:
                    count = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
                    logger.info(f"  Row count in 'races': {count}")
                    if count > 0:
                        sample = conn.execute("SELECT * FROM races LIMIT 1").fetchone()
                        logger.info(
                            "  Sample row: %s",
                            dict(
                                zip(
                                    [
                                        d[0]
                                        for d in conn.execute(
                                            "SELECT * FROM races LIMIT 1"
                                        ).description
                                    ],
                                    sample,
                                    strict=True,
                                )
                            ),
                        )
            except Exception as e:
                logger.error(f"  Error: {e}")
            finally:
                conn.close()
        else:
            logger.warning(f"{f} does not exist.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_db()
