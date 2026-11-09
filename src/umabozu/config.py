# src/umabozu/config.py
import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Database configuration
DATABASE_FILE = os.path.join(BASE_DIR, "data", "umabozu.db")

# Keep the high-coverage default; strategy tuning also enforces a coverage floor.
DEFAULT_MIN_WIN_PROBABILITY = 0.13
MIN_STRATEGY_RACE_COVERAGE = 0.95

# Logging configuration
LOG_FILE = os.path.join(BASE_DIR, "src", "logs", "umabozu.log")
LOG_LEVEL = "INFO"
