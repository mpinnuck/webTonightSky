"""
Application-wide configuration: constants and logging setup.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
VERSION = "5.1"
CSV_FILENAME = "./data/celestial_catalog.csv"

# -----------------------------------------------------------------
# Logging
# -----------------------------------------------------------------
def _build_logger():
    log_directory = "./logs"
    os.makedirs(log_directory, exist_ok=True)

    log_file = os.path.join(log_directory, "tonightsky.log")
    rotating_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB per file
        backupCount=1,
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    rotating_handler.setFormatter(formatter)

    log = logging.getLogger("TonightSky")
    log.setLevel(logging.DEBUG)
    log.addHandler(rotating_handler)

    if os.environ.get("FLASK_DEBUG") == "1":
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        log.info("Console logging enabled in development mode")

    log.info("Logging initialized with file rotation.")
    return log


logger = _build_logger()
