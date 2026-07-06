"""
Application-wide configuration: constants, logging setup, and the
catalog column schema (table_headers / valid_columns).
"""
import os
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
VERSION = "5.0"
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

# -----------------------------------------------------------------
# Catalog column schema
# -----------------------------------------------------------------
# Mapping from CSV headers to table headers with formatting info
table_headers = {
    "Name": {"name": "Name", "type": "string"},
    "RA": {"name": "RA", "type": "time"},
    "Dec": {"name": "Dec", "type": "string"},
    "Transit Time": {"name": "Transit Time", "type": "time"},
    "Transit Alt": {"name": "Transit Alt", "type": "float"},
    "Direction": {"name": "Direction", "type": "string"},
    "Relative TT": {"name": "Relative TT", "type": "time"},
    "Before/After": {"name": "Before/After", "type": "string"},
    "Altitude": {"name": "Altitude", "type": "float"},
    "Azimuth": {"name": "Azimuth", "type": "float"},
    "Alt Name": {"name": "Alt Name", "type": "string"},
    "Type": {"name": "Type", "type": "string"},
    "Magnitude": {"name": "Magnitude", "type": "float"},
    "Size": {"name": "Size", "type": "float"},
    "Info": {"name": "Info", "type": "string"},
    "Catalog": {"name": "Catalog", "type": "string"},
}

# Lower-cased lookup used by the query language / filters
valid_columns = {header.lower(): info for header, info in table_headers.items()}
