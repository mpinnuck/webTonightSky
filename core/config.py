"""
Application-wide configuration: constants and logging setup.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
VERSION = "6.0"
CSV_FILENAME = "./data/celestial_catalog.csv"
# Single server-wide Stellarium .hrz polygonal horizon file, matching
# the single-user deployment model already used for the catalog CSV.
# If missing, core.horizon.HorizonProfile.load() falls back to a flat
# 0 degree horizon so Tonight's Best keeps working without one.
HORIZON_FILENAME = "./data/horizon.hrz"
# Fallback minimum-useful altitude when no horizon.hrz is configured.
# Matches HORIZON_ALTITUDE in client/static/index.html, the app's
# existing "usable sky" threshold used by the altitude graph - keeping
# Tonight's Best consistent with it until a real per-site obstruction
# profile is deployed.
DEFAULT_HORIZON_ALTITUDE_DEG = 25.0

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
