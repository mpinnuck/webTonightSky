"""
Owns the in-memory celestial catalog (loaded from CSV into an Astropy
Table) as a singleton. This is the single place that knows how to
load, hold, and reload that data.
"""
from astropy.table import Table

from core.config import CSV_FILENAME, logger

# Mapping from CSV headers to table headers with formatting info.
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

# Lower-cased lookup used by the query language / filters.
valid_columns = {header.lower(): info for header, info in table_headers.items()}


class CatalogStore:
    """
    Singleton wrapper around the in-memory catalog table.

    Use CatalogStore.instance() to get the shared instance rather than
    constructing CatalogStore() directly.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._table = None
        self._initialized = True

    @classmethod
    def instance(cls) -> "CatalogStore":
        """Return the shared CatalogStore instance."""
        return cls()

    @property
    def table(self):
        """The current in-memory Astropy Table (None until load() succeeds)."""
        return self._table

    def load(self):
        """Load (or reload) the celestial catalog from CSV into memory."""
        try:
            logger.info("Loading catalog...")
            # Handle empty cells and common placeholders during parsing
            loaded = Table.read(
                CSV_FILENAME,
                format="csv",
                fill_values=[("", ""), ("--", "")],  # Map empty cells and '--' to ''
            )
            # Set fill values based on column type
            for col in loaded.colnames:
                if loaded[col].dtype.kind in ("f", "i"):  # Numeric columns
                    loaded[col].fill_value = None
                else:  # String columns
                    loaded[col].fill_value = ""
            loaded = loaded.filled()

            # Log any remaining masked values
            for col in loaded.colnames:
                col_data = loaded[col]
                if hasattr(col_data, "mask") and col_data.mask is not None and any(col_data.mask):
                    masked_indices = [i for i, m in enumerate(col_data.mask) if m]
                    logger.warning(
                        f"Found {len(masked_indices)} masked values in column '{col}' "
                        f"at rows (0-based): {masked_indices[:5]}"
                    )

            self._table = loaded
            logger.info(f"Catalog loaded successfully with {len(self._table)} entries.")
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")

    def reload(self):
        """Reload the catalog from disk (errors are logged, not raised)."""
        self.load()
