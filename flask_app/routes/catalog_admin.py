from flask import Blueprint, jsonify

from core.config import logger
from core.catalog import CatalogStore

catalog_admin_bp = Blueprint("catalog_admin", __name__)


@catalog_admin_bp.route("/api/reload_catalog", methods=["GET", "POST"])
def reload_catalog():
    """Endpoint to reload the celestial catalog from the CSV file."""
    try:
        CatalogStore.instance().reload()
        return jsonify({"status": "success", "message": "Catalog reloaded successfully"}), 200
    except Exception as e:
        logger.error(f"Error reloading catalog: {e}")
        return jsonify({"status": "error", "message": "Failed to reload catalog"}), 500
