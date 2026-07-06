from flask import Blueprint, jsonify

from core.config import VERSION

version_bp = Blueprint("version", __name__)


@version_bp.route("/")
def home():
    return "TonightSky app is running!", 200


@version_bp.route("/api/version", methods=["GET"])
def get_version():
    return jsonify({"version": VERSION})
