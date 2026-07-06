"""
Route blueprints for the TonightSky API. Each module owns exactly one
endpoint's request/response handling.
"""
from flask_app.routes.version import version_bp
from flask_app.routes.altitude import altitude_bp
from flask_app.routes.lst import lst_bp
from flask_app.routes.objects import objects_bp
from flask_app.routes.catalog_admin import catalog_admin_bp
from flask_app.routes.client import client_bp


def register_routes(app):
    """Register all API blueprints on the given Flask app."""
    app.register_blueprint(version_bp)
    app.register_blueprint(altitude_bp)
    app.register_blueprint(lst_bp)
    app.register_blueprint(objects_bp)
    app.register_blueprint(catalog_admin_bp)
    app.register_blueprint(client_bp)
