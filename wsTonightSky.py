"""
TonightSky application entry point.

Responsible only for: creating the Flask app, wiring up CORS and
request-logging middleware, loading the catalog at startup, and
registering route blueprints. Flask-specific modules (routes,
middleware) live under flask_app/; pure domain logic modules
(config, catalog, astro_calc, query_language) live under core/,
since they have no Flask dependency.
"""
import flask
from flask import Flask
from flask_cors import CORS

from core.config import VERSION, logger
from core.catalog import CatalogStore
from flask_app.middleware import register_request_logging
from flask_app.routes import register_routes

app = Flask(__name__, static_folder="client/static", static_url_path="/app")
CORS(app, resources={r"/api/*": {"origins": "*"}})

register_request_logging(app)
register_routes(app)

logger.info(f"TonightSky version: {VERSION}")
logger.info(f"Flask version: {flask.__version__}")
logger.info(f"Type of app: {type(app)}")
logger.info(f"App configuration: {app.config}")

# Load the catalog once at startup
with app.app_context():
    CatalogStore.instance().load()
