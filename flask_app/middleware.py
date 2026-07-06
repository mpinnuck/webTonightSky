"""
Flask request-level middleware (currently: access logging).
"""
import json

from flask import request

from core.config import logger


def register_request_logging(app):
    """Attach a before_request hook that logs details of each request."""

    @app.before_request
    def log_request_info():
        client_ip = request.remote_addr or "Unknown IP"
        method = request.method
        endpoint = request.endpoint or "Unknown Endpoint"
        url = request.url
        headers = dict(request.headers)
        data = request.json if request.is_json else request.form.to_dict()

        formatted_headers = json.dumps(headers, indent=4)
        formatted_data = json.dumps(data, indent=4)

        logger.info(
            f"\nAccess Log:\n"
            f"    IP: {client_ip}\n"
            f"    Method: {method}\n"
            f"    Endpoint: {endpoint}\n"
            f"    URL: {url}\n"
            f"    Headers: \n{formatted_headers}\n"
            f"    Data: \n{formatted_data}"
        )
