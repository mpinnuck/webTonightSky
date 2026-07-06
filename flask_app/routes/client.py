from flask import Blueprint, redirect

client_bp = Blueprint("client", __name__)


@client_bp.route("/app")
@client_bp.route("/app/")
def client_index():
    """Convenience redirect so /app loads the client's index page."""
    return redirect("/app/index.html")
