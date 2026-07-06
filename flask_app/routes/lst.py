from datetime import datetime, timedelta

import pytz
import astropy.units as u
from astropy.time import Time
from flask import Blueprint, jsonify, request

from core.config import logger

lst_bp = Blueprint("lst", __name__)


@lst_bp.route("/api/calculate_lst", methods=["POST"])
def calculate_lst():
    try:
        data = request.json
        longitude = float(data["longitude"])
        local_time_str = f"{data['date']} {data['local_time']}"
        # Strip any seconds if the user supplied them
        local_time_str = ":".join(local_time_str.split(":")[:2])
        timezone = pytz.timezone(data["timezone"])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        # Check if the time is in AM and should be considered as the next day
        if local_time.hour < 12:
            local_time = local_time + timedelta(days=1)

        utc_time = local_time.astimezone(pytz.utc)
        lst_hours = Time(utc_time).sidereal_time("mean", longitude * u.deg).hour
        return jsonify({"LST": f"{int(lst_hours):02}:{int((lst_hours*60)%60):02}:{int((lst_hours*3600)%60):02}"})
    except Exception as e:
        logger.error(f"Error calculating LST: {e}")
        return jsonify({"error": "Failed to calculate Local Sidereal Time"}), 500
