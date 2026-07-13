from datetime import datetime, timedelta

import pytz
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroplan import Observer, FixedTarget
from flask import Blueprint, jsonify, request

from core.config import logger
from core.astro_calc import ra_to_degrees

altitude_bp = Blueprint("altitude", __name__)


@altitude_bp.route("/api/altitude_data", methods=["POST"])
def altitude_data():
    try:
        data = request.json
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        ra = ra_to_degrees(data["ra"])
        dec = float(data["dec"].replace("°", "").strip())
        date_str = data["date"]
        timezone = pytz.timezone(data["timezone"])

        # Setup observer and location
        date = timezone.localize(datetime.strptime(date_str, "%Y-%m-%d"))
        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
        observer = Observer(location=location, timezone=timezone)
        target = FixedTarget(coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg))

        # Calculate sunset, sunrise, astronomical dusk, and dawn explicitly for the correct dates
        sunset = observer.sun_set_time(Time(date), which="next").to_datetime(timezone)
        sunrise = observer.sun_rise_time(Time(date + timedelta(days=1)), which="next").to_datetime(timezone)
        astro_dusk = observer.twilight_evening_astronomical(Time(date), which="next").to_datetime(timezone)
        astro_dawn = observer.twilight_morning_astronomical(Time(date + timedelta(days=1)), which="next").to_datetime(timezone)

        # Adjust start time before sunset
        start_hour = sunset.replace(minute=0, second=0, microsecond=0)
        if (sunset - start_hour).total_seconds() < 600:  # If less than 10 minutes before sunset
            start_time = start_hour - timedelta(minutes=30)
        else:
            start_time = start_hour

        # Adjust end time after sunrise
        end_hour = sunrise.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if (end_hour - sunrise).total_seconds() < 600:  # If less than 10 minutes after sunrise
            end_time = sunrise + timedelta(minutes=30)
        else:
            end_time = end_hour

        # Define the time range, every 10 minutes
        times = []
        current_time = start_time
        while current_time <= end_time:
            times.append(current_time)
            current_time += timedelta(minutes=10)

        # Calculate altitudes and azimuths for each time point
        altitudes = []
        azimuths = []
        for t in times:
            altaz = observer.altaz(Time(t), target)
            altitudes.append(altaz.alt.deg)
            azimuths.append(altaz.az.deg)

        # Calculate approximate transit time for the target
        transit_time = observer.target_meridian_transit_time(
            Time(date), target, which="next"
        ).to_datetime(timezone).strftime("%H:%M:%S")

        response_data = {
            "times": [t.strftime("%H:%M") for t in times],
            "altitudes": altitudes,
            "azimuths": azimuths,
            "sunset": sunset.strftime("%H:%M:%S"),
            "sunrise": sunrise.strftime("%H:%M:%S"),
            "astronomical_dusk": astro_dusk.strftime("%H:%M:%S"),
            "astronomical_dawn": astro_dawn.strftime("%H:%M:%S"),
            "transit_time": transit_time,
        }

        return jsonify(response_data)

    except ValueError as e:
        logger.error(f"Value error in altitude_data: {e}")
        return jsonify({"error": "Invalid input data. Please check latitude, longitude, RA, Dec, or date formatting."}), 400

    except pytz.UnknownTimeZoneError as e:
        logger.error(f"Timezone error in altitude_data: {e}")
        return jsonify({"error": "Invalid timezone. Please check the timezone value."}), 400

    except Exception as e:
        logger.error(f"General error in altitude_data: {e}")
        return jsonify({"error": "An error occurred while calculating altitude data"}), 500
