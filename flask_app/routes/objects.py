import json
import time
from datetime import datetime, timedelta

import pytz
from flask import Blueprint, Response, jsonify, request, stream_with_context
from numpy.ma.core import MaskedConstant

from core.config import logger
from core.catalog import CatalogStore, valid_columns
from core.astro_calc import (
    calc_time_location_and_lst,
    calc_transit_and_alt_az,
    calc_transit_altitude,
    degrees_to_ra,
    format_dec,
    format_transit_time,
)
from core.query_language import parse_query_conditions, evaluate_conditions

objects_bp = Blueprint("objects", __name__)


@objects_bp.route("/api/list_objects", methods=["POST"])
def list_objects():
    """
    Handle requests to list celestial objects based on user filters and
    catalog selections.
    """
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        start_time = time.perf_counter()
        data = request.json
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        local_time_str = f"{data['date']} {data['local_time']}"
        # Strip any seconds if the user supplied them
        local_time_str = ":".join(local_time_str.split(":")[:2])
        timezone = pytz.timezone(data["timezone"])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        # Check if the time is in AM and should be considered as the next day
        if local_time.hour < 12:
            local_time = local_time + timedelta(days=1)

        filter_expression = data.get("filter_expression", "")
        catalog_filters = data.get("catalogs", {})

        # Parse the filter expression into conditions
        try:
            conditions = parse_query_conditions(filter_expression, valid_columns) if filter_expression else []
        except ValueError as e:
            logger.error(f"Query parsing error: {e}")
            return jsonify({"error": f"Query parsing error: {e}"}), 400

        # Check if all catalogs are unselected (treat as select all if true)
        all_catalogs_unselected = all(not selected for selected in catalog_filters.values()) if catalog_filters else True

        catalog_table = CatalogStore.instance().table

        def generate():
            row_count = 0
            included_count = 0

            astropy_time, location, altaz, lst = calc_time_location_and_lst(latitude, longitude, local_time)

            for row in catalog_table:
                row_count += 1

                # Apply catalog filter
                catalog_name = row["Catalog"].strip()
                if not all_catalogs_unselected and catalog_filters and not catalog_filters.get(catalog_name, False):
                    continue

                # Extract RA/Dec for calculations
                ra = float(row["RA"])
                dec = float(row["Dec"])

                # Calculate transit time and AltAz
                transit_time_minutes, local_transit_time, before_after, altitude, azimuth, direction = calc_transit_and_alt_az(
                    ra, dec, local_time, astropy_time, location, altaz, lst
                )

                if altitude < 0:
                    continue

                transit_alt = calc_transit_altitude(ra, dec, latitude, longitude)

                # Build the row object
                current_row = {
                    "Name": row["Name"],
                    "RA": degrees_to_ra(ra),
                    "Dec": format_dec(dec),
                    "Transit Time": local_transit_time,
                    "Transit Alt": f"{transit_alt:.2f}",
                    "Direction": direction,
                    "Relative TT": format_transit_time(transit_time_minutes),
                    "Before/After": before_after,
                    "Altitude": f"{altitude:.2f}",
                    "Azimuth": f"{azimuth:.2f}",
                    "Alt Name": row.get("Alt Name", ""),
                    "Type": row["Type"],
                    "Magnitude": row["Magnitude"],
                    "Size": row["Size"],
                    "Info": row["Info"],
                    "Catalog": row["Catalog"],
                }

                # Check for MaskedConstant values and log the row if found
                for key, value in current_row.items():
                    if isinstance(value, MaskedConstant):
                        logger.error(
                            f"Found MaskedConstant in row {row_count}: "
                            f"Column='{key}', Value={value}, Row={dict(row)}"
                        )

                # Evaluate the row against conditions
                if not evaluate_conditions(current_row, conditions):
                    continue

                # add degrees symbols
                current_row["Altitude"] += "°"
                current_row["Azimuth"] += "°"

                included_count += 1
                try:
                    yield json.dumps(current_row) + "\n"
                except TypeError as e:
                    logger.error(
                        f"JSON serialization failed at row {row_count}: {e}\n"
                        f"Row data: {dict(row)}\n"
                        f"Current row: {current_row}"
                    )
                    continue

            elapsed_time = time.perf_counter() - start_time
            logger.debug(f"Total rows processed: {row_count}")
            logger.debug(f"Objects returned: {included_count} in {elapsed_time:.4f} seconds")

        return Response(stream_with_context(generate()), content_type="application/json")

    except KeyError as e:
        logger.error(f"Missing data field: {e}")
        return jsonify({"error": f"Missing data field: {e}"}), 400
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({"error": f"Value error: {e}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
