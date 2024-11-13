from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import pytz
import csv
import re
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TonightSky")

app = Flask(__name__)
CORS(app)

csv_filename = './data/celestial_catalog.csv'

# Define a mapping from CSV headers to table headers
table_headers = {
    "Name": "Name",
    "RA": "RA",
    "Dec": "Dec",
    "Transit Time": "Transit Time",
    "Relative TT": "Relative TT",
    "Before/After": "Before/After",
    "Altitude": "Altitude",
    "Azimuth": "Azimuth",
    "Alt Name": "Alt Name",
    "Type": "Type",
    "Magnitude": "Magnitude",
    "Info": "Info",
    "Catalog": "Catalog"
}

@app.route('/api/calculate_lst', methods=['POST'])
def calculate_lst():
    try:
        data = request.json
        longitude = float(data['longitude'])
        local_time_str = f"{data['date']} {data['local_time']}"
        timezone = pytz.timezone(data['timezone'])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        utc_time = local_time.astimezone(pytz.utc)
        lst_hours = Time(utc_time).sidereal_time('mean', longitude * u.deg).hour
        return jsonify({"LST": f"{int(lst_hours):02}:{int((lst_hours*60)%60):02}:{int((lst_hours*3600)%60):02}"})
    except Exception as e:
        logger.error(f"Error calculating LST: {e}")
        return jsonify({"error": "Failed to calculate Local Sidereal Time"}), 500

# Helper functions
def calculate_transit_and_alt_az(ra_deg, dec_deg, latitude, longitude, local_time):
    astropy_time = Time(local_time.astimezone(pytz.utc))
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz = AltAz(obstime=astropy_time, location=location)
    altaz_coord = target.transform_to(altaz)
    altitude = altaz_coord.alt.deg
    azimuth = altaz_coord.az.deg
    lst = astropy_time.sidereal_time('mean', longitude * u.deg).hour
    ra_hours = ra_deg / 15.0
    time_diff_hours = ra_hours - lst
    if time_diff_hours > 12:
        time_diff_hours -= 24
    elif time_diff_hours < -12:
        time_diff_hours += 24
    before_after = "After" if time_diff_hours >= 0 else "Before"
    transit_time_minutes = abs(time_diff_hours * 60)
    local_transit_time = local_time + timedelta(minutes=transit_time_minutes if before_after == "After" else -transit_time_minutes)
    return transit_time_minutes, local_transit_time.strftime("%H:%M:%S"), before_after, altitude, azimuth

def degrees_to_ra(degrees):
    hours = int(degrees // 15)
    minutes = int((degrees % 15) * 4)
    seconds = (degrees % 15) * 240 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"

def format_dec(dec):
    return f"{dec:.2f}°"

def format_transit_time(transit_time_minutes):
    time_to_transit_seconds = abs(transit_time_minutes * 60)
    hours = int(time_to_transit_seconds // 3600)
    minutes = int((time_to_transit_seconds % 3600) // 60)
    seconds = int(time_to_transit_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_query_conditions(query, valid_columns):
    pattern = r"([a-zA-Z_][\w\s]*)\s*(>|>=|<|<=|=|!=|like)\s*('[^']*'|\"[^\"]*\"|[\w\.]+)\s*(AND|OR|\+|\|)?"
    conditions = []
    for match in re.finditer(pattern, query, re.IGNORECASE):
        column, operator, value, logic_op = match.groups()
        column = column.lower().strip()
        if column in valid_columns:
            conditions.append((valid_columns[column], operator, value.strip("'\""), logic_op))
        else:
            raise ValueError(f"Invalid column: {column}")
    return conditions

def evaluate_conditions(row, conditions):
    if not conditions:
        return True
    for column, operator, value, logic_op in conditions:
        row_value = row[column].strip('°')
        def is_numeric(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        if is_numeric(row_value) and is_numeric(value):
            row_value = float(row_value)
            value = float(value)
        else:
            row_value = str(row_value).lower()
            value = str(value).lower()
        if operator == '>' and not row_value > value:
            return False
        elif operator == '>=' and not row_value >= value:
            return False
        elif operator == '<' and not row_value < value:
            return False
        elif operator == '<=' and not row_value <= value:
            return False
        elif operator == '=' and not row_value == value:
            return False
        elif operator == '!=' and not row_value != value:
            return False
        elif operator == 'like' and isinstance(row_value, str) and value not in row_value:
            return False
    return True

@app.route('/api/list_objects', methods=['POST'])
def list_objects():
    try:
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        local_time_str = f"{data['date']} {data['local_time']}"
        timezone = pytz.timezone(data['timezone'])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        filter_expression = data.get("filter_expression", "")
        catalog_filters = data.get("catalogs", {})

        objects = []
        with open(csv_filename, mode='r', encoding='ISO-8859-1') as file:
            reader = csv.DictReader(file)

            # Use table headers as valid columns for the query parser
            valid_columns = {header.lower(): header for header in table_headers.keys()}

            # Parse the filter expression into conditions
            try:
                conditions = parse_query_conditions(filter_expression, valid_columns) if filter_expression else []
            except ValueError as e:
                logger.error(f"Query parsing error: {e}")
                return jsonify({"error": f"Query parsing error: {e}"}), 400

            count = 0  # Initialize a counter for tracking progress

            for row in reader:
                count += 1
                if count % 500 == 0:
                    logger.debug(f"Processed {count} rows")

                # Apply catalog filter (check if the object's catalog is enabled)
                catalog = row['Catalog'].strip()
                if catalog_filters and not catalog_filters.get(catalog, False):
                    continue

                # Calculate transit time and AltAz for each object
                ra = float(row['RA'])
                dec = float(row['Dec'])
                transit_time_minutes, local_transit_time, before_after, altitude, azimuth = calculate_transit_and_alt_az(
                    ra, dec, latitude, longitude, local_time)

                if altitude < 0:
                    continue

                # Build the full row object for condition evaluation
                current_row = {
                    'Name': row['Name'],
                    'RA': degrees_to_ra(ra),
                    'Dec': format_dec(dec),
                    'Transit Time': local_transit_time,
                    'Relative TT': format_transit_time(transit_time_minutes),
                    'Before/After': before_after,
                    'Altitude': f"{altitude:.2f}°",  # Calculated Altitude in degrees
                    'Azimuth': f"{azimuth:.2f}°",
                    'Alt Name': row.get('Alt Name', ''),
                    'Type': row['Type'],
                    'Magnitude': row['Magnitude'],
                    'Info': row['Info'],
                    'Catalog': row['Catalog']
                }

                # Evaluate the row against the conditions
                if evaluate_conditions(current_row, conditions):
                    objects.append(current_row)

        logger.debug(f"Total rows processed: {count}")
        logger.debug(f"Number of objects returned: {len(objects)}")
        return jsonify(objects)

    except KeyError as e:
        logger.error(f"Missing data field: {e}")
        return jsonify({"error": f"Missing data field: {e}"}), 400
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({"error": f"Value error: {e}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
