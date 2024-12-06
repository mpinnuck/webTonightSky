import flask
from flask import Flask, jsonify, request
from flask import Response, stream_with_context
from flask_cors import CORS
from flask import Response, stream_with_context
from datetime import datetime, timedelta
import pytz
import re
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
from astropy.table import Table
from astroplan import Observer, FixedTarget
import json
import time
import os
import logging
from logging.handlers import RotatingFileHandler  # Ensure this is imported
import pyparsing as pp
from pyparsing import ParseException
from pyparsing import (
    Word,
    alphas,
    alphanums,
    oneOf,
    quotedString,
    removeQuotes,
    infixNotation,
    opAssoc,
    Group,
    ParseException,
    CaselessKeyword,
)


###########################################################
#   GLOBALS
#
# Constants
VERSION = "2.0"

#############################
# Set up logging
# Ensure the log directory exists
log_directory = "./logs"
os.makedirs(log_directory, exist_ok=True)

# Configure Rotating File Handler
log_file = os.path.join(log_directory, "tonightsky.log")
rotating_handler = RotatingFileHandler(
    log_file,       # Log file path
    maxBytes=5 * 1024 * 1024,  # Maximum size of each log file (5MB in this case)
    backupCount=1   # Keep 1 old log file
)

# Set formatter for the rotating handler
rotating_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

# Create the logger
logger = logging.getLogger("TonightSky")
logger.setLevel(logging.DEBUG)  # Set the logging level
logger.addHandler(rotating_handler)

logger.info("Logging initialized with file rotation.")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

csv_filename = './data/celestial_catalog.csv'
catalog_table = None  # Global in-memory Astropy Table


# Define a mapping from CSV headers to table headers with formatting info
table_headers = {
    "Name": {"name": "Name", "type": "string"},
    "RA": {"name": "RA", "type": "time"},
    "Dec": {"name": "Dec", "type": "string"},  
    "Transit Time": {"name": "Transit Time", "type": "time"},
    "Direction": {"name": "Direction", "type": "string"},
    "Relative TT": {"name": "Relative TT", "type": "time"},
    "Before/After": {"name": "Before/After", "type": "string"},
    "Altitude": {"name": "Altitude", "type": "float"},
    "Azimuth": {"name": "Azimuth", "type": "float"},
    "Alt Name": {"name": "Alt Name", "type": "string"},
    "Type": {"name": "Type", "type": "string"},
    "Magnitude": {"name": "Magnitude", "type": "float"},
    "Info": {"name": "Info", "type": "string"},
    "Catalog": {"name": "Catalog", "type": "string"}
}

# Create valid_columns from table_headers
valid_columns = {header.lower(): info for header, info in table_headers.items()}

########## end GLOBALS ##################
logger.info(f"TonightSky version: {VERSION}")
logger.info(f"Flask version: {flask.__version__}")
logger.info(f"Type of app: {type(app)}")
logger.info(f"App configuration: {app.config}")

@app.before_request
def log_request_info():
    """
    Log details about each incoming request, including IP, endpoint, and query parameters.
    """
    client_ip = request.remote_addr or "Unknown IP"
    method = request.method
    endpoint = request.endpoint or "Unknown Endpoint"
    url = request.url
    headers = dict(request.headers)
    data = request.json if request.is_json else request.form.to_dict()

    # Format the headers and data for better readability
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

def load_catalog():
    global catalog_table
    try:
        catalog_table = Table.read(csv_filename, format='csv')
        catalog_table = catalog_table.filled("")
        logger.info("Catalog loaded successfully with %d entries.", len(catalog_table))
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")

# Initialize resources before the app starts handling requests
with app.app_context():
    load_catalog()  # Call your initialization code here

@app.route('/')
def home():
    return "TonightSky app is running!", 200

@app.route('/api/version', methods=['GET'])
def get_version():
    return jsonify({"version": VERSION})

# Utility function to convert RA from HH:MM:SS to decimal degrees
def ra_to_degrees(ra_str):
    hours, minutes, seconds = map(float, ra_str.split(':'))
    return (hours + minutes / 60 + seconds / 3600) * 15  # Convert to degrees


@app.route('/api/altitude_data', methods=['POST'])
def altitude_data():
    try:
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        ra = ra_to_degrees(data['ra'])
        dec = float(data['dec'].replace('°', '').strip())
        date_str = data['date']
        timezone = pytz.timezone(data['timezone'])

        # Setup observer and location
        date = timezone.localize(datetime.strptime(date_str, "%Y-%m-%d"))
        location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
        observer = Observer(location=location, timezone=timezone)
        target = FixedTarget(coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg))

        # Calculate sunset, sunrise, astronomical dusk, and dawn explicitly for the correct dates
        sunset = observer.sun_set_time(Time(date), which='next').to_datetime(timezone)
        sunrise = observer.sun_rise_time(Time(date + timedelta(days=1)), which='next').to_datetime(timezone)
        astro_dusk = observer.twilight_evening_astronomical(Time(date), which='next').to_datetime(timezone)
        astro_dawn = observer.twilight_morning_astronomical(Time(date + timedelta(days=1)), which='next').to_datetime(timezone)

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

        # Calculate altitudes for each time point
        altitudes = []
        for t in times:
            altaz = observer.altaz(Time(t), target)
            altitudes.append(altaz.alt.deg)

        # Calculate approximate transit time for the target
        transit_time = observer.target_meridian_transit_time(Time(date), target, which='next').to_datetime(timezone).strftime("%H:%M:%S")

        # Prepare response data
        response_data = {
            "times": [t.strftime("%H:%M") for t in times],
            "altitudes": altitudes,
            "sunset": sunset.strftime("%H:%M:%S"),
            "sunrise": sunrise.strftime("%H:%M:%S"),
            "astronomical_dusk": astro_dusk.strftime("%H:%M:%S"),
            "astronomical_dawn": astro_dawn.strftime("%H:%M:%S"),
            "transit_time": transit_time
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
    direction = "south" if azimuth > 90 and azimuth < 270 else "north"
    return transit_time_minutes, local_transit_time.strftime("%H:%M:%S"), before_after, altitude, azimuth, direction

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
    """
    Parse the query string into a list of conditions.
    """
    # Define the grammar for the query language
    column_name = pp.oneOf(list(valid_columns.keys()), caseless=True)
    operator = oneOf("> >= < <= = != like", caseless=True)

    # Define a value parser for time-like columns
    time_value = Word(alphanums + ":")

    # Define a condition with the time_value parser for specific columns
    time_condition = Group(
        pp.oneOf([k.lower() for k in table_headers if table_headers[k]['type'] == 'time'], caseless=True) +
        operator + 
        time_value
    )

    value = (
        quotedString.setParseAction(removeQuotes) | 
        Word(alphanums + ".°") | 
        Word(alphas)
    )

    # Regular condition for other columns
    condition = Group(column_name + operator + value)

    # Define the structure for AND and OR expressions (using symbols)
    and_ = CaselessKeyword("and") | "*" | "&"
    or_ = CaselessKeyword("or") | "|" | "+"

    # Handle nested expressions and single conditions
    expr = infixNotation(
        time_condition | condition,  # Include time_condition in infixNotation
        [
            (and_, 2, opAssoc.LEFT),
            (or_, 2, opAssoc.LEFT),
        ],
    )

    try:
        # Parse the query string
        result = expr.parseString(query)

        # Check if the result is a single condition (not nested)
        if len(result) == 3 and isinstance(result[0], str) and isinstance(result[1], str) and isinstance(result[2], str):
            parsed_query = [pp.ParseResults(result)]  # Wrap it in a list after converting to pp.ParseResults
        else:
            parsed_query = result  # No need to access [0] here

        # --- Flatten the parsed query ---
        def flatten_parsed_query(parsed_expr):
            flattened = []
            for item in parsed_expr:
                if isinstance(item, pp.ParseResults) and isinstance(item[0], pp.ParseResults):
                    flattened.extend(flatten_parsed_query(item))
                else:
                    flattened.append(item)
            return flattened

        parsed_query = flatten_parsed_query(parsed_query)

    except ParseException as e:
        raise ValueError(f"Invalid query syntax: {e}")

    def extract_conditions(parsed_expr):
        """
        Recursively extract conditions from the parsed expression.
        """
        conditions = []
        for item in parsed_expr:
            if isinstance(item, pp.ParseResults) and len(item) == 3:
                column, operator, value = item

                # Get column information from valid_columns
                column_info = valid_columns.get(column.lower())
                if not column_info:
                    raise ValueError(f"Invalid column: {column}")
                column_name = column_info["name"]
                column_type = column_info["type"]

                # Normalize time values
                # Normalize time values
                if column_type == "time":
                    if ":" not in value:  # Add hh:mm:ss if no colon is present
                        value += ":00:00"
                    elif value.count(":") == 1:  # Add :ss if only one colon is present
                        value += ":00" 
                                # Convert to float if it's a float column
                try:
                    if column_type == "float":
                        value = float(value)
                except ValueError:
                    value = value.lower()

                conditions.append((column_name, operator, value))

            elif isinstance(item, str) and item in ("and", "or", "*", "+", "&", "|"):
                # Normalize logical operators
                conditions.append("&" if item in ("and", "*", "&") else "|")  
            else:
                raise ValueError(f"Invalid element in parsed expression: {item}")

        return conditions

    return extract_conditions(parsed_query)

# Define a dictionary to map operators to functions
operator_functions = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "like": lambda a, b: isinstance(a, str) and b in a,
}

def evaluate_conditions(row, conditions):
    """
    Evaluate a list of conditions against a row with optimization.
    """
    if not conditions:
        return True

    def evaluate_condition(row, condition):
        """
        Evaluate a single condition against a row.
        """
        column, operator, value = condition
        row_value = row[column]  # Access directly, no lowercasing

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


        # Get the function for the current operator
        operator_function = operator_functions.get(operator)
        if operator_function:
            # Call the function with appropriate type handling
            if is_numeric(row_value) and is_numeric(value):
                return operator_function(float(row_value), float(value))
            else:
                return operator_function(str(row_value).lower(), str(value).lower())
        else:
            raise ValueError(f"Invalid operator: {operator}")

    # Evaluate the conditions using the appropriate logic (& and |) with optimization
    result = evaluate_condition(row, conditions[0])
    for i in range(1, len(conditions), 2):
        logic_op = conditions[i]  # No need to lowercase or check for other symbols here
        if logic_op == "|":
            if result:
                return True
            # No else here, continue to next condition
        next_result = evaluate_condition(row, conditions[i + 1])
        if logic_op == "&":
            result = result and next_result
        elif logic_op == "|":
            result = result or next_result
        else:
            raise ValueError(f"Invalid logic operator: {logic_op}")

    return result

@app.route('/api/list_objects', methods=['POST'])
def list_objects():
    """
    Handle requests to list celestial objects based on user filters and catalog selections.
    """
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200  # Return a 200 OK for preflight

    try:
        start_time = time.perf_counter()  # Record the start time
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        local_time_str = f"{data['date']} {data['local_time']}"
        timezone = pytz.timezone(data['timezone'])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        filter_expression = data.get("filter_expression", "")
        catalog_filters = data.get("catalogs", {})

        # Check if all catalogs are unselected (treat as select all if true)
        all_catalogs_unselected = all(not selected for selected in catalog_filters.values()) if catalog_filters else True

        def generate():

            # Parse the filter expression into conditions
            try:
                conditions = parse_query_conditions(filter_expression, valid_columns) if filter_expression else []
            except ValueError as e:
                logger.error(f"Query parsing error: {e}")
                yield json.dumps({"error": f"Query parsing error: {e}"})
                return

            row_count = 0
            included_count = 0

            for row in catalog_table:
                row_count += 1
                #if row_count % 500 == 0:
                #    logger.debug(f"Processed {row_count} rows...")

                # Apply catalog filter
                catalog_name = row['Catalog'].strip()
                if not all_catalogs_unselected and catalog_filters and not catalog_filters.get(catalog_name, False):
                    continue

                # Extract RA/Dec for calculations
                ra = float(row['RA'])
                dec = float(row['Dec'])

                # Calculate transit time and AltAz
                transit_time_minutes, local_transit_time, before_after, altitude, azimuth, direction = calculate_transit_and_alt_az(
                    ra, dec, latitude, longitude, local_time)

                if altitude < 0:
                    continue

                # Build the row object
                current_row = {
                    'Name': row['Name'],
                    'RA': degrees_to_ra(ra),
                    'Dec': format_dec(dec),
                    'Transit Time': local_transit_time,
                    'Direction': direction,
                    'Relative TT': format_transit_time(transit_time_minutes),
                    'Before/After': before_after,
                    'Altitude': f"{altitude:.2f}°",
                    'Azimuth': f"{azimuth:.2f}°",
                    'Alt Name': row.get('Alt Name', ''),
                    'Type': row['Type'],
                    'Magnitude': row['Magnitude'],
                    'Info': row['Info'],
                    'Catalog': row['Catalog']
                }

                # Evaluate the row against conditions
                if not evaluate_conditions(current_row, conditions):
                    continue

                # Stream the matching object as a JSON object
                included_count += 1
                yield json.dumps(current_row) + "\n"
                
            elapsed_time = time.perf_counter() - start_time
            logger.debug(f"Total rows processed: {row_count}")
            logger.debug(f"Objects returned: {included_count} in {elapsed_time:.4f} seconds")

        # Stream the response back to the client
        return  Response(stream_with_context(generate()), content_type='application/json')

        # Log the elapsed time

        return response

    except KeyError as e:
        logger.error(f"Missing data field: {e}")
        return jsonify({"error": f"Missing data field: {e}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

   
if __name__ == "__main__":
    logger.info(f"main() TonightSky version: {VERSION}")
    # Determine the Flask environment
    flask_env = os.environ.get("FLASK_ENV", "production").lower()  # Default to 'production'

    print(f"Environment detected: {flask_env}")

    if flask_env == "development":
        # Run in development mode with debug enabled
        app.run(debug=True)
    else:
        # Run in production mode without debug
        app.run(debug=False)
    