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
from numpy.ma.core import MaskedConstant 
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
VERSION = "4.3"
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
formatter = rotating_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

# Create the logger
logger = logging.getLogger("TonightSky")
logger.setLevel(logging.DEBUG)  # Set the logging level
logger.addHandler(rotating_handler)

# Add console handler in development mode
if os.environ.get("FLASK_DEBUG") == "1":
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Console logging enabled in development mode")
    
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
    "Transit Alt": {"name": "Transit Alt", "type": "float"},
    "Direction": {"name": "Direction", "type": "string"},
    "Relative TT": {"name": "Relative TT", "type": "time"},
    "Before/After": {"name": "Before/After", "type": "string"},
    "Altitude": {"name": "Altitude", "type": "float"},
    "Azimuth": {"name": "Azimuth", "type": "float"},
    "Alt Name": {"name": "Alt Name", "type": "string"},
    "Type": {"name": "Type", "type": "string"},
    "Magnitude": {"name": "Magnitude", "type": "float"},
    "Size": {"name": "Size", "type": "float"},
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
    """Load the celestial catalog from CSV into memory."""
    global catalog_table
    try:
        logger.info("Loading catalog...")
        # Handle empty cells and common placeholders during parsing
        catalog_table = Table.read(
            csv_filename,
            format='csv',
            fill_values=[('', ''), ('--', '')]  # Map empty cells and '--' to ''
        )
        # Set fill values based on column type
        for col in catalog_table.colnames:
            if catalog_table[col].dtype.kind in ('f', 'i'):  # Numeric columns
                catalog_table[col].fill_value = None
            else:  # String columns
                catalog_table[col].fill_value = ''
        catalog_table = catalog_table.filled()
        # Log any remaining masked values
        for col in catalog_table.colnames:
            if catalog_table[col].mask is not None and any(catalog_table[col].mask):
                masked_indices = [i for i, m in enumerate(catalog_table[col].mask) if m]
                logger.warning(
                    f"Found {len(masked_indices)} masked values in column '{col}' "
                    f"at rows (0-based): {masked_indices[:5]}"
                )
        logger.info(f"Catalog loaded successfully with {len(catalog_table)} entries.")
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

def calc_transit_altitude(ra_deg, dec_deg, latitude, longitude):
    """
    Calculate the transit altitude of a celestial object based on its declination and observer's latitude.

    The transit altitude is the maximum altitude an object reaches when it crosses the observer's meridian.
    It is given by: 
        Transit Altitude = 90° - |Latitude - Declination|

    Parameters:
    - ra_deg (float): Right Ascension in degrees (not used in altitude calculation).
    - dec_deg (float): Declination of the object in degrees.
    - latitude (float): Observer's latitude in degrees.
    - longitude (float): Observer's longitude in degrees (not used in transit altitude calculation).

    Returns:
    - float: The transit altitude in degrees.
    """
    # Ensure declination and latitude are within valid ranges
    if not (-90 <= dec_deg <= 90):
        raise ValueError("Declination must be between -90 and 90 degrees")
    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")

    # Calculate transit altitude using the absolute difference between latitude and declination
    transit_alt = 90 - abs(latitude - dec_deg)

    # Ensure altitude remains within the valid range of -90° to +90°
    return max(-90, min(90, transit_alt))
    

@app.route('/api/calculate_lst', methods=['POST'])
def calculate_lst():
    try:
        data = request.json
        longitude = float(data['longitude'])
        local_time_str = f"{data['date']} {data['local_time']}"
        timezone = pytz.timezone(data['timezone'])
        local_time = timezone.localize(datetime.strptime(local_time_str, "%Y-%m-%d %H:%M"))
        # Check if the time is in AM and should be considered as the next day
        if local_time.hour < 12:
            local_time = local_time + timedelta(days=1)

        utc_time = local_time.astimezone(pytz.utc)
        lst_hours = Time(utc_time).sidereal_time('mean', longitude * u.deg).hour
        return jsonify({"LST": f"{int(lst_hours):02}:{int((lst_hours*60)%60):02}:{int((lst_hours*3600)%60):02}"})
    except Exception as e:
        logger.error(f"Error calculating LST: {e}")
        return jsonify({"error": "Failed to calculate Local Sidereal Time"}), 500

# Helper functions
def calculate_transit_and_alt_az(ra_deg, dec_deg, latitude, longitude, local_time):
    astropy_time, location, altaz, lst = calc_time_location_and_lst(latitude, longitude, local_time)
    return calc_transit_and_alt_az(ra_deg, dec_deg, local_time, astropy_time, location, altaz, lst)


def calc_time_location_and_lst(latitude, longitude, local_time):
    astropy_time = Time(local_time.astimezone(pytz.utc))
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
    altaz = AltAz(obstime=astropy_time, location=location)
    lst = astropy_time.sidereal_time('mean', longitude * u.deg).hour
    return astropy_time, location, altaz, lst

def calc_transit_and_alt_az(ra_deg, dec_deg, local_time, astropy_time, location, altaz, lst):
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz_coord = target.transform_to(altaz)
    altitude = altaz_coord.alt.deg
    azimuth = altaz_coord.az.deg
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
        pp.oneOf([k.lower() for k in valid_columns if valid_columns[k]['type'] == 'time'], caseless=True) +
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
        parsed_query = expr.parseString(query, parseAll=True)
        return extract_conditions(parsed_query)
    except ParseException as e:
        print(f"Parse Exception: {e}")
        raise ValueError(f"Invalid query syntax: {e}")

def extract_conditions(parsed_expr):
    conditions = []
    for item in parsed_expr:
        if isinstance(item, pp.ParseResults):
            if isinstance(item[0], pp.ParseResults):  # Nested ParseResults
                conditions.append(extract_conditions(item))  # Recursively handle nested conditions
            elif len(item) == 3:  # Simple condition
                column, operator, value = item
                column_info = valid_columns.get(column.lower())
                if not column_info:
                    raise ValueError(f"Invalid column: {column}")
                conditions.append((column_info["name"], operator, value))
        elif isinstance(item, str) and item.lower() in ("and", "or", "*", "+", "&", "|"):
            conditions.append("&" if item.lower() in ("and", "*", "&") else "|")
    return conditions

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
    if not conditions:
        return True

    result = None
    operator = None
    for i, condition in enumerate(conditions):
        if isinstance(condition, list):  # Nested condition
            current = evaluate_conditions(row, condition)
        elif isinstance(condition, tuple):  # Simple condition
            current = evaluate_condition(row, condition)
        else:  # Logical operator
            operator = condition
            continue  # Skip immediate evaluation of operator

        if result is None:
            result = current
        else:
            if operator == '&':
                result = result and current
            elif operator == '|':
                result = result or current
            else:
                raise ValueError(f"Unknown logical operator: {operator}")
            operator = None  # Reset operator for next iteration

    if operator is not None:
        logger.error(f"Unapplied operator found at end of conditions: {operator}")
        raise ValueError("Malformed condition structure: Unapplied operator at the end")

    return result

def evaluate_condition(row, condition):
    column, operator, value = condition
    row_value = row[column]

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

    operator_function = operator_functions.get(operator)
    if operator_function:
        return operator_function(row_value, value)
    else:
        raise ValueError(f"Invalid operator: {operator}")

def test_query_parsing_and_evaluation():
    sample_row = {
        'Altitude': '45.0',
        'Type': 'galaxy',
        'Magnitude': '8.0',
        'Relative TT': '00:15:00'  # Assuming this is how 'relative tt' looks in your data
    }
    valid_columns = {
        'altitude': {'name': 'Altitude', 'type': 'float'},
        'type': {'name': 'Type', 'type': 'string'},
        'magnitude': {'name': 'Magnitude', 'type': 'float'},
        'relative tt': {'name': 'Relative TT', 'type': 'time'}  # Added 'relative tt' with 'time' type
    }

    # Function to convert time string to seconds for comparison
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    # Test cases with query strings and expected outcomes
    test_cases = [
        ("altitude > 55 and relative tt < 00:30:00", False),  # Your problematic query
        ("altitude > 30", True),
        ("altitude > 30 and magnitude < 10", True),
        ("altitude > 30 and (type like galaxy or type like nebula) and magnitude < 10", True),
        ("altitude > 50", False),
        ("altitude > 50 or magnitude < 5", False),
        ("(altitude > 30 and magnitude < 10) or altitude > 50", True),
        ("relative tt < 00:30:00", True),  # Test for relative tt
        ("relative tt > 00:20:00", False),  # Test for relative tt
    ]

    for query, expected in test_cases:
        try:
            # Parse the query string to conditions
            conditions = parse_query_conditions(query, valid_columns)
            
            # Evaluate the conditions
            result = evaluate_conditions(sample_row, conditions)
            
            # Check if the result matches the expected outcome
            if result != expected:
                print(f"Test failed for query '{query}': Expected {expected}, but got {result}")
            else:
                print(f"Test passed for query '{query}'")
        except ValueError as e:
            print(f"Test raised an unexpected error for query '{query}': {e}")

    print("All tests executed.")


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

#        test_query_parsing_and_evaluation()

        start_time = time.perf_counter()  # Record the start time
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        local_time_str = f"{data['date']} {data['local_time']}"
        timezone = pytz.timezone(data['timezone'])
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

        def generate():


            row_count = 0
            included_count = 0

            astropy_time, location, altaz, lst = calc_time_location_and_lst(latitude, longitude, local_time)


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
                transit_time_minutes, local_transit_time, before_after, altitude, azimuth, direction = calc_transit_and_alt_az(
                    ra, dec, local_time, astropy_time, location, altaz, lst)


                if altitude < 0:
                    continue

                transit_alt = calc_transit_altitude(ra, dec, latitude, longitude)

                # Build the row object
                current_row = {
                    'Name': row['Name'],
                    'RA': degrees_to_ra(ra),
                    'Dec': format_dec(dec),
                    'Transit Time': local_transit_time,
                    'Transit Alt': f"{transit_alt:.2f}",
                    'Direction': direction,
                    'Relative TT': format_transit_time(transit_time_minutes),
                    'Before/After': before_after,
                    'Altitude': f"{altitude:.2f}",
                    'Azimuth': f"{azimuth:.2f}",
                    'Alt Name': row.get('Alt Name', ''),
                    'Type': row['Type'],
                    'Magnitude': row['Magnitude'],
                    'Size': row['Size'],
                    'Info': row['Info'],
                    'Catalog': row['Catalog']
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
                current_row['Altitude'] +=  '°'
                current_row['Azimuth'] +=  '°'
                # Stream the matching object as a JSON object
                included_count += 1
#                yield json.dumps(current_row) + "\n"
                              # Try to serialize the row, catch errors
                try:
                    yield json.dumps(current_row) + "\n"
                except TypeError as e:
                    logger.error(
                        f"JSON serialization failed at row {row_count}: {e}\n"
                        f"Row data: {dict(row)}\n"
                        f"Current row: {current_row}"
                    )
                    # Optional: Pause with debugger
                    # import pdb; pdb.set_trace()
                    continue  # Skip this row and continue processing

                
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
    except ValueError as e:
       logger.error(f"Value error: {e}")
       return jsonify({"error": f"Value error: {e}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


@app.route('/api/reload_catalog', methods=['GET', 'POST'])
def reload_catalog():
    """Endpoint to reload the celestial catalog from the CSV file."""
    try:
        load_catalog()  # Reload the catalog from disk
        return jsonify({"status": "success", "message": "Catalog reloaded successfully"}), 200
    except Exception as e:
        logger.error(f"Error reloading catalog: {e}")
        return jsonify({"status": "error", "message": "Failed to reload catalog"}), 500
    
   
