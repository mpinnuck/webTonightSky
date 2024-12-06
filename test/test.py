import logging
from pyparsing import ParseException

import pyparsing as pp
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


# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Define your test cases
test_cases = [
    {
        "query": "type like Galaxy",
        "valid_columns": valid_columns,
        "row": {"Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 AND type = 'Galaxy'",
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.8", "Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "altitude > 40° OR (magnitude < 8 and type = 'Nebula')",
        "valid_columns": valid_columns,
        "row": {"Altitude": "35°", "Magnitude": "7.5", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "name like 'M' or name like 'NGC'",
        "valid_columns": valid_columns,
        "row": {"Name": "NGC 224"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 & (type = 'Galaxy' | type = 'Nebula')",
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "magnitude >= 6 * type = 'Galaxy'",  # Using '*' for AND
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.8", "Type": "Galaxy"},
        "expected_result": False,  # Should fail
    },
    {
        "query": "altitude > 40° + (magnitude >= 8 & type = 'Nebula')",  # Using '+' for AND
        "valid_columns": valid_columns,
        "row": {"Altitude": "35°", "Magnitude": "7.5", "Type": "Nebula"},
        "expected_result": False,  # Should fail
    },
    {
        "query": "name like 'IC'",
        "valid_columns": valid_columns,
        "row": {"Name": "NGC 224"},
        "expected_result": False,  # Should fail
    },
    # Add more test cases here...
    {
        "query": "magnitude < 6 and (type = 'Galaxy' or type = 'Nebula')",
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 & (type = 'Galaxy' | type = 'Nebula')",
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 + (type = 'Galaxy' + type = 'Nebula')",
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    # empty scenarios
    {
        "query": "info = ''",  # Using '=' with an empty string
        "valid_columns": valid_columns,
        "row": {"Info": ""},  # Empty string
        "expected_result": True,  # Should pass because "" is considered empty
    },
    {
        "query": "magnitude != ''",  # Using '!=' with an empty string
        "valid_columns": valid_columns,
        "row": {"Magnitude": "5.2"},  # Non-empty string
        "expected_result": True,  # Should pass because the value is not an empty string
    },
    # relatuve TT tests
    {
        "query": "relative tt < 01",
        "valid_columns": valid_columns,
        "row": {"Relative TT": "00:30:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt >= 02",
        "valid_columns": valid_columns,
        "row": {"Relative TT": "03:00:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 00:30",
        "valid_columns": valid_columns,
        "row": {"Relative TT": "00:20:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 01 and type like 'Galaxy'",
        "valid_columns": valid_columns,
        "row": {"Relative TT": "00:45:00", "Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "relative tt > 02 or magnitude < 6",
        "valid_columns": valid_columns,
        "row": {"Relative TT": "01:30:00", "Magnitude": "5.8"},
        "expected_result": True,
    },

]

# Run the tests
passed_count = 0
failed_count = 0
for i, case in enumerate(test_cases):
    try:
        # --- Parse the query ---
        conditions = parse_query_conditions(case["query"], case["valid_columns"])
        logging.debug(f"Test case {i+1}: Parsed conditions: {conditions}")

        # --- Evaluate the conditions ---
        # Access row values using the correct case from valid_columns
        corrected_row = {case["valid_columns"][k.lower()]['name']: v for k, v in case["row"].items()}
        result = evaluate_conditions(corrected_row, conditions)
        logging.debug(f"Test case {i+1}: Result: {result}")

        # --- Assert the result ---
        assert result == case["expected_result"], f"Test case {i+1} failed! Expected {case['expected_result']}, got {result}, for query: {case['query']}, row: {corrected_row}, conditions: {conditions}"
        print(f"Test case {i+1}: Passed")
        passed_count += 1  # Increment passed count

    except ParseException as e:
        logging.error(f"Test case {i+1}: Parsing error - {e}")
        failed_count += 1  # Increment failed count
    except ValueError as e:
        logging.error(f"Test case {i+1}: Evaluation error - {e}")
        failed_count += 1  # Increment failed count
    except AssertionError as e:
        logging.error(e)
        failed_count += 1  # Increment failed count

# Print summary
total_count = passed_count + failed_count
print("\n--- Summary ---")
print(f"Total test cases: {total_count}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")