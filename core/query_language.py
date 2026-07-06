"""
The SQL-like filter query language used by /api/list_objects
(e.g. "altitude > 50 and relative tt < 03 and direction = south").

This module owns parsing a query string into a condition tree and
evaluating that tree against a row of catalog data. No Flask or
astronomy concerns live here.
"""
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

from core.config import logger


def parse_query_conditions(query, valid_columns):
    """Parse a filter query string into a nested list of conditions."""
    column_name = pp.oneOf(list(valid_columns.keys()), caseless=True)
    operator = oneOf("> >= < <= = != like", caseless=True)

    # Value parser for time-like columns (e.g. "21:00" or "00:30:00")
    time_value = Word(alphanums + ":")

    time_condition = Group(
        pp.oneOf(
            [k.lower() for k in valid_columns if valid_columns[k]["type"] == "time"],
            caseless=True,
        )
        + operator
        + time_value
    )

    value = (
        quotedString.setParseAction(removeQuotes)
        | Word(alphanums + ".°")
        | Word(alphas)
    )

    condition = Group(column_name + operator + value)

    and_ = CaselessKeyword("and") | "*" | "&"
    or_ = CaselessKeyword("or") | "|" | "+"

    expr = infixNotation(
        time_condition | condition,
        [
            (and_, 2, opAssoc.LEFT),
            (or_, 2, opAssoc.LEFT),
        ],
    )

    try:
        parsed_query = expr.parseString(query, parseAll=True)
        return _extract_conditions(parsed_query, valid_columns)
    except ParseException as e:
        print(f"Parse Exception: {e}")
        raise ValueError(f"Invalid query syntax: {e}")


def _extract_conditions(parsed_expr, valid_columns):
    conditions = []
    for item in parsed_expr:
        if isinstance(item, pp.ParseResults):
            if isinstance(item[0], pp.ParseResults):  # Nested ParseResults
                conditions.append(_extract_conditions(item, valid_columns))
            elif len(item) == 3:  # Simple condition
                column, operator, value = item
                column_info = valid_columns.get(column.lower())
                if not column_info:
                    raise ValueError(f"Invalid column: {column}")
                conditions.append((column_info["name"], operator, value))
        elif isinstance(item, str) and item.lower() in ("and", "or", "*", "+", "&", "|"):
            conditions.append("&" if item.lower() in ("and", "*", "&") else "|")
    return conditions


# Maps query operators to comparison functions
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
    """Evaluate a (possibly nested) list of conditions against a row dict."""
    if not conditions:
        return True

    result = None
    operator = None
    for condition in conditions:
        if isinstance(condition, list):  # Nested condition
            current = evaluate_conditions(row, condition)
        elif isinstance(condition, tuple):  # Simple condition
            current = evaluate_condition(row, condition)
        else:  # Logical operator
            operator = condition
            continue

        if result is None:
            result = current
        else:
            if operator == "&":
                result = result and current
            elif operator == "|":
                result = result or current
            else:
                raise ValueError(f"Unknown logical operator: {operator}")
            operator = None

    if operator is not None:
        logger.error(f"Unapplied operator found at end of conditions: {operator}")
        raise ValueError("Malformed condition structure: Unapplied operator at the end")

    return result


def evaluate_condition(row, condition):
    """Evaluate a single (column, operator, value) condition against a row dict."""
    column, operator, value = condition
    row_value = row[column]

    def is_numeric(v):
        try:
            float(v)
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
