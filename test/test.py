"""
Tests for the filter query language (parse_query_conditions / evaluate_conditions).

This previously duplicated the parser/evaluator logic inline, which let it
drift out of sync with the real implementation in query_language.py. It now
imports the actual functions so a change to the grammar can't silently go
untested.
"""
import logging
import sys
import os

# Allow running as `python test/test.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import valid_columns
from core.query_language import parse_query_conditions, evaluate_conditions
from pyparsing import ParseException

logging.basicConfig(level=logging.DEBUG)

test_cases = [
    {
        "query": "type like Galaxy",
        "row": {"Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 AND type = 'Galaxy'",
        "row": {"Magnitude": "5.8", "Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "altitude > 40° OR (magnitude < 8 and type = 'Nebula')",
        "row": {"Altitude": "35°", "Magnitude": "7.5", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "name like 'M' or name like 'NGC'",
        "row": {"Name": "NGC 224"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 & (type = 'Galaxy' | type = 'Nebula')",
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "magnitude >= 6 * type = 'Galaxy'",  # '*' as AND
        "row": {"Magnitude": "5.8", "Type": "Galaxy"},
        "expected_result": False,
    },
    {
        "query": "altitude > 40° + (magnitude >= 8 & type = 'Nebula')",  # '+' as OR
        "row": {"Altitude": "35°", "Magnitude": "7.5", "Type": "Nebula"},
        "expected_result": False,
    },
    {
        "query": "name like 'IC'",
        "row": {"Name": "NGC 224"},
        "expected_result": False,
    },
    {
        "query": "magnitude < 6 and (type = 'Galaxy' or type = 'Nebula')",
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "magnitude < 6 + (type = 'Galaxy' + type = 'Nebula')",
        "row": {"Magnitude": "5.2", "Type": "Nebula"},
        "expected_result": True,
    },
    {
        "query": "info = ''",
        "row": {"Info": ""},
        "expected_result": True,
    },
    {
        "query": "magnitude != ''",
        "row": {"Magnitude": "5.2"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 01:00:00",
        "row": {"Relative TT": "00:30:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt >= 02:00:00",
        "row": {"Relative TT": "03:00:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 00:30:00",
        "row": {"Relative TT": "00:20:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 01:00:00 and type like 'Galaxy'",
        "row": {"Relative TT": "00:45:00", "Type": "Galaxy"},
        "expected_result": True,
    },
    {
        "query": "relative tt > 02:00:00 or magnitude < 6",
        "row": {"Relative TT": "01:30:00", "Magnitude": "5.8"},
        "expected_result": True,
    },
    # Test cases previously embedded directly in wsTonightSky.py
    {
        "query": "altitude > 30",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": True,
    },
    {
        "query": "altitude > 30 and magnitude < 10",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": True,
    },
    {
        "query": "altitude > 30 and (type like galaxy or type like nebula) and magnitude < 10",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": True,
    },
    {
        "query": "altitude > 50",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": False,
    },
    {
        "query": "altitude > 50 or magnitude < 5",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": False,
    },
    {
        "query": "(altitude > 30 and magnitude < 10) or altitude > 50",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt < 00:30:00",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": True,
    },
    {
        "query": "relative tt > 00:20:00",
        "row": {"Altitude": "45.0", "Type": "galaxy", "Magnitude": "8.0", "Relative TT": "00:15:00"},
        "expected_result": False,
    },
]


def run():
    passed_count = 0
    failed_count = 0

    for i, case in enumerate(test_cases):
        try:
            conditions = parse_query_conditions(case["query"], valid_columns)
            logging.debug(f"Test case {i+1}: Parsed conditions: {conditions}")

            # Row keys must use the canonical (proper-case) column names,
            # same as the real column names used by parse_query_conditions.
            corrected_row = {
                valid_columns[k.lower()]["name"]: v
                for k, v in case["row"].items()
                if k.lower() in valid_columns
            }
            result = evaluate_conditions(corrected_row, conditions)
            logging.debug(f"Test case {i+1}: Result: {result}")

            assert result == case["expected_result"], (
                f"Test case {i+1} failed! Expected {case['expected_result']}, got {result}, "
                f"for query: {case['query']}, row: {corrected_row}, conditions: {conditions}"
            )
            print(f"Test case {i+1}: Passed")
            passed_count += 1

        except ParseException as e:
            logging.error(f"Test case {i+1}: Parsing error - {e}")
            failed_count += 1
        except ValueError as e:
            logging.error(f"Test case {i+1}: Evaluation error - {e}")
            failed_count += 1
        except AssertionError as e:
            logging.error(e)
            failed_count += 1

    total_count = passed_count + failed_count
    print("\n--- Summary ---")
    print(f"Total test cases: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    return failed_count == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
