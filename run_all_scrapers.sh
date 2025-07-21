#!/bin/bash
# This script runs all the Polyscraper scrapers in sequence.

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the absolute directory of this script to run everything from the correct path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the full path to the python executable in the virtual environment
PYTHON_EXEC="$DIR/venv/bin/python"

# --- Run Scrapers ---
# The output of each script (both standard output and errors) is appended to a log file.
echo "--- Running all scrapers at $(date) ---"

$PYTHON_EXEC "$DIR/polyscraper.py" --run-once >> "$DIR/btc.log" 2>&1
$PYTHON_EXEC "$DIR/sol_polyscraper.py" --run-once >> "$DIR/sol.log" 2>&1
$PYTHON_EXEC "$DIR/xrp_polyscraper.py" --run-once >> "$DIR/xrp.log" 2>&1
$PYTHON_EXEC "$DIR/eth_polyscraper.py" --run-once >> "$DIR/eth.log" 2>&1

echo "--- All scrapers finished at $(date) ---" 