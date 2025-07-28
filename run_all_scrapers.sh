#!/bin/bash
# This script runs all the Polyscraper scrapers in sequence.

# --- Polymarket Credentials ---
# This ensures the scrapers can place orders when run via cron
export POLYMARKET_PRIVATE_KEY="0xcf71f00a307e65f37e1c7a9111d2822b869b6fa04b78df8833c1e565e37089eb"
export POLYMARKET_PROXY_ADDRESS="0x5423dc7f4beb8c8cf99c1c091ed4717cedc58b45"

# --- Main Script ---
# Navigate to the script's directory
cd "$(dirname "$0")"

# Get the absolute directory of this script to run everything from the correct path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the full path to the python executable in the virtual environment
PYTHON_EXEC="$DIR/venv/bin/python"

# Log the start of the script
echo "--- cron job started at $(date) ---" >> "$DIR/run_all_scrapers.log" 2>&1

# --- Run Scrapers ---
# The output of each script (both standard output and errors) is appended to a log file.
# Updated to use the consolidated polytrader.py with currency arguments
$PYTHON_EXEC "$DIR/polytrader.py" btc --run-once >> "$DIR/btc.log" 2>&1
$PYTHON_EXEC "$DIR/polytrader.py" sol --run-once >> "$DIR/sol.log" 2>&1
$PYTHON_EXEC "$DIR/polytrader.py" xrp --run-once >> "$DIR/xrp.log" 2>&1
$PYTHON_EXEC "$DIR/polytrader.py" eth --run-once >> "$DIR/eth.log" 2>&1

# Log the end of the script
echo "--- cron job finished at $(date) ---" >> "$DIR/run_all_scrapers.log" 2>&1 