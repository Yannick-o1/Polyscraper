#!/bin/bash
# Continuous Trading Script - Runs BTC->ETH->SOL->XRP in endless loop
# This replaces the old 2-minute cron job for maximum trading frequency

# --- Polymarket Credentials ---
export POLYMARKET_PRIVATE_KEY="0xcf71f00a307e65f37e1c7a9111d2822b869b6fa04b78df8833c1e565e37089eb"
export POLYMARKET_PROXY_ADDRESS="0x5423dc7f4beb8c8cf99c1c091ed4717cedc58b45"

# --- Script Setup ---
cd "$(dirname "$0")"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_EXEC="$DIR/venv/bin/python"

# --- Logging ---
LOG_FILE="$DIR/continuous_trading.log"
echo "=== Continuous Trading Started at $(date) ===" >> "$LOG_FILE"

# --- Main Loop ---
echo "🚀 Starting Continuous Trading System..."
echo "📈 Trading: BTC -> ETH -> SOL -> XRP -> BTC -> ..."
echo "📝 Logs: $LOG_FILE"
echo "🛑 Stop with: Ctrl+C"
echo ""

# Run the continuous trading script
$PYTHON_EXEC "$DIR/polytrader_continuous.py" 2>&1 | tee -a "$LOG_FILE"

echo "=== Continuous Trading Stopped at $(date) ===" >> "$LOG_FILE" 