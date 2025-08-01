#!/bin/bash
# Live Trading Deployment Script for EC2
# This script sets up the continuous trading bot for production

echo "🚀 Deploying Live Continuous Trading Bot..."

# --- Environment Setup ---
export POLYMARKET_PRIVATE_KEY="0xcf71f00a307e65f37e1c7a9111d2822b869b6fa04b78df8833c1e565e37089eb"
export POLYMARKET_PROXY_ADDRESS="0x5423dc7f4beb8c8cf99c1c091ed4717cedc58b45"

# --- Script Setup ---
cd "$(dirname "$0")"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_EXEC="$DIR/venv/bin/python"

# --- Logging ---
LOG_FILE="$DIR/live_trading.log"
echo "=== Live Trading Started at $(date) ===" >> "$LOG_FILE"

# --- Main Loop ---
echo "🔥 Starting LIVE Continuous Trading System..."
echo "📈 Trading: BTC -> ETH -> SOL -> XRP -> BTC -> ..."
echo "💰 REAL MONEY TRADING ENABLED"
echo "📝 Logs: $LOG_FILE"
echo "🛑 Stop with: Ctrl+C"
echo ""

# Run the continuous trading script
$PYTHON_EXEC "$DIR/polytrader_continuous.py" 2>&1 | tee -a "$LOG_FILE"

echo "=== Live Trading Stopped at $(date) ===" >> "$LOG_FILE" 