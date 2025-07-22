from flask import Flask, jsonify, render_template_string, url_for
import subprocess
import os
import sqlite3

app = Flask(__name__)

# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom filter for formatting with dynamic decimal places
@app.template_filter('format_price')
def format_price(value, decimals):
    if value is None:
        return 'N/A'
    return f"{value:.{decimals}f}"

# --- Configuration for multiple currencies ---
CURRENCIES = {
    'btc': {
        'name': 'Bitcoin',
        'scraper_script': 'polyscraper.py',
        'db_file': 'polyscraper.db',
        'spot_decimals': 2
    },
    'sol': {
        'name': 'Solana',
        'scraper_script': 'sol_polyscraper.py',
        'db_file': 'sol_polyscraper.db',
        'spot_decimals': 3
    },
    'xrp': {
        'name': 'XRP',
        'scraper_script': 'xrp_polyscraper.py',
        'db_file': 'xrp_polyscraper.db',
        'spot_decimals': 4
    },
    'eth': {
        'name': 'Ethereum',
        'scraper_script': 'eth_polyscraper.py',
        'db_file': 'eth_polyscraper.db',
        'spot_decimals': 3
    }
}

# --- Templates ---
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Polyscraper Dashboard</title>
</head>
<body>
    <h1>Select a currency to view its data:</h1>
    <ul>
        {% for currency_code, currency_data in currencies.items() %}
            <li><a href="{{ url_for('view_data', currency=currency_code) }}">{{ currency_data.name }}</a></li>
        {% endfor %}
    </ul>
</body>
</html>
"""

DATA_VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ currency_name }} Data</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; background-color: #f4f4f4; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 1em; }
        th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #e9e9e9; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <a href="{{ url_for('home') }}">Back to Home</a>
    <h1>Latest {{ currency_name }} Data</h1>
    <p>Displaying the 100 most recent entries, ordered by timestamp.</p>
    <table>
        <thead>
            <tr>
                <th>Timestamp (UTC)</th>
                <th>Market Name</th>
                <th>Outcome</th>
                <th>Spot Price (USDT)</th>
                <th>OFI</th>
                <th>P(Up) Prediction</th>
                <th>Best Bid</th>
                <th>Best Ask</th>
                <th>Delta</th>
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                <td>{{ row.timestamp }}</td>
                <td>{{ row.market_name }}</td>
                <td>{{ row.outcome if row.outcome else 'N/A' }}</td>
                <td>{{ row.spot_price|format_price(spot_decimals) }}</td>
                <td>{{ "%.4f"|format(row.ofi) if row.ofi is not none else 'N/A' }}</td>
                <td>{{ "%.2f"|format(row.p_up_prediction * 100) if row.p_up_prediction is not none else 'N/A' }}</td>
                <td>{{ "%.0f"|format(row.best_bid * 100) if row.best_bid is not none else 'N/A' }}</td>
                <td>{{ "%.0f"|format(row.best_ask * 100) if row.best_ask is not none else 'N/A' }}</td>
                <td>{{ "%.2f"|format((row.p_up_prediction - (row.best_bid + row.best_ask) / 2) * 100) if row.p_up_prediction is not none and row.best_bid is not none and row.best_ask is not none else 'N/A' }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_TEMPLATE, currencies=CURRENCIES)

@app.route('/run-scraper/<currency>', methods=['POST'])
def run_scraper(currency):
    """
    Triggers a specific currency's scraper script to run once.
    """
    if currency not in CURRENCIES:
        return jsonify({"status": "error", "message": "Invalid currency."}), 404

    try:
        config = CURRENCIES[currency]
        script_path = os.path.join(BASE_DIR, config['scraper_script'])
        python_executable = os.path.join(BASE_DIR, 'venv/bin/python')
        
        subprocess.Popen([python_executable, script_path, '--run-once'])
        return jsonify({"status": "success", "message": f"{config['name']} scraper job initiated."}), 202
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/view/<currency>', methods=['GET'])
def view_data(currency):
    """Displays the last 100 entries from the database for a specific currency."""
    if currency not in CURRENCIES:
        return "<h1>Invalid Currency</h1>", 404

    config = CURRENCIES[currency]
    db_path = os.path.join(BASE_DIR, config['db_file'])
    spot_price_column = f"{currency}_usdt_spot"

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if the table and column exist to provide better error messages
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='polydata';")
        if cursor.fetchone() is None:
            return f"<h1>Data not available yet</h1><p>The table 'polydata' does not exist in {config['db_file']}. Please run the scraper for {config['name']}.</p>", 404
        
        query = f"SELECT timestamp, market_name, best_bid, best_ask, {spot_price_column} as spot_price, ofi, p_up_prediction, outcome FROM polydata ORDER BY timestamp DESC LIMIT 100"
        cursor.execute(query)
        data = cursor.fetchall()
        
        conn.close()
        
        return render_template_string(DATA_VIEWER_TEMPLATE, data=data, currency_name=config['name'], spot_decimals=config['spot_decimals'])
    except sqlite3.OperationalError as e:
        if "no such column" in str(e):
             return f"<h1>Data not available yet</h1><p>The column '{spot_price_column}' does not exist yet. Please run the scraper for {config['name']}.</p>", 404
        return f"<h1>Data not available yet</h1><p>The database for {config['name']} has not been created or is empty. Please trigger the scraper at least once.</p>", 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 