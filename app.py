from flask import Flask, jsonify, render_template_string
import subprocess
import os
import sqlite3

app = Flask(__name__)

# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the polyscraper.py script
SCRAPER_SCRIPT_PATH = os.path.join(BASE_DIR, 'polyscraper.py')
DB_FILE = os.path.join(BASE_DIR, 'polyscraper.db')

# HTML Template for the data viewer page
DATA_VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polyscraper Data</title>
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
    <h1>Latest Polymarket Data</h1>
    <p>Displaying the 50 most recent entries, ordered by timestamp.</p>
    <table>
        <thead>
            <tr>
                <th>Timestamp (UTC)</th>
                <th>Market Name</th>
                <th>Best Bid</th>
                <th>Best Ask</th>
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                <td>{{ row.timestamp }}</td>
                <td>{{ row.market_name }}</td>
                <td>{{ row.best_bid }}</td>
                <td>{{ row.best_ask }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

@app.route('/run-scraper', methods=['POST'])
def run_scraper():
    """
    This endpoint triggers the polyscraper.py script to run once.
    It runs the scraper as a non-blocking background process.
    """
    try:
        # Use subprocess.Popen to run the script in the background
        # This ensures the web server can respond immediately without waiting for the scraper
        # We also need to specify the python executable from our virtual environment.
        python_executable = os.path.join(BASE_DIR, 'venv/bin/python')
        subprocess.Popen([python_executable, SCRAPER_SCRIPT_PATH, '--run-once'])
        
        return jsonify({"status": "success", "message": "Scraper job initiated."}), 202
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/view-data', methods=['GET'])
def view_data():
    """This endpoint displays the last 50 entries from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        # This row_factory allows us to access columns by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch the last 50 records, ordered by timestamp descending
        cursor.execute("SELECT timestamp, market_name, best_bid, best_ask FROM polydata ORDER BY timestamp DESC LIMIT 50")
        data = cursor.fetchall()
        
        conn.close()
        
        return render_template_string(DATA_VIEWER_TEMPLATE, data=data)
    except sqlite3.OperationalError:
        # This handles the case where the database file or table doesn't exist yet
        return "<h1>Data not available yet</h1><p>The database has not been created. Please trigger the scraper at least once.</p>", 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Listen on all available network interfaces, which is necessary for it
    # to be accessible from outside the local machine on AWS.
    app.run(host='0.0.0.0', port=5000) 