from flask import Flask, jsonify
import subprocess
import os

app = Flask(__name__)

# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the polyscraper.py script
SCRAPER_SCRIPT_PATH = os.path.join(BASE_DIR, 'polyscraper.py')

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

if __name__ == '__main__':
    # Listen on all available network interfaces, which is necessary for it
    # to be accessible from outside the local machine on AWS.
    app.run(host='0.0.0.0', port=5000) 