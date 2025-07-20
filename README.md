# BTC Polymarket Scraper

Automated scraper for Bitcoin hourly markets on Polymarket.

## Features

- **Automated Data Collection**: Scrapes current market data every minute
- **Market List Updates**: Refreshes the list of available markets every other day
- **GitHub Actions Integration**: Runs automatically on GitHub's servers
- **Data Filtering**: Only includes markets from July 18, 2025 onwards
- **Data Validation**: Rejects markets with missing or invalid data

## Files

- `polyscraper.py` - Main scraper script
- `btc_polydata.csv` - Live market data (updated every minute)
- `btc_polymarkets.csv` - Available markets list (updated every other day)
- `.github/workflows/polymarket-scraper.yml` - GitHub Actions workflow
- `requirements.txt` - Python dependencies

## Setup

1. **Push to GitHub**: Push this repository to your GitHub account

2. **Enable Actions**: Go to your GitHub repository > Actions tab and enable GitHub Actions if prompted

3. **Manual Test**: Test the workflow manually:
   - Go to Actions tab
   - Click "Polymarket Bitcoin Scraper"
   - Click "Run workflow"

## How it Works

### Every Minute
- Downloads the current order book for the active Bitcoin hourly market
- Extracts best bid and ask prices
- Appends new data to `btc_polydata.csv`

### Every Other Day (2 AM UTC)
- Fetches all available markets from Polymarket
- Filters for Bitcoin hourly markets
- Updates `btc_polymarkets.csv` with the latest market list

## Local Usage

```bash
# Run data scraping
python3 polyscraper.py

# Update markets list only
python3 polyscraper.py --update-markets
```

## Data Format

### btc_polydata.csv
```
timestamp,market_name,token_id,best_bid,best_ask
2025-01-18 01:37:24,Will Bitcoin be up or down on January 18 at 5 PM ET?,123456,0.45,0.55
```

### btc_polymarkets.csv
```
market_name,token_id,date_time,market_slug
Will Bitcoin be up or down on January 18 at 5 PM ET?,123456,2025-01-18 17:00 EDT,bitcoin-up-or-down-january-18-5-pm-et
```
