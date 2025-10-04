# Polyscraper - Polymarket Continuous Trading Bot

An automated cryptocurrency prediction market trading system that runs continuously on AWS EC2, analyzing crypto price movements and executing trades on Polymarket.

##  **How It Works**

The bot continuously cycles through Bitcoin, Ethereum, Solana, and XRP markets:

1. **Price Analysis**: Fetches real-time crypto prices from Binance
2. **Market Data**: Gets current Polymarket odds using the price API  
3. **ML Predictions**: Uses LightGBM models trained on price movements and order flow
4. **Dynamic Trading**: Adjusts positions based on prediction confidence and available bankroll
5. **Risk Management**: Implements minimum order sizes, spread checks, and position limits

##  **Key Features**

- **Continuous Operation**: Runs 24/7 with ~2-second cycle times
- **Dynamic Position Sizing**: Trades proportional to prediction confidence
- **Proper Price Tracking**: Uses exact hour-start prices for accurate ratios
- **Parallel API Calls**: Optimized for speed with concurrent bid/ask fetching
- **Rate Limiting**: Respects API limits for both Binance and Polymarket
- **Comprehensive Logging**: Detailed timing and position tracking

##  **Trading Logic**

- **Entry Threshold**: ±3 percentage points delta between prediction and market
- **Position Sizing**: `bankroll_fraction × bankroll × |delta|`  
- **Risk Limits**: Maximum 80% of bankroll exposure
- **Order Management**: Cancels previous orders each cycle for maximum dynamism

##  **Installation**

1. **Clone and Setup**:
   ```bash
   git clone <repo>
   cd Polyscraper
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Configure Credentials**:
   ```bash
   export POLYMARKET_PRIVATE_KEY="your_private_key"
   export POLYMARKET_PROXY_ADDRESS="your_proxy_address"
   ```

3. **Run Locally**:
   ```bash
   python3 polytrader_continuous.py
   ```

##  **Deployment (AWS EC2)**

1. **Setup EC2 Instance** with the repository
2. **Install Dependencies** and configure environment
3. **Deploy as System Service**:
   ```bash
   sudo cp polytrader.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable polytrader
   sudo systemctl start polytrader
   ```

4. **Monitor Status**:
   ```bash
   sudo systemctl status polytrader
   sudo journalctl -u polytrader -f
   ```

##  **Project Structure**

```
Polyscraper/
├── polytrader_continuous.py    #  Main trading bot
├── polytrader.service          #  Systemd service config  
├── app.py                      #  Flask web dashboard
├── *_lgbm.txt                  #  Pre-trained ML models
├── requirements.txt            #  Python dependencies
└── README.md                   #  This file
```

##  **Performance**

- **Cycle Time**: ~2 seconds per full cycle (4 currencies)
- **API Efficiency**: Parallel bid/ask calls, intelligent caching
- **Uptime**: Designed for 24/7 operation with automatic restarts
- **Accuracy**: Uses exact Binance hour-start prices for ML features

##  **Risk Disclosure**

This is experimental trading software. Only use with funds you can afford to lose. Past performance does not guarantee future results. Review all code and test thoroughly before live deployment.

##  **Development**

- **Languages**: Python 3.8+
- **ML Framework**: LightGBM
- **APIs**: Binance (price data), Polymarket (trading)
- **Database**: SQLite for data storage
- **Deployment**: AWS EC2 with systemd

////// 

PUSH:

git add .
git commit -m "ADD"
git push origin main

START BOT:

sudo systemctl stop polytrader

sudo cp polytrader.service /etc/systemd/system/

sudo systemctl daemon-reload

sudo systemctl start polytrader

sudo journalctl -f -u polytrader

LOGIN

chmod 400 /Users/yannickofungi/Downloads/aws-github-runner.pem 
ssh -i /Users/yannickofungi/Downloads/aws-github-runner.pem  ubuntu@3.71.4.27

cd ~/Polyscraper
source venv/bin/activate


DOWNLOAD DATA (On Local Machine)

scp -i /Users/yannickofungi/Downloads/aws-github-runner.pem \
  "ubuntu@3.71.4.27:/home/ubuntu/Polyscraper/*polyscraper.db*" \
  ~/Downloads/
