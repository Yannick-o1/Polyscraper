#!/bin/bash
# One-Command EC2 Update - Uploads and restarts automatically

# Configuration (UPDATE THESE)
EC2_IP="3.71.4.27"  # Replace with your actual EC2 IP
EC2_USER="ubuntu"

echo "🚀 Quick EC2 Update Starting..."

# Upload the main files
echo "📤 Uploading updated files..."
scp polytrader_continuous.py "$EC2_USER@$EC2_IP:/home/ubuntu/Polyscraper/"
scp polytrader.service "$EC2_USER@$EC2_IP:/home/ubuntu/Polyscraper/"

# Restart the service remotely
echo "🔄 Restarting service on EC2..."
ssh "$EC2_USER@$EC2_IP" << 'EOF'
cd /home/ubuntu/Polyscraper
sudo cp polytrader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart polytrader
sudo systemctl status polytrader --no-pager
echo "✅ Service restarted!"
EOF

echo "🎉 Update complete! Check the status above." 