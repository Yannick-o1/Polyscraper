#!/bin/bash
# Fast EC2 Sync Script - Updates only changed files

# Configuration
EC2_IP="3.71.4.27"  # Replace with your actual EC2 IP
EC2_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/Polyscraper"

# Files to sync (only the important ones)
FILES_TO_SYNC=(
    "polytrader_continuous.py"
    "polytrader.py"
    "app.py"
    "deploy_live.sh"
    "run_continuous_trading.sh"
    "polytrader.service"
    "check_status.py"
    "requirements.txt"
)

echo "🚀 Fast EC2 Sync Starting..."
echo "📁 Syncing ${#FILES_TO_SYNC[@]} files to $EC2_USER@$EC2_IP:$REMOTE_DIR"

# Sync each file
for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "$file" ]; then
        echo "📤 Uploading $file..."
        scp "$file" "$EC2_USER@$EC2_IP:$REMOTE_DIR/"
    else
        echo "⚠️ File $file not found, skipping..."
    fi
done

echo "✅ Sync complete!"
echo ""
echo "🔄 Now restart the service on EC2:"
echo "ssh $EC2_USER@$EC2_IP"
echo "sudo systemctl restart polytrader"
echo "sudo systemctl status polytrader" 