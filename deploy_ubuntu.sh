#!/bin/bash
# Quant MVP - Ubuntu Deploy Script
# Run this on your fresh Ubuntu 22.04+ Cloud Server

echo "🚀 Starting Quant MVP Ubuntu Deployment Setup..."

# 1. System Updates & Python
echo "📦 Installing system dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl cron screen tzdata

# Set Timezone to EST (New York) to match market hours
sudo timedatectl set-timezone America/New_York

# 2. Setup Python Environment
echo "🐍 Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    # Explicitly add ML models if not captured
    pip install lightgbm catboost
fi

# 3. FutuOpenD Setup Instructions
echo "=========================================================="
echo "⚠️  CRITICAL STEP: FUTU OPEND GATEWAY"
echo "=========================================================="
echo "You must download FutuOpenD for Ubuntu from Futu's website."
echo "1. wget https://softwarefile.futunn.com/FutuOpenD_X.Y.Z_Ubuntu16.04.tar.gz"
echo "2. tar -xvf FutuOpenD*.tar.gz"
echo "3. cd FutuOpenD"
echo "4. ./FutuOpenD -login_account=PHONE -login_pwd_md5=MD5_PWD"
echo ""
echo "When you run it the first time, Futu will detect a new IP"
echo "and output a CAPTCHA link in the console. Copy that link,"
echo "open it in your local browser, solve it, and OpenD will log in."
echo ""
echo "After successful login, start OpenD in the background:"
echo "nohup ./FutuOpenD > opend.log 2>&1 &"
echo "=========================================================="

# 4. Cronjob Example
echo "⏰ Cron Job Setup Example"
echo "Once FutuOpenD is running and .env is configured, add this to 'crontab -e':"
echo "30 16 * * 1-5 cd $(pwd) && $(pwd)/venv/bin/python $(pwd)/src/ops/daily_job.py >> $(pwd)/logs/cron.log 2>&1"
echo ""
echo "✅ Environment setup complete! Activate with: source venv/bin/activate"
