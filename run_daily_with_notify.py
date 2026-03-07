#!/usr/bin/env python3
"""
Daily Job Runner with Telegram Notification

Runs daily_job.py and sends result notification to Telegram with trading signals.
"""

import os
import sys
import subprocess
import requests
import pandas as pd
from datetime import datetime

# Determine project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load environment
def load_env():
    env = {}
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    env[key] = val
    return env

def send_telegram(env, message):
    token = env.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = env.get('TELEGRAM_CHAT_ID', '')
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, data=data, timeout=10)
            return resp.ok
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    return False

def get_trading_signals(trade_date):
    """Parse signals from the daily job output."""
    signals_path = os.path.join(PROJECT_ROOT, f'data/processed/signals_{trade_date}.parquet')
    
    if not os.path.exists(signals_path):
        return None, "No signals file found"
    
    try:
        df = pd.read_parquet(signals_path)
        if df.empty or len(df) == 0:
            return None, "No active signals today"
        
        # Filter for actionable signals
        buys = df[df['side'] > 0][['symbol', 'prob']].sort_values('prob', ascending=False)
        sells = df[df['side'] < 0][['symbol', 'prob']].sort_values('prob', ascending=False)
        
        return {
            'buys': buys.head(10).to_dict('records') if len(buys) > 0 else [],
            'sells': sells.head(10).to_dict('records') if len(sells) > 0 else [],
            'total': len(df)
        }, None
    except Exception as e:
        return None, str(e)

def get_universe_count(trade_date):
    """Get the stock universe count."""
    data_path = os.path.join(PROJECT_ROOT, f'data/processed/final_daily_{trade_date}.parquet')
    if os.path.exists(data_path):
        try:
            df = pd.read_parquet(data_path)
            return df['symbol'].nunique()
        except:
            pass
    return None

def main():
    env = load_env()
    start_time = datetime.now()
    
    # Get today's date
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    # Run daily_job.py
    result = subprocess.run(
        ['python3', 'src/ops/daily_job.py'],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env, 'PYTHONPATH': PROJECT_ROOT}
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Build message
    status = "✅ 成功" if result.returncode == 0 else "❌ 失败"
    
    message = f"""📊 *Quant MVP Daily Report*

🕐 *日期*: {trade_date}
📈 *状态*: {status}
⏱️ *耗时*: {elapsed:.1f}秒

"""
    
    if result.returncode != 0:
        message += f"*错误*:\n```\n{result.stderr[-500:]}\n```"
    else:
        # Get trading signals
        signals, err = get_trading_signals(trade_date)
        universe = get_universe_count(trade_date)
        
        if universe:
            message += f"*关注股票数*: {universe} 只\n"
        
        if signals:
            message += f"\n*信号总数*: {signals['total']} 个\n"
            
            if signals['buys']:
                message += f"\n🟢 *买入信号* ({len(signals['buys'])} 个):\n"
                for s in signals['buys'][:5]:
                    prob = s.get('prob', 0)
                    prob_str = f"{prob*100:.1f}%" if prob else "N/A"
                    message += f"  • {s['symbol']} (置信度: {prob_str})\n"
                if len(signals['buys']) > 5:
                    message += f"  ... 还有 {len(signals['buys'])-5} 个\n"
            
            if signals['sells']:
                message += f"\n🔴 *卖出信号* ({len(signals['sells'])} 个):\n"
                for s in signals['sells'][:5]:
                    prob = s.get('prob', 0)
                    prob_str = f"{prob*100:.1f}%" if prob else "N/A"
                    message += f"  • {s['symbol']} (置信度: {prob_str})\n"
                if len(signals['sells']) > 5:
                    message += f"  ... 还有 {len(signals['sells'])-5} 个\n"
            
            if not signals['buys'] and not signals['sells']:
                message += "\n⚪ *今日无交易信号* — 市场未满足买入/卖出条件"
        elif err:
            message += f"\n⚠️ *信号解析*: {err}"
    
    # Send message
    send_telegram(env, message)
    
    # Also print to stdout
    print(message)
    
    # Exit with same code
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
