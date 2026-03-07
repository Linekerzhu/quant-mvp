import os
import sys
from dotenv import load_dotenv

# Ensure the root of the project is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

import pandas as pd
from src.models.expert_oracle import KronosOracleClient

def test_oracle():
    url = os.environ.get("KRONOS_ENDPOINT_URL")
    if not url:
        print("ERROR: KRONOS_ENDPOINT_URL not found in .env")
        return
        
    print(f"Testing Oracle Endpoint: {url}")
    client = KronosOracleClient(url)
    
    # Create fake OHLCV data for testing
    dates = pd.date_range("2024-01-01", periods=10)
    df = pd.DataFrame({
        'symbol': ['AAPL'] * 10,
        'date': dates,
        'open': [150 + i for i in range(10)],
        'high': [155 + i for i in range(10)],
        'low': [148 + i for i in range(10)],
        'close': [152 + i for i in range(10)],
        'volume': [1000000 + (i*1000) for i in range(10)]
    })
    
    print("\nSending request to Modal Oracle...")
    result = client.request_veto("AAPL", 0.05, df, lookback=10)
    print(set_color("Success!", "green"))
    print("Response payload:")
    print(result)

def set_color(text, color):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "end": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"

if __name__ == "__main__":
    test_oracle()
