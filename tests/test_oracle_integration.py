import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.ops.daily_job import DailyJob
from src.data.wap_utils import read_parquet_safe

def main():
    date = "2026-03-07"
    print(f"\n--- Testing Oracle Veto Integration for {date} ---")
    
    # Ensure targets parquet doesn't exist so we can see the fresh output
    target_path = f"data/processed/targets_{date}.parquet"
    if os.path.exists(target_path):
        os.remove(target_path)
        
    job = DailyJob()
    
    # We just run the risk and size step because signals and features already exist locally!
    print("Running _step_risk_and_size ... (This will call the Modal Oracle for each ticker)")
    result = job._step_risk_and_size(date)
    
    print("\nResult:")
    print(result)
    
    if os.path.exists(target_path):
        targets = read_parquet_safe(target_path)
        print("\nFinal Targets Post-Oracle Veto:")
        print(targets[['symbol', 'target_weight']])
    else:
        print("\nNo targets generated.")

if __name__ == "__main__":
    main()
