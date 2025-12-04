from pathlib import Path

import pandas as pd


file_path = Path('user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather')

if file_path.exists():
    try:
        df = pd.read_feather(file_path)
        print(f"File: {file_path}")
        print("\n--- DataFrame Info ---")
        print(df.info())
        print("\n--- First 5 rows ---")
        print(df.head())
        print("\n--- Last 5 rows ---")
        print(df.tail())
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File not found: {file_path}")
