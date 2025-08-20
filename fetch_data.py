import yfinance as yf
import pandas as pd
import yaml

def fetch_and_save_data():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ticker = config["data"]["ticker"]
    interval = config["data"]["interval"]
    period = config["data"]["period"]
    output_path = config["data"]["output_path"]

    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    df = df.reset_index()

    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    if 'datetime' in df.columns:
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)
    elif 'date' in df.columns:
        df.rename(columns={'date': 'timestamp'}, inplace=True)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    fetch_and_save_data()
