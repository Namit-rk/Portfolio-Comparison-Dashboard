import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from scr.config import STOCKS, START_DATE
from scr.logger import setup_logger

logger = setup_logger()

RAW_FOLDER = "data/raw"

def format_yfinance_dataset(df:pd.DataFrame):
    """
    Formats the datframe obtained by yfinance into a 
    cleaner more workable version
    """
    df = df.reset_index()
    df.columns = df.columns.droplevel(1)
    df.columns = ['Date','Close','High','Low','Open','Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df


def download_full_history(ticker:str) -> None:
    """
    Downloads the Close, High, Open, Low, Volume raw data of a given ticker, 
    cleans its formatting and makes a csv file for it and adds to data/raw

    Parameter
        ticker: ticker symbol (eg AAPL) for a stock
    Returns
        None
    """
    logger.info(f"Full download for {ticker} from {START_DATE}")

    df = yf.download(ticker, start=START_DATE, progress=False)
    if df.empty:
        logger.error(f"No data returned for {ticker}")
        return
    
    df = df.reset_index()
    df.columns = df.columns.droplevel(1)
    df.columns = ['Date','Close','High','Low','Open','Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df.to_csv(f"{RAW_FOLDER}/{ticker}.csv")
    logger.info(f"{ticker} saved with {len(df)} rows")


def download_incremental(ticker:str):
    """
    Downloads the Close, High, Open, Low, Volume raw data of the given ticker 
    for the day right after the last stored date stored in the csv file of that ticker

    Parameter
        ticker: ticker symbol (eg AAPL) for a stock
    Returns
        None
    """
    path = f"{RAW_FOLDER}/{ticker}.csv"
    existing = pd.read_csv(path, parse_dates=["Date"])

    last_date = existing["Date"].max()
    start_date = last_date + timedelta(days=1) # The next day date

    logger.info(f"Updating {ticker} from {start_date.date()}")

    new_data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), progress=False)
    if new_data.empty:
        logger.info(f"No new data for {ticker}")
        return

    new_data.to_csv(path, mode="a", header=False)
    logger.info(f"{ticker} appended {len(new_data)} new rows")


def update_all_tickers():
    """
    Function to obtain the raw data of each individual stock mentioned in config file
    if we do not have the total data else adds data of next day to csv file
    """
    os.makedirs(RAW_FOLDER, exist_ok=True)
    for ticker in STOCKS:
        try:
            path = f"{RAW_FOLDER}/{ticker}.csv"
            if not os.path.exists(path):
                download_full_history(ticker)
            else:
                download_incremental(ticker)
        except Exception as e:
            logger.error(f"{ticker} failed: {e}")
            

if __name__ == "__main__":
    update_all_tickers()
