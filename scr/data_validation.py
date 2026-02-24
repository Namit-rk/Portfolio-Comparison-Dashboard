import os
import pandas as pd
from datetime import timedelta
from scr.config import STOCKS
from scr.logger import setup_logger

logger = setup_logger()

RAW_FOLDER = "data/raw"

def load_ticker_file(ticker:str)->None | pd.DataFrame:
    """
    Loads the csv file with data of given ticker, returns it as 
    datframe if availabe else returns None
    """
    path = f"{RAW_FOLDER}/{ticker}.csv"
    if not os.path.exists(path):
        logger.error(f"{ticker} file missing")
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"{ticker} could not be read: {e}")
        return None

def check_duplicates(ticker:str, df:pd.DataFrame)->None:
    """
    Logs the number of duplicate entries if present
    """
    dup = df.duplicated().sum()
    if dup > 0:
        logger.warning(f"{ticker} has {dup} duplicate rows")


def check_missing_values(ticker:str, df:pd.DataFrame)->None:
    """
    Logs the number of Nan entries if present
    """
    if df.isnull().sum().sum() > 0:
        logger.warning(f"{ticker} contains NaN values")


def check_abnormal_returns(ticker:str, df:pd.DataFrame)->None:
    """
    Logs the number of abnormal returns (>20%)
    """
    df["returns"] = df["Close"].pct_change()
    abnormal = df[df["returns"].abs() > 0.20]
    if len(abnormal) > 0:
        logger.warning(f"{ticker} abnormal price move (>20%) detected")


def validate_single_ticker(ticker:str)->None:
    """
    Starts the Data validation process for a single ticker
    """
    df = load_ticker_file(ticker)
    if df is None:
        return
    
    logger.info(f"Validating {ticker}")

    check_duplicates(ticker, df)
    check_missing_values(ticker, df)
    check_abnormal_returns(ticker, df)

    logger.info(f"{ticker} validation complete")


def validate_all_tickers():
    """
    Starts the data validation process for all stocks in the portfolio
    """
    logger.info("Starting data validation for all stocks")

    for ticker in STOCKS:
        try:
            validate_single_ticker(ticker)
        except Exception as e:
            logger.error(f"{ticker} validation crashed: {e}")

    logger.info("Data validation finished")


if __name__ == "__main__":
    validate_all_tickers()

