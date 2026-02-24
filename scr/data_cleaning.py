import pandas as pd
import os
from scr.config import STOCKS
from scr.logger import setup_logger
import numpy as np

logger = setup_logger()

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def load_all_prices()-> pd.DataFrame:
    """
    Load all ticker CSVs and combine into one dataframe of Close prices
    """
    all_prices = []
    for ticker in STOCKS:
        path = f"{RAW_FOLDER}/{ticker}.csv"
        try:
            df = pd.read_csv(path, parse_dates=["Date"])

            df = df[["Date", "Close"]]
            df = df.rename(columns={"Close": ticker})
            all_prices.append(df)

            logger.info(f"{ticker} loaded for feature engineering")
        except Exception as e:
            logger.error(f"{ticker} failed to load in feature stage: {e}")

    market = all_prices[0]
    for df in all_prices[1:]:
        market = pd.merge(market, df, on="Date", how="outer")
    market = market.sort_values("Date")
    market = market.set_index("Date")

    return market

def clean_market_data(market:pd.DataFrame) -> None:
    """
    Prepare aligned dataset for modeling
    """
    market = market.dropna(how="all")
    market = market.ffill()
    market = market.dropna()

    logger.info("Market data cleaned and aligned")

    return market


def create_returns(market:pd.DataFrame) -> None:
    """
    Create log returns and save as a .parquet file
    """
    returns = market.pct_change().dropna()
    returns.to_parquet(f"{PROCESSED_FOLDER}/returns.parquet")
    logger.info("Returns dataset created")

    return returns

def run_feature_engineering() -> None:
    """
    Starts the feature engineering process, by loading all prices into one dataset
    cleans it, computes log returns and saves it as a .parquest file.
    """
    logger.info("Starting feature engineering")

    market = load_all_prices()
    market = clean_market_data(market)
    returns = create_returns(market)

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    run_feature_engineering()
