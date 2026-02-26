# tests/test_feature_engineering.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# import your module
import scr.data_cleaning as fe


# =========================================================
# FIXTURE: Create a fake market environment
# =========================================================

@pytest.fixture
def fake_market_environment(tmp_path, monkeypatch):
    """
    Creates temporary raw/processed folders and synthetic CSV price files.
    This prevents tests from touching real data.
    """

    # create temp folders
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    raw_dir.mkdir()
    processed_dir.mkdir()

    # patch module constants
    monkeypatch.setattr(fe, "RAW_FOLDER", str(raw_dir))
    monkeypatch.setattr(fe, "PROCESSED_FOLDER", str(processed_dir))

    # fake tickers
    tickers = ["AAPL", "MSFT", "GOOG"]
    monkeypatch.setattr(fe, "STOCKS", tickers)

    # create synthetic price data
    dates = pd.date_range("2020-01-01", periods=6)

    for t in tickers:
        df = pd.DataFrame({
            "Date": dates,
            "Close": np.linspace(100, 110, len(dates)) + np.random.rand(len(dates))
        })
        df.to_csv(raw_dir / f"{t}.csv", index=False)

    return raw_dir, processed_dir, tickers


# =========================================================
# TEST: load_all_prices
# =========================================================

def test_load_all_prices(fake_market_environment):
    raw_dir, processed_dir, tickers = fake_market_environment

    market = fe.load_all_prices()

    # type check
    assert isinstance(market, pd.DataFrame)

    # columns should match tickers
    assert set(market.columns) == set(tickers)

    # index should be datetime
    assert isinstance(market.index, pd.DatetimeIndex)

    # dataframe not empty
    assert len(market) > 0


# =========================================================
# TEST: clean_market_data
# =========================================================

def test_clean_market_data(fake_market_environment):
    market = fe.load_all_prices()

    # deliberately inject missing values
    market.iloc[0, 0] = np.nan
    market.iloc[1, 1] = np.nan

    cleaned = fe.clean_market_data(market)

    # no NaNs should remain
    assert cleaned.isna().sum().sum() == 0

    # still same columns
    assert isinstance(cleaned, pd.DataFrame)


# =========================================================
# TEST: create_returns
# =========================================================

def test_create_returns(fake_market_environment):
    raw_dir, processed_dir, tickers = fake_market_environment

    market = fe.load_all_prices()
    market = fe.clean_market_data(market)

    returns = fe.create_returns(market)

    # dataframe returned
    assert isinstance(returns, pd.DataFrame)

    # number of rows should be one less than price data
    assert len(returns) == len(market) - 1

    # parquet file must exist
    parquet_file = Path(processed_dir) / "returns.parquet"
    assert parquet_file.exists()

    # read file and verify
    saved = pd.read_parquet(parquet_file)
    assert isinstance(saved, pd.DataFrame)
    assert saved.shape == returns.shape


# =========================================================
# INTEGRATION TEST: full pipeline
# =========================================================

def test_run_feature_engineering_pipeline(fake_market_environment):
    raw_dir, processed_dir, tickers = fake_market_environment

    fe.run_feature_engineering()

    # check output file created
    parquet_file = Path(processed_dir) / "returns.parquet"
    assert parquet_file.exists()

    df = pd.read_parquet(parquet_file)

    # should have one column per ticker
    assert len(df.columns) == len(tickers)
    # should not be empty
    assert len(df) > 0