import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import scr.data_ingestion as di


# =========================================================
# FIXTURE: Fake environment (no real files, no real downloads)
# =========================================================

@pytest.fixture
def fake_environment(tmp_path, monkeypatch):
    """
    Creates a temporary RAW folder and patches module variables.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # patch module constants
    monkeypatch.setattr(di, "RAW_FOLDER", str(raw_dir))
    monkeypatch.setattr(di, "STOCKS", ["AAPL", "MSFT"])
    monkeypatch.setattr(di, "START_DATE", "2020-01-01")
    return raw_dir

# =========================================================
# HELPER: Fake yfinance dataframe
# =========================================================

def make_yfinance_df(n=5):
    """
    Simulates yfinance MultiIndex OHLCV dataframe
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["AAPL"]]
    )
    data = {col: np.arange(n) for col in columns}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df

# =========================================================
# TEST: format_yfinance_dataset
# =========================================================

def test_format_yfinance_dataset():
    raw = make_yfinance_df()
    formatted = di.format_yfinance_dataset(raw)
    assert isinstance(formatted, pd.DataFrame)
    assert list(formatted.columns) == ["Close", "High", "Low", "Open", "Volume"]
    assert isinstance(formatted.index, pd.DatetimeIndex)
    assert formatted.index.name == "Date"
    assert len(formatted) == len(raw)

# =========================================================
# TEST: download_full_history
# =========================================================

def test_download_full_history_creates_file(fake_environment, monkeypatch):
    raw_dir = fake_environment
    # fake yfinance download
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: make_yfinance_df())
    di.download_full_history("AAPL")
    file_path = raw_dir / "AAPL.csv"
    assert file_path.exists()
    df = pd.read_csv(file_path)
    assert len(df) > 0


def test_download_full_history_empty_data(fake_environment, monkeypatch):
    # return empty dataframe
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    # should not crash
    di.download_full_history("FAKE")

# =========================================================
# TEST: download_incremental
# =========================================================

def test_download_incremental_appends_data(fake_environment, monkeypatch):
    raw_dir = fake_environment
    # existing csv
    existing = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="B"),
        "Close": np.arange(5),
        "High": np.arange(5),
        "Low": np.arange(5),
        "Open": np.arange(5),
        "Volume": np.arange(5),
    })
    existing.to_csv(raw_dir / "AAPL.csv", index=False)
    # new data from yfinance
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: make_yfinance_df(n=3))
    di.download_incremental("AAPL")
    updated = pd.read_csv(raw_dir / "AAPL.csv")
    assert len(updated) > len(existing)


def test_download_incremental_no_new_data(fake_environment, monkeypatch):
    raw_dir = fake_environment
    existing = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="B"),
        "Close": np.arange(5),
        "High": np.arange(5),
        "Low": np.arange(5),
        "Open": np.arange(5),
        "Volume": np.arange(5),
    })
    existing.to_csv(raw_dir / "AAPL.csv", index=False)
    # empty download
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    di.download_incremental("AAPL")
    updated = pd.read_csv(raw_dir / "AAPL.csv")
    assert len(updated) == len(existing)

# =========================================================
# TEST: update_all_tickers
# =========================================================

def test_update_all_tickers_full_download(fake_environment, monkeypatch):
    raw_dir = fake_environment
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: make_yfinance_df())
    di.update_all_tickers()
    assert (raw_dir / "AAPL.csv").exists()
    assert (raw_dir / "MSFT.csv").exists()


def test_update_all_tickers_incremental(fake_environment, monkeypatch):
    raw_dir = fake_environment
    # create existing files
    for ticker in ["AAPL", "MSFT"]:
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "Close": np.arange(5),
            "High": np.arange(5),
            "Low": np.arange(5),
            "Open": np.arange(5),
            "Volume": np.arange(5),
        })
        df.to_csv(raw_dir / f"{ticker}.csv", index=False)
    monkeypatch.setattr(di.yf, "download", lambda *args, **kwargs: make_yfinance_df(n=2))
    di.update_all_tickers()
    for ticker in ["AAPL", "MSFT"]:
        df = pd.read_csv(raw_dir / f"{ticker}.csv")
        assert len(df) > 5