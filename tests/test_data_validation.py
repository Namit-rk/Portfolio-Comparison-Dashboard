# tests/test_data_validation.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import scr.data_validation as dv


# =========================================================
# FIXTURE: Fake raw data environment
# =========================================================

@pytest.fixture
def fake_validation_env(tmp_path, monkeypatch):
    """
    Creates temporary RAW folder and patches module constants.
    """

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    monkeypatch.setattr(dv, "RAW_FOLDER", str(raw_dir))
    monkeypatch.setattr(dv, "STOCKS", ["AAPL", "MSFT"])

    return raw_dir


# =========================================================
# HELPER: create ticker csv
# =========================================================

def create_csv(path, df):
    df.to_csv(path, index=False)


def valid_dataframe(n=6):
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "Close": np.linspace(100, 105, n),
        "High": np.linspace(101, 106, n),
        "Low": np.linspace(99, 104, n),
        "Open": np.linspace(100, 105, n),
        "Volume": np.arange(n) + 1000
    })


# =========================================================
# TEST: load_ticker_file
# =========================================================

def test_load_ticker_file_success(fake_validation_env):
    raw_dir = fake_validation_env

    df = valid_dataframe()
    create_csv(raw_dir / "AAPL.csv", df)

    loaded = dv.load_ticker_file("AAPL")

    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == len(df)


def test_load_ticker_file_missing(fake_validation_env):
    # file does not exist
    result = dv.load_ticker_file("FAKE")
    assert result is None


# =========================================================
# TEST: duplicate detection
# =========================================================

def test_check_duplicates(fake_validation_env, monkeypatch):
    df = valid_dataframe()
    df = pd.concat([df, df.iloc[[0]]])  # create duplicate row

    called = {"warned": False}

    def fake_warning(msg):
        called["warned"] = True

    monkeypatch.setattr(dv.logger, "warning", fake_warning)

    dv.check_duplicates("AAPL", df)

    assert called["warned"] is True


# =========================================================
# TEST: missing values detection
# =========================================================

def test_check_missing_values(fake_validation_env, monkeypatch):
    df = valid_dataframe()
    df.loc[2, "Close"] = np.nan

    called = {"warned": False}

    def fake_warning(msg):
        called["warned"] = True

    monkeypatch.setattr(dv.logger, "warning", fake_warning)

    dv.check_missing_values("AAPL", df)

    assert called["warned"] is True


# =========================================================
# TEST: abnormal returns
# =========================================================

def test_check_abnormal_returns(fake_validation_env, monkeypatch):
    df = valid_dataframe()

    # create a 30% jump
    df.loc[3, "Close"] = df.loc[2, "Close"] * 1.30

    called = {"warned": False}

    def fake_warning(msg):
        called["warned"] = True

    monkeypatch.setattr(dv.logger, "warning", fake_warning)

    dv.check_abnormal_returns("AAPL", df)

    assert called["warned"] is True


# =========================================================
# TEST: validate_single_ticker
# =========================================================

def test_validate_single_ticker_runs(fake_validation_env):
    raw_dir = fake_validation_env

    df = valid_dataframe()
    create_csv(raw_dir / "AAPL.csv", df)

    # should run without crashing
    dv.validate_single_ticker("AAPL")


def test_validate_single_ticker_missing_file(fake_validation_env):
    # should safely exit
    dv.validate_single_ticker("FAKE")


# =========================================================
# TEST: validate_all_tickers
# =========================================================

def test_validate_all_tickers(fake_validation_env):
    raw_dir = fake_validation_env

    # create csvs for both tickers
    df = valid_dataframe()
    create_csv(raw_dir / "AAPL.csv", df)
    create_csv(raw_dir / "MSFT.csv", df)

    # should not raise exception
    dv.validate_all_tickers()