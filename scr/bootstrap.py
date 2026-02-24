from pathlib import Path
from scr.data_ingestion import update_all_tickers
from scr.data_validation import validate_all_tickers
from scr.data_cleaning import run_feature_engineering

DATA_PATH = Path("data/processed/returns.parquet")

def ensure_dataset():
    """
    Ensures dataset exists before dashboard loads.
    Runs only first time on server.
    """

    if DATA_PATH.exists():
        print("Dataset already exists. Skipping download.")
        return

    print("Preparing dataset for first launch...")

    update_all_tickers()
    validate_all_tickers()
    run_feature_engineering()

    print("Dataset ready.")

if __name__ == "__main__":
    ensure_dataset()