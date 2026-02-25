# from scr.bootstrap import ensure_dataset
from scr.data_ingestion import update_all_tickers
from scr.data_validation import validate_all_tickers
from scr.data_cleaning import run_feature_engineering
from scr.dash_app import app

def main():
    # ensure_dataset()
    update_all_tickers()
    validate_all_tickers()
    run_feature_engineering()

    app.run(debug=True)

if __name__ == "__main__":
    main()