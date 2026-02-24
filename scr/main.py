from scr.bootstrap import ensure_dataset
from scr.dash_app import app

def main():
    ensure_dataset()
    app.run(debug=True)

if __name__ == "__main__":
    main()