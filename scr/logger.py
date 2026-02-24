import logging
import os

def setup_logger():
    # create logs folder if not exists
    os.makedirs("logs", exist_ok=True)

    log_file = "logs/workflow.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )

    return logging.getLogger()
