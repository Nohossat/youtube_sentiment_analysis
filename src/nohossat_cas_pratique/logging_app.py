import os
import logging


def start_logging(module_path):
    try:
        logging.basicConfig(filename=os.path.join(module_path, "logs", "monitoring.log"), level=logging.DEBUG)
    except FileNotFoundError:
        logs_folder = os.path.join(module_path, "logs")
        filename = os.path.join(logs_folder, "monitoring.log")
        os.mkdir(logs_folder)
        with open(filename, "w") as f:
            f.write("")

        logging.basicConfig(filename=filename, level=logging.DEBUG)
