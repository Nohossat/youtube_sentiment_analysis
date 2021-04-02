import os
import logging


def start_logging(module_path):
    """
    Fetch or create a logging file for the app
    :param module_path: path to the module root
    :return: None
    """
    try:
        logging.basicConfig(filename=os.path.join(module_path, "logs", "monitoring.log"), level=logging.DEBUG)
    except FileNotFoundError:
        logs_folder = os.path.join(module_path, "logs")
        filename = os.path.join(logs_folder, "monitoring.log")

        if not os.exists(filename):
            os.mkdir(logs_folder)
            with open(filename, "w") as f:
                f.write("")

        logging.basicConfig(filename=filename, level=logging.DEBUG)
