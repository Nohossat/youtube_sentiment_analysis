import os

import pytest

import nohossat_cas_pratique
from nohossat_cas_pratique.logging_app import start_logging

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))


def test_start_logging():
    start_logging(module_path)

    assert os.path.exists(os.path.join(module_path, "logs", "monitoring.log")), "The log should exists"


def test_start_logging_file_error():
    log_path = "fake.log"
    start_logging(log_path)

    assert os.path.exists(os.path.join(module_path, "logs", "monitoring.log")), "The log should exists"
