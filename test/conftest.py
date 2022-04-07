# -*- coding: utf-8 -*-

import pytest

"""
--runfast cmd option to avoid slower tests
"""

def pytest_addoption(parser):
    parser.addoption(
        "--runfast", action="store_true", default=False, help="run only fast tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runfast"):
        # --runfast given in cli: do skip slow tests
        skip_slow = pytest.mark.skip(reason="--runfast option is passed")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)