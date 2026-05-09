"""Pytest configuration for the test suite.

Some tests use a working-directory-relative path such as
``../data/data.csv``. They were written assuming pytest is invoked from
inside the ``tests/`` directory, but the Makefile runs pytest from the
repository root, which breaks that path.

This autouse, session-scoped fixture switches the working directory to
the ``tests/`` folder for the duration of the test session and restores
the previous one on teardown, keeping the original tests untouched.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_tests_dir():
    previous_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        yield
    finally:
        os.chdir(previous_cwd)
