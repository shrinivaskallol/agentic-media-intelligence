"""Shared pytest fixtures and setup."""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Project root and env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Suppress asyncio debug noise in tests
from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT
