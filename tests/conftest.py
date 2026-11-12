import pytest

from tqqq_desk.config import load_config
from tqqq_desk.data import synthetic_bars


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def bars():
    return synthetic_bars(n=1400, seed=3)
