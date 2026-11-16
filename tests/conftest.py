import pytest

from tqqq_desk import strategies as _strategies


def _always_long(x, prev, p):
    return _strategies.LONG, 1.0, 1.0, ["test: always long"]


# test-only strategy used to exercise order flow regardless of the synthetic market's trend
_strategies.REGISTRY.setdefault("always_long", _strategies.Spec("always_long", "test", _always_long, ()))

from tqqq_desk.config import load_config
from tqqq_desk.data import synthetic_bars


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def bars():
    return synthetic_bars(n=1400, seed=3)
