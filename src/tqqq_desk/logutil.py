"""Logging setup with secret redaction: values of secret env vars never reach any log output."""
from __future__ import annotations

import logging
import os

SECRET_ENV = ("ALPACA_KEY", "ALPACA_SECRET", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY")


class RedactSecrets(logging.Filter):
    def __init__(self, names=SECRET_ENV):
        super().__init__()
        self.secrets = [v for n in names if (v := os.environ.get(n)) and len(v) >= 4]

    def _scrub(self, s: str) -> str:
        for v in self.secrets:
            s = s.replace(v, "***REDACTED***")
        return s

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secrets:
            record.msg = self._scrub(record.getMessage())
            record.args = ()
        return True


def setup_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    for h in root.handlers:
        h.filters = [f for f in h.filters if not isinstance(f, RedactSecrets)]
        h.addFilter(RedactSecrets())
    for noisy in ("urllib3", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
