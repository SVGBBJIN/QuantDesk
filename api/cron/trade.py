"""Vercel Cron entry point: trade. Configuration comes from Vercel env vars (see tqqq_desk/vercel_jobs.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tqqq_desk.vercel_jobs import http_handler  # noqa: E402


class handler(http_handler("trade")):  # noqa: N801 - Vercel looks for a class named `handler`
    pass
