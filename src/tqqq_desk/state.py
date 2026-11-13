"""Persistent state (state.json) + append-only CSV logs (trades / signals / equity)."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_STATE: dict[str, Any] = {
    "mode": "paper",
    "position": {"position": "CASH", "days_held": 0, "sqqq_cooldown": 0},
    "expected_positions": {},
    "pending_orders": [],
    "last_signal_date": None,
    "last_run_at": None,
    "peak_equity": 0.0,
    "kill_switch": {"tripped": False, "reason": "", "action": "none", "tripped_at": None},
    "last_decision": {},
    "model": {},
    "data_last_bar": None,
}

CSV_FIELDS = {
    "trades.csv": ["ts", "signal_date", "event", "symbol", "side", "qty", "client_order_id", "status",
                   "filled_qty", "fill_price", "mode", "note"],
    "signals.csv": ["signal_date", "regime", "p_up", "p_hivol", "model_mode", "state", "w_TQQQ", "w_SQQQ",
                    "target_symbol", "target_qty", "halted", "reason"],
    "equity.csv": ["date", "equity", "last_equity", "cash", "peak_equity", "drawdown", "mode"],
}


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def load(self) -> dict:
        if not self.state_path.exists():
            return json.loads(json.dumps(DEFAULT_STATE))
        s = json.loads(self.state_path.read_text())
        for k, v in DEFAULT_STATE.items():
            s.setdefault(k, json.loads(json.dumps(v)))
        return s

    def save(self, state: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.state_path)  # atomic

    def append(self, name: str, row: dict) -> None:
        path = self.root / name
        fields = CSV_FIELDS[name]
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})
