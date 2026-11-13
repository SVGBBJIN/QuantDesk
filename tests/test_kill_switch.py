from tqqq_desk.risk import evaluate_kill_switch


def ev(cfg, **kw):
    base = dict(equity=100_000, last_equity=100_000, peak_equity=100_000, data_problems={}, mismatch=[],
                already_tripped=False, tripped_reason="", tripped_action="none", risk_cfg=cfg["risk"])
    base.update(kw)
    return evaluate_kill_switch(**base)


def test_all_clear(cfg):
    k = ev(cfg)
    assert not k.halt and k.action == "none" and not k.latch


def test_daily_loss_trips_and_latches(cfg):
    assert not ev(cfg, equity=93_000).halt                 # -7% < 8% limit
    k = ev(cfg, equity=91_000)                             # -9%
    assert k.halt and k.latch and k.action == "flatten" and "daily loss" in k.reasons[0]


def test_drawdown_trips(cfg):
    k = ev(cfg, equity=69_000, last_equity=70_000, peak_equity=100_000)
    assert k.halt and k.latch and any("drawdown" in r for r in k.reasons)
    assert not ev(cfg, equity=71_000, last_equity=71_500, peak_equity=100_000).halt


def test_stale_data_halts_without_latch(cfg):
    k = ev(cfg, data_problems={"QQQ": "last bar 2024-07-03 != expected 2024-07-05"})
    assert k.halt and not k.latch and k.action == "freeze"


def test_mismatch_freezes_even_with_loss(cfg):
    k = ev(cfg, equity=80_000, mismatch=["TQQQ: expected 10 sh, broker has 0 sh"])
    assert k.halt and k.latch and k.action == "freeze"     # unknown book -> never auto-trade


def test_latched_stays_latched(cfg):
    k = ev(cfg, already_tripped=True, tripped_reason="daily loss -9%", tripped_action="flatten")
    assert k.halt and k.latch and k.action == "flatten"


def test_freeze_config(cfg):
    c = cfg.copy_with({"risk.kill_action": "freeze"})
    assert ev(c, equity=85_000).action == "freeze"
