from __future__ import annotations

import main as main_module


class _FakeDashboard:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


def test_backtest_viewers_do_not_block_when_hold_disabled(monkeypatch) -> None:
    dashboard = _FakeDashboard()
    proc = _FakeProc()
    monkeypatch.setattr(main_module, "_dashboard_server", dashboard)
    monkeypatch.setattr(main_module, "_mc_viewer_proc", proc)
    monkeypatch.setattr(main_module, "_mc_viewer_stderr_fh", None)

    main_module._maybe_block_dashboard(hold_open=False)

    assert dashboard.shutdown_called is True
    assert proc.terminated is True
    assert proc.waited is True
    assert main_module._dashboard_server is None
    assert main_module._mc_viewer_proc is None
