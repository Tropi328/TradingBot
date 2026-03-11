from __future__ import annotations

import bot.app.viewer as viewer_module


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

    # Patch the ViewerManager instance that _maybe_block_dashboard actually uses
    monkeypatch.setattr(viewer_module._viewer_manager, "dashboard_server", dashboard)
    monkeypatch.setattr(viewer_module._viewer_manager, "mc_viewer_proc", proc)
    monkeypatch.setattr(viewer_module._viewer_manager, "mc_viewer_stderr_fh", None)

    # Also patch the legacy module-level variables so _load_legacy_viewer_state
    # doesn't overwrite the ViewerManager state
    monkeypatch.setattr(viewer_module, "_dashboard_server", dashboard)
    monkeypatch.setattr(viewer_module, "_mc_viewer_proc", proc)
    monkeypatch.setattr(viewer_module, "_mc_viewer_stderr_fh", None)

    viewer_module._maybe_block_dashboard(hold_open=False)

    assert dashboard.shutdown_called is True
    assert proc.terminated is True
    assert proc.waited is True
    # After cleanup, legacy state should be None
    assert viewer_module._dashboard_server is None
    assert viewer_module._mc_viewer_proc is None
