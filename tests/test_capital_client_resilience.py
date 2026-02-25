from __future__ import annotations

import time

import pytest

from bot.data.capital_client import CapitalAPIError, CapitalClient, TokenBucketLimiter, _parse_retry_after


def test_token_bucket_limiter_applies_wait() -> None:
    limiter = TokenBucketLimiter(rate_per_second=5.0, burst=1)
    limiter.acquire()
    start = time.perf_counter()
    limiter.acquire()
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.15


def test_parse_retry_after_seconds() -> None:
    headers = {"Retry-After": "3"}
    assert _parse_retry_after(headers) == 3.0


def test_parse_retry_after_invalid() -> None:
    headers = {"Retry-After": "not-a-number"}
    assert _parse_retry_after(headers) is None


def _build_client() -> CapitalClient:
    return CapitalClient(
        base_url="https://demo-api-capital.backend-capital.com/api/v1",
        api_key="key",
        identifier="id",
        password="pwd",
    )


def test_get_market_details_validates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"snapshot": {"bid": "2000.0", "offer": "2000.5"}, "instrument": {"epic": "XAUUSD"}}

    monkeypatch.setattr(client, "_request", fake_request)
    payload = client.get_market_details("XAUUSD")
    assert payload["snapshot"]["bid"] == 2000.0
    assert payload["snapshot"]["offer"] == 2000.5


def test_get_market_details_invalid_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"snapshot": {"bid": "2000.0"}}

    monkeypatch.setattr(client, "_request", fake_request)
    with pytest.raises(CapitalAPIError):
        client.get_market_details("XAUUSD")


def test_partial_close_position_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    captured: dict[str, object] = {}

    def fake_request(method, path, **kwargs):  # type: ignore[no-untyped-def]
        captured["method"] = method
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return {"ok": True}

    monkeypatch.setattr(client, "_request", fake_request)
    response = client.partial_close_position("DEAL-1", 0.25)
    assert response == {"ok": True}
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/positions/DEAL-1"
    assert captured["json"] == {"size": 0.25}


def test_partial_close_position_rejects_non_positive_size() -> None:
    client = _build_client()
    with pytest.raises(CapitalAPIError):
        client.partial_close_position("DEAL-1", 0.0)
