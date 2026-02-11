from __future__ import annotations

import time

from bot.data.capital_client import TokenBucketLimiter, _parse_retry_after


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
