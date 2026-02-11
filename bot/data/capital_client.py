from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class CapitalAPIError(RuntimeError):
    """Non-retryable Capital.com API error."""


class RetryableCapitalAPIError(CapitalAPIError):
    """Retryable API/network error."""


class CapitalAuthError(CapitalAPIError):
    """Authentication/authorization error for session creation."""


@dataclass(slots=True)
class CapitalClientMetrics:
    total_requests: int = 0
    total_retries: int = 0
    http_429_count: int = 0
    network_disconnects: int = 0
    session_refreshes: int = 0
    auth_failures: int = 0


class TokenBucketLimiter:
    def __init__(self, rate_per_second: float, burst: int):
        self.rate_per_second = max(0.1, float(rate_per_second))
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            wait_seconds = 0.0
            with self.lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self.last_refill)
                self.tokens = min(
                    float(self.capacity),
                    self.tokens + elapsed * self.rate_per_second,
                )
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_seconds = (1.0 - self.tokens) / self.rate_per_second
            time.sleep(wait_seconds)


def _is_disconnect_error(exc: requests.RequestException) -> bool:
    text = str(exc).lower()
    patterns = (
        "remote end closed connection",
        "remote disconnected",
        "connection aborted",
        "connection reset",
    )
    return any(item in text for item in patterns)


def _parse_retry_after(headers: Any) -> float | None:
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


class CapitalClient:
    """
    Capital.com REST API client for DEMO.

    Auth flow:
    - POST /session with X-CAP-API-KEY + identifier/password.
    - Read CST and X-SECURITY-TOKEN from response headers.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        identifier: str,
        password: str,
        account_id: str | None = None,
        timeout_seconds: int = 10,
        *,
        rate_limit_rps: float = 2.0,
        rate_limit_burst: int = 5,
        request_max_attempts: int = 6,
        backoff_base_seconds: float = 0.5,
        backoff_max_seconds: float = 20.0,
        reconnect_short_retries: int = 2,
        session_refresh_min_interval_seconds: int = 5,
    ):
        self.base_url = self._normalize_base_url(base_url)
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.account_id = account_id.strip() if account_id else None
        self.timeout_seconds = timeout_seconds
        self.request_max_attempts = max(1, int(request_max_attempts))
        self.backoff_base_seconds = max(0.1, float(backoff_base_seconds))
        self.backoff_max_seconds = max(self.backoff_base_seconds, float(backoff_max_seconds))
        self.reconnect_short_retries = max(0, int(reconnect_short_retries))
        self.session_refresh_min_interval_seconds = max(1, int(session_refresh_min_interval_seconds))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.cst: str | None = None
        self.security_token: str | None = None
        self._auth_cooldown_until: datetime | None = None
        self._auth_error_message: str | None = None
        self._epic_aliases: dict[str, str] = {}
        self._limiter = TokenBucketLimiter(rate_per_second=rate_limit_rps, burst=rate_limit_burst)
        self._session_lock = threading.Lock()
        self._last_session_refresh_at: datetime | None = None
        self._metrics = CapitalClientMetrics()
        self._metrics_lock = threading.Lock()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if normalized.endswith("/api/v1"):
            return normalized
        if normalized.endswith("/api"):
            return f"{normalized}/v1"
        return f"{normalized}/api/v1"

    @staticmethod
    def _extract_error_code(response: requests.Response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        error_code = payload.get("errorCode")
        return str(error_code) if error_code else None

    def _metric_add(self, field_name: str, value: int = 1) -> None:
        with self._metrics_lock:
            setattr(self._metrics, field_name, getattr(self._metrics, field_name) + value)

    def metrics_snapshot(self) -> dict[str, int]:
        with self._metrics_lock:
            snapshot = CapitalClientMetrics(
                total_requests=self._metrics.total_requests,
                total_retries=self._metrics.total_retries,
                http_429_count=self._metrics.http_429_count,
                network_disconnects=self._metrics.network_disconnects,
                session_refreshes=self._metrics.session_refreshes,
                auth_failures=self._metrics.auth_failures,
            )
        return asdict(snapshot)

    def _set_auth_cooldown(self, *, seconds: int, message: str) -> None:
        self._auth_cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        self._auth_error_message = message

    def _auth_headers(self) -> dict[str, str]:
        headers = {"X-CAP-API-KEY": self.api_key}
        if self.cst and self.security_token:
            headers["CST"] = self.cst
            headers["X-SECURITY-TOKEN"] = self.security_token
        return headers

    @staticmethod
    def _normalize_epic(epic: str) -> str:
        return epic.strip().upper()

    def _cache_epic_alias(self, requested: str, actual: str) -> None:
        req = self._normalize_epic(requested)
        act = self._normalize_epic(actual)
        if req and act and req != act:
            self._epic_aliases[req] = act
            LOGGER.info("Resolved epic alias: %s -> %s", req, act)

    def _resolve_cached_epic(self, epic: str) -> str:
        req = self._normalize_epic(epic)
        return self._epic_aliases.get(req, req)

    def _sleep_retry(self, *, endpoint: str, attempt: int, reason: str, retry_after: float | None = None) -> None:
        if retry_after is not None:
            sleep_seconds = max(0.0, retry_after)
        else:
            exponential = min(
                self.backoff_max_seconds,
                self.backoff_base_seconds * (2 ** max(0, attempt - 1)),
            )
            jitter = random.uniform(0.0, max(0.01, exponential * 0.2))
            sleep_seconds = min(self.backoff_max_seconds, exponential + jitter)
        self._metric_add("total_retries", 1)
        LOGGER.warning(
            "Retrying Capital API call endpoint=%s attempt=%d/%d sleep=%.2fs reason=%s",
            endpoint,
            attempt,
            self.request_max_attempts,
            sleep_seconds,
            reason,
        )
        time.sleep(sleep_seconds)

    def _send_http(
        self,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> requests.Response:
        self._limiter.acquire()
        self._metric_add("total_requests", 1)
        return self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=json_payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def _is_recent_session_refresh(self) -> bool:
        if self._last_session_refresh_at is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self._last_session_refresh_at).total_seconds()
        return elapsed < self.session_refresh_min_interval_seconds

    def _invalidate_session(self) -> None:
        self.cst = None
        self.security_token = None

    def _refresh_session(self, reason: str) -> None:
        with self._session_lock:
            if self._is_recent_session_refresh():
                LOGGER.info("Session refresh skipped (recent) reason=%s", reason)
                return
            LOGGER.info("Refreshing Capital session reason=%s", reason)
            self._invalidate_session()
            self.create_session()
            self._last_session_refresh_at = datetime.now(timezone.utc)
            self._metric_add("session_refreshes", 1)

    def _resolve_epic_via_search(self, epic: str) -> str | None:
        term = self._normalize_epic(epic)
        payload = self._request("GET", "/markets", params={"searchTerm": term})
        markets = payload.get("markets", [])
        if not isinstance(markets, list) or not markets:
            return None

        exact = [
            m
            for m in markets
            if str(m.get("epic", "")).upper() == term
        ]
        if exact:
            return str(exact[0].get("epic"))

        if term in {"XAU", "XAUUSD", "GOLD"}:
            commodities_gold = [
                m
                for m in markets
                if str(m.get("instrumentType", "")).upper() == "COMMODITIES"
                and str(m.get("instrumentName", "")).strip().upper() == "GOLD"
            ]
            if commodities_gold:
                return str(commodities_gold[0].get("epic"))

        commodities = [
            m for m in markets if str(m.get("instrumentType", "")).upper() == "COMMODITIES"
        ]
        if commodities:
            return str(commodities[0].get("epic"))
        return str(markets[0].get("epic")) if markets[0].get("epic") else None

    def create_session(self) -> None:
        now = datetime.now(timezone.utc)
        if self._auth_cooldown_until and now < self._auth_cooldown_until:
            wait_seconds = int((self._auth_cooldown_until - now).total_seconds())
            message = self._auth_error_message or "Session auth is temporarily blocked"
            raise CapitalAuthError(f"{message}. Retry in ~{wait_seconds}s")

        last_exc: Exception | None = None
        for attempt in range(1, self.request_max_attempts + 1):
            try:
                response = self._send_http(
                    method="POST",
                    path="/session",
                    headers={"X-CAP-API-KEY": self.api_key},
                    json_payload={
                        "identifier": self.identifier,
                        "password": self.password,
                        "encryptedPassword": False,
                    },
                )
            except requests.RequestException as exc:
                last_exc = exc
                if _is_disconnect_error(exc):
                    self._metric_add("network_disconnects", 1)
                if attempt >= self.request_max_attempts:
                    break
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason=f"network:{type(exc).__name__}",
                )
                continue

            if response.status_code == 429:
                self._metric_add("http_429_count", 1)
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Retryable session error: HTTP {response.status_code} {response.text}"
                    )
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason="http_429",
                    retry_after=_parse_retry_after(response.headers),
                )
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Retryable session error: HTTP {response.status_code} {response.text}"
                    )
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason=f"http_{response.status_code}",
                )
                continue

            if response.status_code in (401, 403):
                self._metric_add("auth_failures", 1)
                error_code = self._extract_error_code(response)
                if error_code in {"error.invalid.details", "error.invalid.api.key"}:
                    if error_code == "error.invalid.api.key":
                        message = (
                            "Invalid Capital.com API key (wrong key, disabled key, or key from another environment)"
                        )
                    else:
                        message = (
                            "Invalid Capital.com credentials (API key/identifier/password) "
                            "or wrong account environment"
                        )
                    self._set_auth_cooldown(seconds=90, message=message)
                    raise CapitalAuthError(message)
                if error_code == "error.null.accountId":
                    message = (
                        "Capital.com rejected accountId. Verify CAPITAL_ACCOUNT_ID belongs to this DEMO account."
                    )
                    self._set_auth_cooldown(seconds=90, message=message)
                    raise CapitalAuthError(message)
                if error_code == "error.null.client.token":
                    message = "Missing/invalid API key token. Verify CAPITAL_API_KEY and environment (DEMO vs LIVE)."
                    self._set_auth_cooldown(seconds=90, message=message)
                    raise CapitalAuthError(message)
                raise CapitalAuthError(
                    f"Session authorization failed: HTTP {response.status_code} {response.text}"
                )

            if response.status_code >= 400:
                raise CapitalAPIError(
                    f"Failed to create session: HTTP {response.status_code} {response.text}"
                )

            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")
            if not self.cst or not self.security_token:
                raise CapitalAPIError("Session tokens missing in response headers")
            if self.account_id:
                self._switch_account(self.account_id)
            self._auth_cooldown_until = None
            self._auth_error_message = None
            return

        if last_exc is not None:
            raise RetryableCapitalAPIError(f"Network error creating session: {last_exc}") from last_exc
        raise RetryableCapitalAPIError("Could not create session after retries")

    def _switch_account(self, account_id: str) -> None:
        for attempt in range(1, self.request_max_attempts + 1):
            try:
                response = self._send_http(
                    method="PUT",
                    path="/session",
                    headers=self._auth_headers(),
                    json_payload={"accountId": account_id, "defaultAccount": False},
                )
            except requests.RequestException as exc:
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Could not switch accountId={account_id}: {exc}"
                    ) from exc
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason=f"account_switch_network:{type(exc).__name__}",
                )
                continue
            if response.status_code == 429:
                self._metric_add("http_429_count", 1)
                if attempt >= self.request_max_attempts:
                    break
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason="account_switch_429",
                    retry_after=_parse_retry_after(response.headers),
                )
                continue
            if response.status_code in (500, 502, 503, 504):
                if attempt >= self.request_max_attempts:
                    break
                self._sleep_retry(
                    endpoint="/session",
                    attempt=attempt,
                    reason=f"account_switch_{response.status_code}",
                )
                continue
            if response.status_code >= 400:
                raise CapitalAuthError(
                    f"Could not switch to accountId={account_id}: HTTP {response.status_code} {response.text}"
                )
            return
        raise RetryableCapitalAPIError(f"Could not switch accountId={account_id} after retries")

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        refreshed_once = False

        for attempt in range(1, self.request_max_attempts + 1):
            if auth and (not self.cst or not self.security_token):
                self.create_session()

            try:
                response = self._send_http(
                    method=method,
                    path=path,
                    headers=self._auth_headers() if auth else {},
                    params=params,
                    json_payload=json,
                )
            except requests.RequestException as exc:
                if _is_disconnect_error(exc):
                    self._metric_add("network_disconnects", 1)
                    if auth and attempt > self.reconnect_short_retries and not refreshed_once:
                        self._refresh_session("remote_disconnect")
                        refreshed_once = True
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Network error {method} {path}: {exc}"
                    ) from exc
                self._sleep_retry(
                    endpoint=path,
                    attempt=attempt,
                    reason=f"network:{type(exc).__name__}",
                )
                continue

            if response.status_code in (401, 403) and auth:
                if not refreshed_once:
                    self._refresh_session("session_expired")
                    refreshed_once = True
                    continue
                if attempt >= self.request_max_attempts:
                    raise CapitalAuthError(
                        f"Session authorization failed: HTTP {response.status_code} {response.text}"
                    )
                self._sleep_retry(
                    endpoint=path,
                    attempt=attempt,
                    reason=f"auth_{response.status_code}",
                )
                continue

            if response.status_code == 429:
                self._metric_add("http_429_count", 1)
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Retryable API error: HTTP {response.status_code} {response.text}"
                    )
                self._sleep_retry(
                    endpoint=path,
                    attempt=attempt,
                    reason="http_429",
                    retry_after=_parse_retry_after(response.headers),
                )
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt >= self.request_max_attempts:
                    raise RetryableCapitalAPIError(
                        f"Retryable API error: HTTP {response.status_code} {response.text}"
                    )
                self._sleep_retry(
                    endpoint=path,
                    attempt=attempt,
                    reason=f"http_{response.status_code}",
                )
                continue

            if response.status_code == 404 and allow_404:
                return {}

            if response.status_code >= 400:
                raise CapitalAPIError(
                    f"API error {method} {path}: HTTP {response.status_code} {response.text}"
                )

            if not response.text:
                return {}
            return response.json()

        raise RetryableCapitalAPIError(f"Could not complete request {method} {path}")

    def get_prices(self, epic: str, resolution: str, max_points: int = 500) -> list[dict[str, Any]]:
        resolved_epic = self._resolve_cached_epic(epic)
        payload = self._request(
            "GET",
            f"/prices/{resolved_epic}",
            params={"resolution": resolution, "max": max_points},
            allow_404=True,
        )
        if not payload:
            discovered = self._resolve_epic_via_search(epic)
            if discovered and discovered != resolved_epic:
                self._cache_epic_alias(epic, discovered)
                payload = self._request(
                    "GET",
                    f"/prices/{discovered}",
                    params={"resolution": resolution, "max": max_points},
                    allow_404=True,
                )
        if not payload:
            raise CapitalAPIError(
                f"API error GET /prices/{epic}: epic not found. Try CAPITAL_EPIC=GOLD for Capital DEMO."
            )
        prices = payload.get("prices", [])
        return prices if isinstance(prices, list) else []

    def get_market_details(self, epic: str) -> dict[str, Any]:
        resolved_epic = self._resolve_cached_epic(epic)
        payload = self._request("GET", f"/markets/{resolved_epic}", allow_404=True)
        if payload:
            return payload
        discovered = self._resolve_epic_via_search(epic)
        if discovered and discovered != resolved_epic:
            self._cache_epic_alias(epic, discovered)
            payload = self._request("GET", f"/markets/{discovered}", allow_404=True)
            if payload:
                return payload
        raise CapitalAPIError(
            f"API error GET /markets/{epic}: epic not found. Try CAPITAL_EPIC=GOLD for Capital DEMO."
        )

    def get_quote(self, epic: str) -> tuple[float, float]:
        payload = self.get_market_details(epic)
        snapshot = payload.get("snapshot", {})
        bid = snapshot.get("bid")
        ask = snapshot.get("offer")
        if bid is None or ask is None:
            raise CapitalAPIError(f"Missing bid/ask in market snapshot for {epic}")
        return float(bid), float(ask)

    def place_working_order(
        self,
        *,
        epic: str,
        side: str,
        size: float,
        level: float,
        stop_level: float,
        profit_level: float,
        currency: str,
        expires_at: datetime,
        deal_reference: str | None = None,
    ) -> dict[str, Any]:
        direction = "BUY" if side.upper() == "LONG" else "SELL"
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "level": level,
            "type": "LIMIT",
            "currencyCode": currency,
            "forceOpen": True,
            "guaranteedStop": False,
            "timeInForce": "GOOD_TILL_DATE",
            "goodTillDate": expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "stopLevel": stop_level,
            "profitLevel": profit_level,
        }
        if deal_reference:
            payload["dealReference"] = deal_reference
        return self._request("POST", "/workingorders", json=payload)

    def cancel_working_order(self, deal_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/workingorders/{deal_id}")

    def get_working_orders(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/workingorders")
        orders = payload.get("workingOrders", [])
        return orders if isinstance(orders, list) else []

    def get_positions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/positions")
        positions = payload.get("positions", [])
        return positions if isinstance(positions, list) else []

    def get_confirmation(self, deal_reference: str) -> dict[str, Any]:
        return self._request("GET", f"/confirms/{deal_reference}", allow_404=True)

    def update_position(
        self,
        deal_id: str,
        *,
        stop_level: float | None = None,
        profit_level: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if stop_level is not None:
            payload["stopLevel"] = stop_level
        if profit_level is not None:
            payload["profitLevel"] = profit_level
        if not payload:
            return {}
        return self._request("PUT", f"/positions/{deal_id}", json=payload)

    def partial_close_position(self, deal_id: str, size: float) -> dict[str, Any]:
        # TODO: Verify exact partial-close payload for Capital.com in your account type.
        # In dry-run mode this call is not used. In paper mode this attempts a common pattern.
        payload = {"size": size}
        return self._request("DELETE", f"/positions/{deal_id}", json=payload)
