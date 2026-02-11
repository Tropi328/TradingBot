from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AlertConfig:
    enabled: bool = True
    discord_webhook: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    cooldown_seconds: int = 30


class AlertDispatcher:
    def __init__(self, config: AlertConfig):
        self.config = config
        self._last_sent_ts: dict[str, float] = {}

    def send(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        context: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        key = dedupe_key or event
        now = time.monotonic()
        prev = self._last_sent_ts.get(key, 0.0)
        if (now - prev) < self.config.cooldown_seconds:
            return
        self._last_sent_ts[key] = now

        details = f"[{level.upper()}] {event}: {message}"
        if context:
            context_suffix = " | " + " ".join(f"{k}={v}" for k, v in context.items())
            details += context_suffix

        self._send_discord(details)
        self._send_telegram(details)

    def _send_discord(self, text: str) -> None:
        webhook = (self.config.discord_webhook or "").strip()
        if not webhook:
            return
        try:
            response = requests.post(webhook, json={"content": text}, timeout=10)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Discord alert failed: %s", exc)

    def _send_telegram(self, text: str) -> None:
        bot_token = (self.config.telegram_bot_token or "").strip()
        chat_id = (self.config.telegram_chat_id or "").strip()
        if not bot_token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Telegram alert failed: %s", exc)
