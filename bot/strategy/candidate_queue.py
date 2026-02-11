from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bot.strategy.contracts import SetupCandidate


@dataclass(slots=True)
class _QueuedCandidate:
    key: str
    candidate: SetupCandidate


class CandidateQueue:
    def __init__(self) -> None:
        self._by_symbol: dict[str, dict[str, _QueuedCandidate]] = {}

    @staticmethod
    def _key(symbol: str, strategy: str, candidate: SetupCandidate) -> str:
        setup_id = str(candidate.metadata.get("setup_id") or candidate.candidate_id)
        return f"{symbol}:{strategy}:{setup_id}:{candidate.side}"

    def put_many(
        self,
        *,
        symbol: str,
        strategy: str,
        candidates: list[SetupCandidate],
        now: datetime,
    ) -> list[SetupCandidate]:
        symbol_key = symbol.strip().upper()
        queue = self._by_symbol.setdefault(symbol_key, {})
        expired_keys = [
            key
            for key, item in queue.items()
            if item.candidate.expires_at <= now
        ]
        for key in expired_keys:
            queue.pop(key, None)

        for candidate in candidates:
            key = self._key(symbol_key, strategy, candidate)
            existing = queue.get(key)
            if existing is None or candidate.expires_at > existing.candidate.expires_at:
                queue[key] = _QueuedCandidate(key=key, candidate=candidate)

        output: list[SetupCandidate] = []
        for item in queue.values():
            if item.candidate.strategy_name == strategy and item.candidate.expires_at > now:
                output.append(item.candidate)
        output.sort(key=lambda candidate: candidate.created_at, reverse=True)
        return output
