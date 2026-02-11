from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from bot.data.candles import Candle
from bot.storage.models import PositionRecord
from bot.strategy.state_machine import StrategySignal


class DecisionAction(str, Enum):
    TRADE = "TRADE"
    SMALL = "SMALL"
    OBSERVE = "OBSERVE"
    MANAGE = "MANAGE"
    SKIP = "SKIP"


@dataclass(slots=True)
class BiasState:
    symbol: str
    strategy_name: str
    direction: str
    timeframe: str
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SetupCandidate:
    candidate_id: str
    symbol: str
    strategy_name: str
    side: str
    created_at: datetime
    expires_at: datetime
    source_timeframe: str
    setup_type: str
    origin_strategy: str | None = None
    setup_id: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreBreakdown:
    total: float | None
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrategyEvaluation:
    action: DecisionAction
    score_total: float | None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    score_layers: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    gate_blocked: str | None = None
    reasons_blocking: list[str] = field(default_factory=list)
    would_enter_if: list[str] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyDataBundle:
    symbol: str
    now: datetime
    candles_h1: list[Candle]
    candles_m15: list[Candle]
    candles_m5: list[Candle]
    spread: float | None
    spread_history: list[float]
    news_blocked: bool
    entry_state: str
    h1_new_close: bool = False
    m15_new_close: bool = False
    m5_new_close: bool = False
    quote: tuple[float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyOutcome:
    symbol: str
    strategy_name: str
    bias: BiasState
    candidate: SetupCandidate | None
    evaluation: StrategyEvaluation
    order_request: StrategySignal | None
    reason_codes: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MissedOpportunityRecord:
    symbol: str
    origin_strategy: str
    setup_id: str
    direction: str
    created_at: datetime
    hit: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class StrategyPlugin(Protocol):
    name: str

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        ...

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        ...

    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        ...

    def evaluate_candidate(
        self,
        symbol: str,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategyEvaluation:
        ...

    def generate_order(
        self,
        symbol: str,
        evaluation: StrategyEvaluation,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategySignal | None:
        ...

    def manage_position(
        self,
        symbol: str,
        position: PositionRecord,
        data: StrategyDataBundle,
    ) -> list[StrategySignal]:
        ...
