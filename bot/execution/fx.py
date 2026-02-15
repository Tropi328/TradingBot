from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class FxConversionResult:
    converted_amount: float
    spot_converted_amount: float
    fx_cost: float
    spot_rate: float
    all_in_rate: float
    pair: str


def _normalize_currency(value: str) -> str:
    return str(value or "").strip().upper()


def _pair_key(base: str, quote: str) -> str:
    return f"{_normalize_currency(base)}{_normalize_currency(quote)}"


def resolve_spot_rate(
    *,
    from_currency: str,
    to_currency: str,
    static_rates: Mapping[str, float],
) -> float:
    src = _normalize_currency(from_currency)
    dst = _normalize_currency(to_currency)
    if not src or not dst:
        raise ValueError("from_currency/to_currency must be non-empty")
    if src == dst:
        return 1.0

    direct_key = _pair_key(src, dst)
    direct = static_rates.get(direct_key)
    if direct is not None:
        rate = float(direct)
        if rate <= 0:
            raise ValueError(f"FX rate for {direct_key} must be > 0")
        return rate

    inverse_key = _pair_key(dst, src)
    inverse = static_rates.get(inverse_key)
    if inverse is not None:
        inv = float(inverse)
        if inv <= 0:
            raise ValueError(f"FX rate for {inverse_key} must be > 0")
        return 1.0 / inv

    raise ValueError(f"Missing FX static rate for pair {direct_key} (or inverse {inverse_key})")


class FxConverter:
    def __init__(
        self,
        *,
        fee_rate: float = 0.0,
        fee_mode: str = "all_in_rate",
        rate_source: str = "static",
        static_rates: Mapping[str, float] | None = None,
    ) -> None:
        self.fee_rate = max(0.0, float(fee_rate))
        self.fee_mode = str(fee_mode or "all_in_rate").strip().lower()
        self.rate_source = str(rate_source or "static").strip().lower()
        self.static_rates = {
            _pair_key(key[:3], key[3:]): float(value)
            for key, value in (dict(static_rates or {})).items()
        }
        if self.fee_mode != "all_in_rate":
            raise ValueError("Unsupported fx_fee_mode. Supported: all_in_rate")
        if self.rate_source != "static":
            raise ValueError("Unsupported fx_rate_source. Supported: static")

    def convert(
        self,
        *,
        amount: float,
        from_currency: str,
        to_currency: str,
        apply_fee: bool = True,
    ) -> FxConversionResult:
        src = _normalize_currency(from_currency)
        dst = _normalize_currency(to_currency)
        if src == dst:
            amount_f = float(amount)
            return FxConversionResult(
                converted_amount=amount_f,
                spot_converted_amount=amount_f,
                fx_cost=0.0,
                spot_rate=1.0,
                all_in_rate=1.0,
                pair=_pair_key(src, dst),
            )

        spot_rate = resolve_spot_rate(
            from_currency=src,
            to_currency=dst,
            static_rates=self.static_rates,
        )
        amount_f = float(amount)
        spot_converted = amount_f * spot_rate

        if not apply_fee or self.fee_rate <= 0:
            all_in_rate = spot_rate
            converted = spot_converted
            fx_cost = 0.0
        else:
            # Fee is embedded in a less favorable conversion rate for the client.
            # Positive amounts are converted at a lower rate; negative amounts at a higher one.
            if amount_f >= 0:
                all_in_rate = spot_rate * (1.0 - self.fee_rate)
            else:
                all_in_rate = spot_rate * (1.0 + self.fee_rate)
            converted = amount_f * all_in_rate
            fx_cost = spot_converted - converted

        return FxConversionResult(
            converted_amount=converted,
            spot_converted_amount=spot_converted,
            fx_cost=abs(float(fx_cost)),
            spot_rate=spot_rate,
            all_in_rate=all_in_rate,
            pair=_pair_key(src, dst),
        )

