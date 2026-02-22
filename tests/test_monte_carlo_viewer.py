"""Tests for Monte Carlo live viewer summary parser."""
from __future__ import annotations

import pytest

from tools.monte_carlo_live_viewer import parse_mc_summary


class TestParseMcSummary:
    """Unit tests for parse_mc_summary()."""

    def test_full_data_fraction(self) -> None:
        """All fields present, values in 0-1 fraction format."""
        data = {
            "prob_ruin": 0.124,
            "ruin_dd": 0.50,
            "equity_end_p5": 88.1,
            "equity_end_p50": 132.5,
            "equity_end_p95": 210.7,
            "max_dd_p95": 0.43,
        }
        result = parse_mc_summary(data)
        assert "P(ruin>=50%)=12.4%" in result
        assert "EqEnd p50=132.5 p5=88.1 p95=210.7" in result
        assert "maxDD p95=43.0%" in result

    def test_full_data_percent(self) -> None:
        """Values already in percent (>1) format."""
        data = {
            "prob_ruin": 12.4,
            "ruin_dd": 50,
            "equity_end_p5": 88.1,
            "equity_end_p50": 132.5,
            "equity_end_p95": 210.7,
            "max_dd_p95": 43.0,
        }
        result = parse_mc_summary(data)
        assert "P(ruin>=50%)=12.4%" in result
        assert "maxDD p95=43.0%" in result

    def test_missing_all_fields(self) -> None:
        """Empty dict renders all values as '?'."""
        result = parse_mc_summary({})
        assert "P(ruin>=50%)=?" in result
        assert "EqEnd p50=? p5=? p95=?" in result
        assert "maxDD p95=?" in result

    def test_missing_some_fields(self) -> None:
        """Partial data: present fields render, missing ones show '?'."""
        data = {
            "prob_ruin": 0.05,
            "equity_end_p50": 200.0,
        }
        result = parse_mc_summary(data)
        assert "5.0%" in result
        assert "p50=200.0" in result
        assert "p5=?" in result
        assert "p95=?" in result

    def test_zero_prob_ruin(self) -> None:
        """Zero probability of ruin."""
        data = {
            "prob_ruin": 0.0,
            "ruin_dd": 0.50,
            "equity_end_p5": 100.0,
            "equity_end_p50": 150.0,
            "equity_end_p95": 200.0,
            "max_dd_p95": 0.10,
        }
        result = parse_mc_summary(data)
        assert "P(ruin>=50%)=0.0%" in result

    def test_returns_string(self) -> None:
        """parse_mc_summary always returns a str."""
        assert isinstance(parse_mc_summary({}), str)
        assert isinstance(parse_mc_summary({"prob_ruin": 0.1}), str)

    def test_ruin_dd_fraction_vs_percent(self) -> None:
        """ruin_dd=0.30 should display as 30%."""
        data = {"ruin_dd": 0.30, "prob_ruin": 0.05}
        result = parse_mc_summary(data)
        assert "P(ruin>=30%)=" in result

    def test_non_numeric_values(self) -> None:
        """Non-numeric values degrade gracefully to '?'."""
        data = {
            "prob_ruin": "N/A",
            "equity_end_p50": "bad",
            "max_dd_p95": None,
        }
        result = parse_mc_summary(data)
        assert "?" in result

    def test_pipe_separator(self) -> None:
        """Result uses ' | ' as separator between sections."""
        result = parse_mc_summary({"prob_ruin": 0.1})
        assert result.count(" | ") == 2
