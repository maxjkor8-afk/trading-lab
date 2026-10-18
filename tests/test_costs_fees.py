"""Fee arithmetic and the venue table.

Expected values are worked out from the published percentages by hand, not read
back from the implementation.
"""

from __future__ import annotations

import pytest

from harness.costs import (
    VENUE_FEES,
    ZERO_FEES,
    FeeSchedule,
    pessimistic_costs,
    worst_fees,
)
from harness.enums import InstrumentType, Liquidity, Side


def test_taker_fee_on_round_notional():
    """0.10% of $10,000 is $10. Checkable without a calculator."""
    fees = FeeSchedule(maker_bps=10.0, taker_bps=10.0)
    assert fees.fee(10_000.0, Liquidity.TAKER) == pytest.approx(10.0)


def test_maker_and_taker_differ_on_perps():
    """Binance USD-M base tier: 0.02% maker, 0.05% taker."""
    fees = VENUE_FEES["binance_perp"]
    assert fees.fee(100_000.0, Liquidity.MAKER) == pytest.approx(20.0)
    assert fees.fee(100_000.0, Liquidity.TAKER) == pytest.approx(50.0)


def test_bybit_perp_taker_is_five_and_a_half_bps():
    """0.055% of $100,000 is $55."""
    assert VENUE_FEES["bybit_perp"].fee(100_000.0, Liquidity.TAKER) == pytest.approx(55.0)


def test_spot_schedules_are_ten_bps_both_sides():
    for key in ("binance_spot", "bybit_spot"):
        fees = VENUE_FEES[key]
        assert fees.maker_bps == 10.0
        assert fees.taker_bps == 10.0
        assert fees.instrument is InstrumentType.SPOT


def test_fee_is_charged_on_absolute_notional():
    """A sell is not a rebate: fees are paid on the size, whatever the sign."""
    fees = FeeSchedule(maker_bps=10.0, taker_bps=10.0)
    assert fees.fee(-50_000.0, Liquidity.TAKER) == pytest.approx(50.0)


def test_zero_fees_charges_nothing():
    assert ZERO_FEES.fee(1_000_000.0, Liquidity.TAKER) == 0.0


def test_worst_fees_picks_the_harshest_taker():
    """The no-argument default must never understate costs."""
    assert worst_fees().taker_bps == max(f.taker_bps for f in VENUE_FEES.values())
    assert worst_fees().taker_bps == 10.0  # the spot schedules
    perp = worst_fees(InstrumentType.PERP)
    assert perp.taker_bps == 5.5  # bybit perp is the worse of the two perps
    assert perp.instrument is InstrumentType.PERP


def test_default_cost_model_is_pessimistic():
    """A forgotten venue argument must cost more, not less."""
    costs = pessimistic_costs()
    assert costs.default_liquidity is Liquidity.TAKER
    assert costs.fees.taker_bps == 10.0
    assert costs.funding.always_adverse is True
    assert costs.slippage.half_spread_bps > 0


def test_cost_model_assumes_taker_unless_told_otherwise():
    """Assuming maker fills is OpenPM failure mode #2."""
    costs = pessimistic_costs("binance", InstrumentType.PERP)
    assert costs.fee(100_000.0) == pytest.approx(50.0)  # taker
    assert costs.fee(100_000.0, Liquidity.MAKER) == pytest.approx(20.0)


def test_pessimistic_costs_accepts_venue_and_instrument():
    costs = pessimistic_costs("bybit", InstrumentType.PERP)
    assert costs.fees.venue == "bybit"
    assert costs.fees.taker_bps == 5.5
    assert pessimistic_costs("binance_spot").fees.taker_bps == 10.0


def test_unknown_venue_is_rejected_loudly():
    with pytest.raises(KeyError, match="unknown venue"):
        pessimistic_costs("ftx", InstrumentType.SPOT)


def test_with_zero_fees_keeps_everything_else():
    costs = pessimistic_costs("binance", InstrumentType.PERP)
    stripped = costs.with_zero_fees()
    assert stripped.fee(1_000_000.0) == 0.0
    assert stripped.slippage is costs.slippage
    assert stripped.funding is costs.funding


def test_cost_model_rejects_a_slippage_model_that_improves_a_fill():
    """An adverse-only invariant, enforced rather than trusted."""

    class Generous:
        def slippage_bps(self, side, reference_price, quantity):
            return -5.0

        def fill_price(self, side, reference_price, quantity):
            return reference_price * (1.0 - side.sign * 0.001)

    from harness.costs import CostModel

    costs = CostModel(slippage=Generous())
    with pytest.raises(ValueError, match="improved a BUY"):
        costs.fill_price(Side.BUY, 100.0, 1.0)
    with pytest.raises(ValueError, match="improved a SELL"):
        costs.fill_price(Side.SELL, 100.0, 1.0)


def test_cost_model_rejects_a_non_positive_reference_price():
    costs = pessimistic_costs()
    with pytest.raises(ValueError, match="must be positive"):
        costs.fill_price(Side.BUY, 0.0, 1.0)


def test_describe_mentions_the_assumptions():
    text = pessimistic_costs("binance", InstrumentType.PERP).describe()
    assert "binance" in text and "taker" in text and "adverse" in text
