"""Tests for the tariff / bill calculation."""

from __future__ import annotations

import pytest

from tariffs import Tariff, compute_bill, parse_slabs


def test_flat_bill():
    bill = compute_bill([1.0, 2.0, 3.0], Tariff(rate=7.0))
    assert bill["total_kwh"] == pytest.approx(6.0)
    assert bill["energy_charge"] == pytest.approx(42.0)
    assert bill["daily_costs"] == pytest.approx([7.0, 14.0, 21.0])


def test_flat_with_tax_and_fixed():
    bill = compute_bill([10.0], Tariff(rate=10.0, fixed=50.0, tax_pct=5.0))
    assert bill["energy_charge"] == pytest.approx(100.0)
    assert bill["tax"] == pytest.approx(5.0)
    assert bill["total"] == pytest.approx(155.0)


def test_slab_daily_costs_sum_to_energy_charge():
    tariff = Tariff(slabs=parse_slabs("0-100:3.5,100-200:4.5,200-400:6.5,400-100000:7"))
    values = [50.0, 80.0, 120.0, 60.0]
    bill = compute_bill(values, tariff)
    assert sum(bill["daily_costs"]) == pytest.approx(bill["energy_charge"])
    # Telescopic over the 310 kWh total: 100@3.5 + 100@4.5 + 110@6.5.
    expected = 100 * 3.5 + 100 * 4.5 + 110 * 6.5
    assert bill["energy_charge"] == pytest.approx(expected)


def test_slab_progressive_marginal():
    tariff = Tariff(slabs=[(100.0, 3.5), (200.0, 4.5)])
    bill = compute_bill([150.0], tariff)
    assert bill["energy_charge"] == pytest.approx(100 * 3.5 + 50 * 4.5)


def test_parse_slabs_ok():
    slabs = parse_slabs("0-100:3.5,100-200:4.5,200-100000:7")
    assert slabs == [(100.0, 3.5), (200.0, 4.5), (100000.0, 7.0)]


@pytest.mark.parametrize(
    "spec",
    [
        "0-100:3.5,150-200:4.5",  # gap between slabs
        "50-100:3.5",             # does not start at 0
        "100-50:3.5",             # lower bound >= upper bound
        "0-100:-1",               # negative rate
        "nonsense",               # unparseable
    ],
)
def test_parse_slabs_rejects_bad(spec):
    with pytest.raises(ValueError):
        parse_slabs(spec)
