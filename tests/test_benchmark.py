import pytest

from slashcompute.agent.benchmark import MIN_CONTRIB_BYTES, contrib_bytes

MIB = 1024 * 1024
GIB = 1024 * MIB


@pytest.mark.parametrize("cap", [1, 16 * MIB, MIN_CONTRIB_BYTES - 1])
def test_small_memory_cap_is_never_exceeded(cap):
    assert contrib_bytes(8 * GIB, cap) == cap
    # ... not even when the Mac itself has less than the floor available
    assert contrib_bytes(8 * MIB, cap) == cap


def test_memory_cap_above_the_floor_bounds_available_memory():
    assert contrib_bytes(8 * GIB, 512 * MIB) == 512 * MIB
    assert contrib_bytes(8 * GIB, 64 * GIB) == 8 * GIB


def test_uncapped_contribution_keeps_the_floor():
    assert contrib_bytes(8 * GIB) == 8 * GIB
    assert contrib_bytes(8 * MIB) == MIN_CONTRIB_BYTES
    assert contrib_bytes(8 * MIB, 512 * MIB) == MIN_CONTRIB_BYTES


def test_negative_cap_lends_nothing():
    assert contrib_bytes(8 * GIB, -GIB) == 0
