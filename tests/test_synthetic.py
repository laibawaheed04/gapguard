import numpy as np
import pytest

from gapguard.synthetic import make_station_network


def test_shapes():
    data, factors, loadings = make_station_network(n_days=50, n_stations=8)
    assert data.shape == (50, 8)
    assert factors.shape == (50, 3)
    assert loadings.shape == (8, 3)


def test_reproducible():
    a, _, _ = make_station_network(seed=1)
    b, _, _ = make_station_network(seed=1)
    c, _, _ = make_station_network(seed=2)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_really_low_rank():
    data, _, _ = make_station_network(noise=0.0)
    assert np.linalg.matrix_rank(data) == 3


def test_no_nans():
    data, _, _ = make_station_network()
    assert not np.isnan(data).any()


# Stretch goals
@pytest.mark.parametrize("rank", [1, 2, 5])
def test_rank_parameter(rank):
    data, factors, loadings = make_station_network(noise=0.0, rank=rank)
    assert factors.shape[1] == rank
    assert loadings.shape[1] == rank
    assert np.linalg.matrix_rank(data) == rank


def test_offsets_change_station_means():
    flat, _, _ = make_station_network(noise=0.0, offset_scale=0.0)
    shifted, _, _ = make_station_network(noise=0.0, offset_scale=5.0)
    assert not np.allclose(flat.mean(axis=0), shifted.mean(axis=0))
    # The constant offset column adds one to the rank.
    assert np.linalg.matrix_rank(shifted) == 4


def test_invalid_rank():
    with pytest.raises(ValueError):
        make_station_network(rank=0)


def test_short_series_raises_clear_error():
    with pytest.raises(ValueError, match="n_days"):
        make_station_network(n_days=4, n_stations=3)


def test_small_valid_size_works():
    data, factors, _ = make_station_network(n_days=10, n_stations=4)
    assert data.shape == (10, 4)
    assert factors.shape == (10, 3)


def test_factors_are_standardised():
    _, factors, _ = make_station_network()
    assert np.allclose(factors.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(factors.std(axis=0), 1.0)