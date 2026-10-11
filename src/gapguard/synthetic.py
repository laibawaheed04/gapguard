"""Generate realistic fake sensor networks for testing GapGuard.

Real stations in one region share a few hidden patterns (season, weather
fronts). We build data the same way, so we know the true values and can
measure how well missing ones are recovered.
"""

import numpy as np


def _seasonal(n_days, rng):
    """Sine wave with a 365-day period."""
    t = np.arange(n_days)
    return np.sin(2 * np.pi * t / 365)


def _trend(n_days, rng):
    """Slow random drift: cumulative sum of small random steps."""
    return np.cumsum(rng.normal(loc=0.0, scale=0.05, size=n_days))


def _front_window(k):
    """Smoothing window (in days) for the k-th pattern, k >= 2: 5, 7, 9, ..."""
    return 3 + 2 * (k - 1)


def _fronts(n_days, rng, window=5):
    """Smooth random wiggles lasting a few days (moving average of noise)."""
    raw = rng.normal(size=n_days)
    return np.convolve(raw, np.ones(window) / window, mode="same")


def make_station_network(
    n_days=730,
    n_stations=20,
    noise=0.1,
    seed=0,
    rank=3,
    offset_scale=0.0,
):
    """Return (data, factors, loadings) for a fake station network.

    data     : array (n_days, n_stations), the clean readings
    factors  : array (n_days, rank), the shared hidden patterns
    loadings : array (n_stations, rank), how each station follows them

    rank         : number of hidden patterns. The first three are season,
                   trend and weather fronts; extra ones are weather fronts
                   with progressively wider smoothing windows (5, 7, 9, ...).
    offset_scale : if > 0, each station gets its own average level drawn
                   from N(0, offset_scale). Note this adds a constant
                   column, so the data is then no longer exactly rank
                   `rank` when noise=0 (it becomes rank + 1).
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")

    # The moving average needs at least as many days as its window (otherwise
    # np.convolve returns a longer array), and we need n_days >= rank for the
    # data to be able to reach that rank.
    longest_window = _front_window(rank - 1) if rank >= 3 else 0
    min_days = max(2, rank, longest_window)
    if n_days < min_days:
        raise ValueError(
            f"n_days must be at least {min_days} for rank={rank}, got {n_days}"
        )

    # One generator for all randomness, so the same seed gives the same data.
    rng = np.random.default_rng(seed)

    # Build each hidden pattern, one 1D array of length n_days each.
    patterns = []
    for k in range(rank):
        if k == 0:
            p = _seasonal(n_days, rng)
        elif k == 1:
            p = _trend(n_days, rng)
        else:
            p = _fronts(n_days, rng, window=_front_window(k))
        # Centre and put every pattern on the same scale so none dominates.
        patterns.append((p - p.mean()) / p.std())

    # Stack into (n_days, rank).
    factors = np.column_stack(patterns)

    # How strongly each station follows each pattern.
    loadings = rng.normal(size=(n_stations, rank))

    # (n_days, rank) @ (rank, n_stations) -> (n_days, n_stations)
    data = factors @ loadings.T

    # Optional per-station average level (e.g. a cold mountain station).
    if offset_scale > 0:
        data = data + offset_scale * rng.normal(size=n_stations)

    # Measurement noise.
    data = data + noise * rng.normal(size=data.shape)

    return data, factors, loadings