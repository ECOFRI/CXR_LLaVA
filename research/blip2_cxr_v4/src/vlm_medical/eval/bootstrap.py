"""Deterministic grouped bootstrap helpers."""

import random


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def resamples(n, reps=1000, seed=42):
    rng = random.Random(seed)
    for _ in range(reps):
        yield [rng.randrange(n) for _ in range(n)]


def interval(values):
    return [percentile(values, 0.025), percentile(values, 0.975)]
