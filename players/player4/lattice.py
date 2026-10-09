"""Exact coordinates for paths whose headings are multiples of 45 degrees.

A coordinate is a + b * sqrt(2)/2 with integers a and b, stored as the pair
(a, b), so a point is the 4-tuple (xa, xb, ya, yb). Pieces have integer
lengths, so every point a path reaches has this form. sqrt(2) is irrational,
which makes a point the origin exactly when all four integers are zero: the
loop-closing test needs no floating-point tolerance.

Headings are indices 0..7 (index * 45 degrees, counterclockwise from +x).
"""

import math

HALF_ROOT2 = math.sqrt(2) / 2

ORIGIN = (0, 0, 0, 0)

UNIT = (
    (1, 0, 0, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 0),
    (0, -1, 0, 1),
    (-1, 0, 0, 0),
    (0, -1, 0, -1),
    (0, 0, -1, 0),
    (0, 1, 0, -1),
)

UNIT_XY = tuple(
    (math.cos(math.radians(45 * h)), math.sin(math.radians(45 * h))) for h in range(8)
)


def step(p: tuple, heading: int, length: int) -> tuple:
    u = UNIT[heading]
    return (
        p[0] + length * u[0],
        p[1] + length * u[1],
        p[2] + length * u[2],
        p[3] + length * u[3],
    )


def to_xy(p: tuple) -> tuple[float, float]:
    return (p[0] + p[1] * HALF_ROOT2, p[2] + p[3] * HALF_ROOT2)


def turn_between(h_from: int, h_to: int) -> int:
    """The signed turn, in 45-degree steps (-3..4), from one heading to another."""
    return (h_to - h_from + 3) % 8 - 3


def length_to_origin(p: tuple, heading: int) -> int | None:
    """The length L > 0 with p + L * UNIT[heading] exactly at the origin, if any."""
    u = UNIT[heading]
    i = next(i for i in range(4) if u[i])
    length = -p[i] * u[i]
    if length > 0 and step(p, heading, length) == ORIGIN:
        return length
    return None


def lengths_to_origin(p: tuple, h1: int, h2: int) -> tuple[int, int] | None:
    """Lengths L1, L2 > 0 with p + L1 * UNIT[h1] + L2 * UNIT[h2] exactly at the
    origin, if any. h1 and h2 must not be parallel."""
    x, y = to_xy(p)
    (ax, ay), (bx, by) = UNIT_XY[h1], UNIT_XY[h2]
    det = ax * by - ay * bx
    l1 = round((y * bx - x * by) / det)
    l2 = round((ay * x - ax * y) / det)
    if l1 > 0 and l2 > 0 and step(step(p, h1, l1), h2, l2) == ORIGIN:
        return l1, l2
    return None
