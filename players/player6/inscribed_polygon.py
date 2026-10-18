"""
Fast heuristic: large inscribed polygon with integer sides, every maximal
straight run in [5, 30], only 90/270 degree corners (in a possibly rotated frame).

Pipeline per (angle, grid offset):
  1. rotate input by -angle, lay a 5-unit grid, keep cells fully inside
  2. take the largest connected blob
  3. repeatedly: drop pinch cells, drop cells that break the 30-unit run cap
     (notches every 7th cell along an over-long edge), keep largest component
  4. union cells -> polygon, rotate back, validate against the original
Every side is a multiple of 5, so length is an integer, >= 5 automatically.
"""

from collections import deque

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon

CELL = 5  # grid size == minimum side length
MAX_RUN = 6  # 6 cells * 5 = 30 = maximum straight run


# ---------------------------------------------------------------- orientation
def _circ_dist90(a, b):
    d = abs(a - b) % 90
    return min(d, 90 - d)


def dominant_angles(poly, max_peaks=3, min_sep=5.0):
    """Candidate grid angles in degrees (mod 90): length-weighted 4-theta
    circular mean, top histogram peaks, and 0."""
    c = np.asarray(poly.exterior.coords)
    d = np.diff(c, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])

    z = np.sum(L * np.exp(4j * th))
    main = float(np.degrees(np.angle(z) / 4) % 90)

    deg = np.degrees(th) % 90
    hist, _ = np.histogram(deg, bins=90, range=(0, 90), weights=L)
    k = np.array([1, 2, 3, 2, 1]) / 9.0
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="valid")
    peaks = []
    for i in np.argsort(-sm):
        a = i + 0.5
        if sm[i] <= 0:
            break
        if all(_circ_dist90(a, p) >= min_sep for p in peaks):
            peaks.append(a)
        if len(peaks) >= max_peaks:
            break

    out = []
    for a in [main] + peaks + [0.0]:
        if all(_circ_dist90(a, b) > 0.5 for b in out):
            out.append(a)
    return out


# ------------------------------------------------------------------ grid ops
def available_mask(poly, x0, y0, W, H):
    """bool[H, W]: cell (r, c) is fully inside poly. poly should be prepared."""
    xs = x0 + CELL * np.arange(W)
    ys = y0 + CELL * np.arange(H)
    X, Y = np.meshgrid(xs, ys)
    mask = shapely.contains_xy(poly, X + CELL / 2, Y + CELL / 2)  # cheap prefilter
    if mask.any():
        boxes = shapely.box(X[mask], Y[mask], X[mask] + CELL, Y[mask] + CELL)
        mask[mask] = shapely.contains(poly, boxes)
    return mask


def largest_component(g):
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    best = []
    for r, c in np.argwhere(g):
        if seen[r, c]:
            continue
        comp = [(r, c)]
        seen[r, c] = True
        dq = deque(comp)
        while dq:
            a, b = dq.popleft()
            for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= na < H and 0 <= nb < W and g[na, nb] and not seen[na, nb]:
                    seen[na, nb] = True
                    comp.append((na, nb))
                    dq.append((na, nb))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(g, dtype=bool)
    for r, c in best:
        out[r, c] = True
    return out


def fix_pinches(g):
    """Remove one cell from every diagonal-only contact. Returns (grid, changed)."""
    P = np.pad(g, 1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    p1 = a & d & ~b & ~c  # \ diagonal
    p2 = b & c & ~a & ~d  # / diagonal
    changed = p1.any() or p2.any()
    for r, cc in np.argwhere(p1):
        P[r, cc] = False  # top-left cell
    for r, cc in np.argwhere(p2):
        P[r, cc + 1] = False  # top-right cell
    return P[1:-1, 1:-1].copy(), changed


def _run_spans(row):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))  # [s, e)


def cells_to_kill(g):
    """Cells to delete so no straight boundary run exceeds MAX_RUN cells."""
    P = np.pad(g, 1)
    ins = P[1:-1, 1:-1]
    flags = [
        ins & ~P[:-2, 1:-1],  # up-facing edges    (horizontal runs)
        ins & ~P[2:, 1:-1],  # down-facing edges  (horizontal runs)
        ins & ~P[1:-1, :-2],  # left-facing edges  (vertical runs)
        ins & ~P[1:-1, 2:],  # right-facing edges (vertical runs)
    ]
    kill = []
    for k, f in enumerate(flags):
        F = f if k < 2 else f.T
        for i, row in enumerate(F):
            for s, e in _run_spans(row):
                if e - s > MAX_RUN:
                    for j in range(s + MAX_RUN, e, MAX_RUN + 1):
                        kill.append((i, j) if k < 2 else (j, i))
    return kill


def enforce(g, max_iter=1000):
    for _ in range(max_iter):
        g = largest_component(g)
        if not g.any():
            return None
        g, pinched = fix_pinches(g)
        if pinched:
            continue
        kill = cells_to_kill(g)
        if not kill:
            return g
        rr, cc = zip(*kill)
        g[list(rr), list(cc)] = False
    return None


def grid_to_polygon(g, x0, y0):
    rr, cc = np.nonzero(g)
    boxes = shapely.box(
        x0 + cc * CELL, y0 + rr * CELL, x0 + (cc + 1) * CELL, y0 + (rr + 1) * CELL
    )
    u = shapely.union_all(boxes)
    if u.geom_type != "Polygon" or len(u.interiors):
        return None
    return u.simplify(0)  # drops exactly-collinear vertices


def merged_edge_lengths(poly):
    """Lengths of maximal straight runs, merging collinear edges cyclically
    (robust to the ring's start vertex sitting in the middle of a run)."""
    c = np.asarray(poly.exterior.coords)
    keep = np.r_[True, np.hypot(*np.diff(c, axis=0).T) > 0]
    c = c[keep][:-1] if np.allclose(c[keep][0], c[keep][-1]) else c[keep]
    d = np.roll(c, -1, axis=0) - c  # edge i: c[i] -> c[i+1]
    L = np.hypot(d[:, 0], d[:, 1])
    u = d / L[:, None]
    cont = (u * np.roll(u, 1, axis=0)).sum(axis=1) > 1 - 1e-9  # continues edge i-1
    if cont.all():
        return np.array([L.sum()])
    start = int(np.flatnonzero(~cont)[0])
    runs, cur = [], 0.0
    for i in list(range(start, len(L))) + list(range(start)):
        if not cont[i] and cur > 0:
            runs.append(cur)
            cur = 0.0
        cur += L[i]
    runs.append(cur)
    return np.array(runs)


def validate(res, src, tol=1e-6):
    if not res.is_valid or not src.buffer(tol).contains(res):
        return False
    L = merged_edge_lengths(res)
    return bool(
        np.all(np.abs(L - np.round(L)) < tol)
        and L.min() >= 5 - tol
        and L.max() <= 30 + tol
    )


# ---------------------------------------------------------------------- main
def solve(poly, n_off=3, top_k=6, angles=None):
    """
    poly   : shapely Polygon, no holes
    n_off  : grid offsets tried per axis (n_off**2 per angle)
    top_k  : only the top_k candidates (by raw cell count) get the slower
             run-cap enforcement
    Returns best valid shapely Polygon, or None.
    """
    if angles is None:
        angles = dominant_angles(poly)
    c0 = poly.centroid
    offs = np.linspace(0, CELL, n_off, endpoint=False)

    cands = []
    for ang in angles:
        rp = affinity.rotate(poly, -ang, origin=c0) if ang else Polygon(poly.exterior)
        shapely.prepare(rp)
        minx, miny, maxx, maxy = rp.bounds
        for ox in offs:
            for oy in offs:
                x0, y0 = minx + ox, miny + oy
                W = int((maxx - x0) // CELL)
                H = int((maxy - y0) // CELL)
                if W < 1 or H < 1:
                    continue
                g = available_mask(rp, x0, y0, W, H)
                cands.append((int(g.sum()), ang, x0, y0, g))

    cands.sort(key=lambda t: -t[0])
    best = None
    for _, ang, x0, y0, g in cands[:top_k]:
        g = enforce(g.copy())
        if g is None:
            continue
        res = grid_to_polygon(g, x0, y0)
        if res is None:
            continue
        if ang:
            res = affinity.rotate(res, ang, origin=c0)
        if validate(res, poly) and (best is None or res.area > best.area):
            best = res
    return best


if __name__ == "__main__":
    # L-shaped test polygon, plus a rotated copy
    L_shape = Polygon([(0, 0), (200, 0), (200, 80), (90, 80), (90, 210), (0, 210)])
    for name, p in [
        ("axis-aligned", L_shape),
        ("rotated 23.7 deg", affinity.rotate(L_shape, 23.7, origin=(0, 0))),
    ]:
        r = solve(p)
        print(
            f"{name}: input area {p.area:.0f}, "
            f"result area {r.area if r else 0:.0f}, "
            f"vertices {len(r.exterior.coords) - 1 if r else 0}"
        )
