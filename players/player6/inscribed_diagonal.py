"""Turn zigzag staircases of an inscribed rectilinear polygon into diagonals.

The inscribed solver returns a polygon whose faces are all axis-aligned in some
frame, so a room wall at 45 degrees is followed with a staircase: many short
faces and a corner (a right connector) at every step. A single diagonal face
replaces a whole staircase with two diagonal connectors.

Why stairs have to be replaced in *pairs*
-----------------------------------------
A diagonal face of integer length L moves L/sqrt(2) along each axis, which is
irrational, while every other face has integer length. The loop can only close
if those irrational parts cancel, so two staircases of the same size that run
in opposite directions are replaced by two diagonals of the same length L.
Each replacement shifts the faces between the two staircases by the same small
vector along the diagonal, in opposite directions, so nothing else changes
length and the polygon still closes exactly.

Everything here is on abstract face lists: ``lengths[i]`` is the length of face
i and ``turns[i]`` the turn after it in 45-degree units (+-2 for a right
connector, +-1 for a diagonal one). Headings are in the same units.
"""

import math
from itertools import combinations

from players.player6.diagonal_search import DIRS, SQ


def _norm(t):
    """Wrap a turn in 45-degree units to (-4, 4]."""
    t = t % 8
    return t - 8 if t > 4 else t


def to_faces(coords, tol=1e-3):
    """Face list of a rectilinear polygon given as vertices.

    Returns (lengths, turns, alpha) where alpha is the heading in degrees of
    face 0, or None if the polygon is not rectilinear with integer faces."""
    pts = list(coords)
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    # drop collinear vertices
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
            if abs(cross) < 1e-9 and dot > 0:
                del pts[i]
                changed = True
                break
    n = len(pts)
    if n < 4:
        return None
    lengths, turns = [], []
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if abs(length - round(length)) > tol:
            return None
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        ang = math.degrees(
            math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
        )
        units = round(ang / 45.0)
        if abs(ang - 45.0 * units) > 0.5 or abs(units) != 2:
            return None
        lengths.append(int(round(length)))
        turns.append(units)
    alpha = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    return lengths, turns, alpha


def headings(turns):
    """Heading (45-degree units, face 0 = 0) of every face."""
    h, out = 0, []
    for t in turns:
        out.append(h % 8)
        h += t
    return out


def walk(lengths, turns):
    """Float vertices (start of each face) from the origin at heading 0, plus
    the exact closure error as integers (xm, xn, ym, yn)."""
    pos = (0, 0, 0, 0)
    h = 0
    verts = []
    for length, t in zip(lengths, turns):
        xm, xn, ym, yn = pos
        verts.append((xm + xn * SQ, ym + yn * SQ))
        dxm, dxn, dym, dyn = DIRS[h % 8]
        pos = (
            xm + dxm * length,
            xn + dxn * length,
            ym + dym * length,
            yn + dyn * length,
        )
        h += t
    return verts, pos


_DIAG_HEADING = {(1, 1): 1, (-1, 1): 3, (-1, -1): 5, (1, -1): 7}


def staircases(lengths, turns, min_faces=2):
    """Every run of consecutive faces that is a staircase replaceable by one
    diagonal: only two orthogonal headings, equal total length along each, and
    legal turns into and out of the diagonal.

    Yields dicts: start, count, a (leg), sign (sx, sy), diag heading, turn_in,
    turn_out."""
    n = len(lengths)
    hd = headings(turns)
    out = []
    for i in range(n):
        for m in range(min_faces, n - 1):
            idx = [(i + k) % n for k in range(m)]
            hs = {hd[j] for j in idx}
            if len(hs) != 2:
                break  # a longer window starting here only has more headings
            d1, d2 = sorted(hs)
            if (d2 - d1) % 8 not in (2, 6):
                break
            dx = dy = 0
            for j in idx:
                ddx = DIRS[hd[j]][0]
                ddy = DIRS[hd[j]][2]
                dx += lengths[j] * ddx
                dy += lengths[j] * ddy
            if abs(dx) != abs(dy) or dx == 0:
                continue
            a = abs(dx)
            sign = (1 if dx > 0 else -1, 1 if dy > 0 else -1)
            diag = _DIAG_HEADING[sign]
            h_prev = hd[(i - 1) % n]
            h_next = hd[(i + m) % n]
            t_in = _norm(diag - h_prev)
            t_out = _norm(h_next - diag)
            if abs(t_in) not in (1, 2) or abs(t_out) not in (1, 2):
                continue
            out.append(
                {
                    "start": i,
                    "idx": idx,
                    "count": m,
                    "a": a,
                    "sign": sign,
                    "t_in": t_in,
                    "t_out": t_out,
                }
            )
    return out


def apply_pair(lengths, turns, w1, w2, L):
    """New (lengths, turns, first_face_heading) with both staircases replaced
    by a diagonal of length L. The list is rotated to start at an unreplaced
    face, whose original heading is returned so the caller knows the frame.
    Returns None if the windows overlap or touch."""
    n = len(lengths)
    s1, s2 = set(w1["idx"]), set(w2["idx"])
    if s1 & s2:
        return None
    hd = headings(turns)
    first = next((j for j in range(n) if j not in s1 and j not in s2), None)
    if first is None:
        return None
    order = [(first + k) % n for k in range(n)]
    new_len, new_turn = [], []
    prev_was_window = False
    k = 0
    while k < n:
        j = order[k]
        if j in s1 or j in s2:
            w = w1 if j in s1 else w2
            if prev_was_window:
                return None  # two diagonals back to back: skip
            new_len.append(L)
            new_turn.append(w["t_out"])
            new_turn[-2] = w["t_in"]  # the face before now turns into it
            k += w["count"]
            prev_was_window = True
            continue
        new_len.append(lengths[j])
        new_turn.append(turns[j])
        prev_was_window = False
        k += 1
    if abs(sum(new_turn)) != 8:
        return None
    return new_len, new_turn, hd[first]


def pairs(lengths, turns, max_pairs=400):
    """Candidate (window, window) pairs: equal leg, opposite direction.
    Biggest simplification first."""
    stairs = staircases(lengths, turns)
    out = []
    for w1, w2 in combinations(stairs, 2):
        if w1["a"] != w2["a"]:
            continue
        if w1["sign"] != (-w2["sign"][0], -w2["sign"][1]):
            continue
        if set(w1["idx"]) & set(w2["idx"]):
            continue
        out.append((w1, w2))
    out.sort(key=lambda p: -(p[0]["count"] + p[1]["count"]))
    return out[:max_pairs]


def diag_lengths(a):
    """Integer diagonal lengths close to the stair chord a*sqrt(2)."""
    base = round(a * math.sqrt(2))
    return [L for L in (base, base - 1, base + 1) if 5 <= L <= 30]
