"""Extra anchor points: interior starts and anchors derived from a closed loop."""

import math

import shapely
from shapely import affinity
from shapely.geometry.polygon import orient
from shapely.ops import polylabel

from players.player4.search import Anchor
from src.constants import TOL
from src.enclosure import Construction, compute_faces, walk_points


def frame_headings(polygon, keep: int = 3) -> list[float]:
    ring = list(orient(polygon, 1.0).exterior.coords)[:-1]
    weight: dict[float, float] = {}
    for a, b in zip(ring, ring[1:] + ring[:1]):
        h = round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90, 2)
        weight[h] = weight.get(h, 0) + math.dist(a, b)
    hs = sorted(weight, key=lambda h: -weight[h])[:keep]
    if not any(abs(h) < 1e-6 or abs(h - 90) < 1e-6 for h in hs):
        hs.append(0.0)
    return hs


def _largest_rectangle(mask: list[list[bool]]) -> tuple[int, int, int, int, int]:
    """Returns (area, row0, row1, col0, col1) for largest all-True rectangle."""
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    heights = [0] * cols
    best = (0, 0, 0, 0, 0)
    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if mask[r][c] else 0
        stack: list[tuple[int, int]] = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                start, sh = stack.pop()
                area = sh * (c - start)
                if area > best[0]:
                    best = (area, r - sh + 1, r, start, c - 1)
            stack.append((start, h))
    return best


def interior_anchors(polygon, cells: int = 48) -> list[Anchor]:
    """Starts away from walls: corners of a large inscribed rectangle and a
    point just inside the largest inscribed circle, in several orientations."""
    anchors: list[Anchor] = []
    for h in frame_headings(polygon):
        for turn in (0, 90, 180, 270):
            heading = h + turn
            rot = affinity.rotate(polygon, -heading, origin=(0, 0))
            minx, miny, maxx, maxy = rot.bounds
            span = max(maxx - minx, maxy - miny)
            if span < TOL:
                continue
            step = span / cells
            nx = max(2, int((maxx - minx) / step) + 1)
            ny = max(2, int((maxy - miny) / step) + 1)
            xs = [minx + step * i for i in range(nx)]
            ys = [miny + step * j for j in range(ny)]
            shrunk = rot.buffer(-1e-3)
            corner = shapely.contains_xy(
                shrunk,
                [x for x in xs for _ in ys],
                [y for _ in xs for y in ys],
            )
            # cell (i,j) uses corners (i,j),(i+1,j),(i,j+1),(i+1,j+1)
            mask = [[False] * (nx - 1) for _ in range(ny - 1)]
            for j in range(ny - 1):
                for i in range(nx - 1):
                    idx = j * nx + i
                    mask[j][i] = (
                        corner[idx]
                        and corner[idx + 1]
                        and corner[idx + nx]
                        and corner[idx + nx + 1]
                    )
            area, r0, _, c0, _ = _largest_rectangle(mask)
            if area == 0:
                continue
            x, y = xs[c0], ys[r0]
            t = math.radians(heading)
            point = (
                x * math.cos(t) - y * math.sin(t),
                x * math.sin(t) + y * math.cos(t),
            )
            anchors.append(Anchor(point, heading))

        rot = affinity.rotate(polygon, -h, origin=(0, 0))
        c = polylabel(rot, tolerance=0.5)
        radius = rot.exterior.distance(c)
        x, y = c.x, c.y - 0.95 * radius
        t = math.radians(h)
        anchors.append(
            Anchor(
                (x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)),
                h,
            )
        )
    return anchors


def wall_heading_offset_anchors(
    wall_anchors: list[Anchor], offset_deg: float = 45
) -> list[Anchor]:
    return [Anchor(a.start, a.heading_deg + offset_deg) for a in wall_anchors]


def anchors_from_construction(construction: Construction) -> list[Anchor]:
    """Each face start vertex, with heading along that face (counterclockwise)."""
    pts = walk_points(construction)[:-1]
    faces = compute_faces(construction)
    starts = [f.piece_indices[0] for f in faces]
    verts = [pts[i] for i in starts]
    out: list[tuple[tuple[float, float], float]] = []
    for k, v in enumerate(verts):
        w = verts[(k + 1) % len(verts)]
        out.append((v, math.degrees(math.atan2(w[1] - v[1], w[0] - v[0]))))
    area2 = sum(
        out[k][0][0] * out[(k + 1) % len(out)][0][1]
        - out[(k + 1) % len(out)][0][0] * out[k][0][1]
        for k in range(len(out))
    )
    if area2 < 0:
        rev = [v for v, _ in out][::-1]
        out = []
        for k, v in enumerate(rev):
            w = rev[(k + 1) % len(rev)]
            out.append((v, math.degrees(math.atan2(w[1] - v[1], w[0] - v[0]))))
    return [Anchor(v, h) for v, h in out]
