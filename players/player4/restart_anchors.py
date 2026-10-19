"""Restart beam search from each vertex of a closed loop."""

import math

from players.player4.search import Anchor
from src.enclosure import Construction, compute_faces, walk_points


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
