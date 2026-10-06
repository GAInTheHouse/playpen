"""Keeps the documents honest: every code block in PLACEMENT_GUIDE.md runs, the
numbers it claims are true, every file the docs mention exists, and FILES.md
lists every file in the repository. Run with
`uv run python -m tests.test_placement_guide`."""

import contextlib
import io
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
GUIDE = PROJECT / "PLACEMENT_GUIDE.md"
DOCS = [
    "README.md",
    "PLACEMENT_GUIDE.md",
    "SIMULATOR_GUIDE.md",
    "MAINTAINERS.md",
    "FILES.md",
]

# not in a fresh clone, by design
GENERATED = {"logs/", ".venv/", "__pycache__/", ".ruff_cache/"}


def _blocks(markdown: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", markdown, re.DOTALL | re.MULTILINE)


_namespace_cache: dict = {}


def _run_guide() -> dict:
    """Run every python block of the guide, in order, in one shared namespace."""
    if _namespace_cache:
        return _namespace_cache
    namespace: dict = {"__name__": "placement_guide"}
    previous = Path.cwd()
    os.chdir(PROJECT)  # the guide reads scenarios/... relative to the project
    try:
        for number, code in enumerate(_blocks(GUIDE.read_text()), start=1):
            compiled = compile(code, f"PLACEMENT_GUIDE.md, code block {number}", "exec")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, namespace)  # noqa: S102 -- that is the test
    finally:
        os.chdir(previous)
    _namespace_cache.update(namespace)
    return namespace


def test_the_guide_has_code_and_every_block_runs_in_order():
    assert len(_blocks(GUIDE.read_text())) >= 8
    assert "find_placement" in _run_guide()  # got all the way to the last blocks


def _corners(vertices):
    return [(round(x, 3), round(y, 3)) for x, y in vertices]


def test_the_hand_walked_rectangle_matches_the_table_in_the_guide():
    from src.enclosure import build_geometry

    ns = _run_guide()
    vertices, position_gap, heading_gap = build_geometry(ns["rect"])
    assert _corners(vertices) == [(5, 5), (15, 5), (15, 11), (5, 11)]
    assert (
        0 <= position_gap < 1e-9 and heading_gap == 0.0
    )  # a float rounding error, not exactly 0


def test_the_validation_claims_in_the_guide():
    from src.enclosure import validate_construction

    ns = _run_guide()
    good = validate_construction(ns["rect"], ns["room"], ns["inventory"])
    assert (
        good.valid and round(good.area, 6) == 60.0 and round(good.perimeter, 6) == 32.0
    )
    bad = validate_construction(ns["no_gate"], ns["room"], ns["inventory"])
    assert not bad.valid and bad.reason == "enclosure must have at least one gate"


def test_the_rectangle_placement_claims():
    ns = _run_guide()
    room, shape_of, rectangle, rectangle_at = (
        ns["room"],
        ns["shape_of"],
        ns["rectangle"],
        ns["rectangle_at"],
    )
    centred = shape_of(rectangle_at((20, 15), 30, 12, 8))
    assert _corners([centred.centroid.coords[0]]) == [(20.0, 15.0)]
    assert round(centred.area, 6) == 96.0 and room.contains_enclosure(centred)
    assert room.contains_enclosure(shape_of(rectangle((0, 0), 0, 12, 8)))  # flush
    assert not room.contains_enclosure(
        shape_of(rectangle((35, 5), 0, 12, 8))
    )  # pokes out


def test_the_non_convex_room_claims():
    from shapely.geometry import Point

    ns = _run_guide()
    l_room, bridge = ns["l_room"], ns["bridge"]
    corners = bridge.exterior.coords[:-1]
    assert all(
        l_room.polygon.contains(Point(p)) for p in corners
    )  # every corner inside...
    assert not l_room.contains_enclosure(bridge)  # ...and it still doesn't fit

    u_room = ns["u_room"]
    centroid, inside = ns["centroid"], ns["inside"]
    assert _corners([(centroid.x, centroid.y)]) == [(15.0, 13.571)]
    assert not u_room.polygon.contains(centroid)
    assert _corners([(inside.x, inside.y)]) == [(5.0, 20.0)]
    assert u_room.polygon.contains(inside)


def test_the_tilted_room_claims():
    ns = _run_guide()
    assert ns["edge_headings"](ns["tilted_room"].polygon) == [33.0]
    assert round(ns["found"].start_heading) == 33
    assert ns["sweep"](15) == []
    assert ns["sweep"](1) == [32, 33, 34]


def test_the_l_shaped_enclosure_claims():
    from src.enclosure import build_geometry

    ns = _run_guide()
    l_shape, result = ns["l_shape"], ns["result"]
    assert sum(c.turn_angle() for c in l_shape.connectors) == 360.0
    assert _corners(build_geometry(l_shape)[0]) == [
        (10, 10),
        (30, 10),
        (30, 20),
        (20, 20),
        (20, 30),
        (10, 30),
    ]
    assert result.valid and round(result.area, 6) == 300.0


def test_find_placement_is_not_fooled_and_gives_up_on_shapes_that_cannot_fit():
    ns = _run_guide()
    find_placement, shape_of = ns["find_placement"], ns["shape_of"]
    for room in (ns["room"], ns["l_room"], ns["u_room"], ns["tilted_room"]):
        found = find_placement(room, 12, 8)
        assert found is not None and room.contains_enclosure(shape_of(found))
    assert find_placement(ns["u_room"], 45, 12) is None  # longer than the room


# -------------------------------------------------------------- docs <-> files


def _mentioned_paths(markdown: str) -> set[str]:
    """The file and folder names mentioned in `backticks`."""
    paths = set()
    for token in re.findall(r"`([^`\n]+)`", markdown):
        token = token.split(":")[0].strip()
        if (
            not token
            or not re.match(r"^[\w./-]+$", token)
            or token.startswith(
                ("-", "/", "logs/")
            )  # an option; a root; an example log
            or "playerN" in token  # a placeholder
            or re.match(
                r"^player\d+/", token
            )  # the tail of a range: "player1/ ... player10/"
        ):
            continue
        if token.endswith("/") or re.search(r"\.(py|md|json|yml|toml|lock)$", token):
            paths.add(token)
    return paths


def _exists(path: str) -> bool:
    if (PROJECT / path).exists():
        return True
    if (
        "/" not in path
    ):  # a bare file name, as in "constants.py": found somewhere in the project
        return any(
            ".venv" not in p.parts and "__pycache__" not in p.parts
            for p in PROJECT.rglob(path)
        )
    return False


def test_every_file_the_docs_mention_exists():
    missing = []
    for doc in DOCS:
        for path in sorted(_mentioned_paths((PROJECT / doc).read_text())):
            if path in GENERATED or path in {".gitignore"}:
                continue
            if not _exists(path):
                missing.append(f"{doc}: {path}")
    assert not missing, "docs mention files that don't exist: " + ", ".join(missing)


def test_files_md_lists_every_file_in_the_repository():
    text = (PROJECT / "FILES.md").read_text()
    skipped_dirs = {".venv", "__pycache__", ".ruff_cache", "logs", ".git"}
    unlisted = []
    for path in sorted(PROJECT.rglob("*")):
        relative = path.relative_to(PROJECT)
        if (
            path.is_dir()
            or set(relative.parts) & skipped_dirs
            or path.name == ".DS_Store"
        ):
            continue
        shown = relative.as_posix()
        # the ten group folders are documented once, as a range
        shown = re.sub(
            r"^(players|scenarios/players)/player\d+/", r"\1/player1/", shown
        )
        if shown.endswith("/.gitkeep"):
            shown = shown[: -len(".gitkeep")]
        if shown not in text:
            unlisted.append(shown)
    assert not unlisted, "FILES.md does not mention: " + ", ".join(
        sorted(set(unlisted))
    )


ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
