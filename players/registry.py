import importlib

from demos.invalid_examples import EXAMPLES, InvalidExamplesPlayer
from players.example_player import ExamplePlayer
from players.player0 import Player

NUM_GROUPS = 10
GROUP_NAMES = [str(n) for n in range(1, NUM_GROUPS + 1)]

PLAYERS: dict[str, type[Player]] = {
    "e": ExamplePlayer,  # a worked example strategy for groups to read
    "i": InvalidExamplesPlayer,  # demo: cycles through deliberately invalid enclosures
}

# i1 ... i10: the invalid-examples demo pinned to one specific example, so each
# kind of mistake has its own command (e.g. `--player i6` = "walls cross")
for _n in range(1, len(EXAMPLES) + 1):
    PLAYERS[f"i{_n}"] = type(
        f"InvalidExample{_n}", (InvalidExamplesPlayer,), {"only": _n - 1}
    )

PLAYER_HELP = (
    f"1..{NUM_GROUPS} = that group's player (players/player<N>/player.py), "
    "e = the worked example strategy, "
    "i = cycle through invalid-enclosure examples (one per click), "
    f"i1..i{len(EXAMPLES)} = one specific invalid example"
)


class PlayerLoadError(Exception):
    """A group's player could not be loaded (the message says how to fix it)."""


def player_names() -> list[str]:
    return sorted(set(PLAYERS) | set(GROUP_NAMES), key=lambda name: (len(name), name))


def load_group_player(number: int) -> type[Player]:
    """The class `Player<number>` from players/player<number>/player.py."""
    module_name = f"players.player{number}.player"
    class_name = f"Player{number}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # only "the file isn't there" is ours to explain; a missing import
        # *inside* the group's code should show its own error
        if e.name in (module_name, f"players.player{number}"):
            raise PlayerLoadError(
                f"players/player{number}/player.py not found"
            ) from None
        raise
    cls = getattr(module, class_name, None)
    if cls is None:
        raise PlayerLoadError(
            f"players/player{number}/player.py must define a class {class_name}"
        )
    if not (isinstance(cls, type) and issubclass(cls, Player)):
        raise PlayerLoadError(
            f"{class_name} in players/player{number}/player.py must subclass "
            "players.player0.Player"
        )
    return cls


def get_player_class(name: str) -> type[Player]:
    if name in PLAYERS:
        return PLAYERS[name]
    if name in GROUP_NAMES:
        return load_group_player(int(name))
    raise KeyError(f"unknown player {name!r}; choose from: {', '.join(player_names())}")
