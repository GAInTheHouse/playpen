"""Player 4: boundary-search seeds followed by non-raster growth and repair."""

from players.player4.search_pipeline import SearchPlayer

SEARCH_SECONDS = 40.0


class Player4(SearchPlayer):
    """Spend about 20 CPU seconds finding seeds and 20 growing/repairing them.

    Joint piece assignment and the simulator validator guard every accepted
    construction. The budget is soft: an in-flight operation may finish later.
    """

    def __init__(self, room, inventory, weights):
        super().__init__(
            room, inventory, weights, variant="grown", budget=SEARCH_SECONDS
        )
