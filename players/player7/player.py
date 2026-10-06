from players.player0 import Player
from src.enclosure import Construction


class Player7(Player):
    """Group 7: <one line on your strategy -- fill this in>."""

    def build_enclosure(self) -> Construction | None:
        # Set up for you by Player.__init__:
        #   self.room       the room footprint (self.room.polygon is a shapely Polygon;
        #                   self.room.contains_enclosure(polygon) checks a candidate fits)
        #   self.inventory  the walls, gates and connectors you may use
        #   self.weights    the scoring weights A, C and G
        #
        # Return a Construction describing a closed loop of walls/gates and
        # connectors, or None for "no solution".
        #
        # None scores 0; an invalid Construction scores -1000; a valid one scores
        # 1000 + A*area + C*perimeter (+ G for a second gate on another face).
        # So when you are not sure, return None.
        #
        # You get 300 s of CPU time for the whole run, including this class's
        # constructor (--cpu-limit changes it when testing). Past that the run
        # is stopped and scores 0.
        #
        # PLACEMENT_GUIDE.md covers placing a shape in the room; SIMULATOR_GUIDE.md
        # covers everything else; players/example_player.py is a worked example.
        return None
