from abc import ABC, abstractmethod

from src.enclosure import Construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights


class PlayerException(Exception):
    pass


class Player(ABC):
    def __init__(self, room: Room, inventory: Inventory, weights: Weights) -> None:
        self.room = room
        self.inventory = inventory
        self.weights = weights
        # optional one-liner about what this player just did; the GUI shows
        # it in the result panel and the console prints it
        self.note: str | None = None

    def lacks_connectors(self, connector_type: ConnectorType, needed: int) -> bool:
        have = self.inventory.connectors.get(connector_type, 0)
        if have >= needed:
            return False
        self.note = (
            f"Not enough {connector_type.value} connectors "
            f"(need {needed}, have {have}) -- not attempting a build"
        )
        return True

    def __str__(self) -> str:
        return f"{self.__module__}()"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def build_enclosure(self) -> Construction | None:
        pass
