"""A Scenario bundles everything a player is given up front: the room
footprint, their inventory of pieces/connectors, and the scoring weights."""

import json
import random
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.inventory import Inventory, generate_inventory
from src.room import Room, generate_room
from src.weights import Weights


@dataclass
class Scenario:
    room: Room
    inventory: Inventory
    weights: Weights

    def copy(self) -> "Scenario":
        return Scenario(
            room=self.room.copy(), inventory=self.inventory.copy(), weights=self.weights
        )

    def to_dict(self) -> dict:
        return {
            "room": list(self.room.get_boundary_points()),
            "inventory": self.inventory.to_dict(),
            "weights": self.weights.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            room=Room(Polygon(d["room"])),
            inventory=Inventory.from_dict(d["inventory"]),
            weights=Weights.from_dict(d["weights"]),
        )


def read_scenario(path: str) -> Scenario:
    with open(path, "r") as f:
        return Scenario.from_dict(json.load(f))


def write_scenario(path: str, scenario: Scenario):
    with open(path, "w") as f:
        json.dump(scenario.to_dict(), f, indent=2)
    print(f"wrote generated scenario to '{path}'")


def generate_scenario(seed: int, difficulty: str = "generous") -> Scenario:
    rng = random.Random(seed)
    room = generate_room(rng)
    inventory = generate_inventory(rng, difficulty=difficulty)
    weights = Weights(A=2.0, C=-3.0, G=100.0)
    return Scenario(room=room, inventory=inventory, weights=weights)
