"""Scoring weights, provided to the player in advance (see spec)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    A: float  # multiplies enclosure area; positive, bigger area is better
    C: float  # multiplies enclosure perimeter; negative, longer perimeter costs more
    G: float  # flat bonus if a second gate exists on a different face; nonnegative

    def __post_init__(self):
        if self.C > 0:
            raise ValueError(
                f"C should be <= 0 (perimeter should cost, not pay), got {self.C}"
            )
        if self.G < 0:
            raise ValueError(f"G should be >= 0, got {self.G}")

    def to_dict(self) -> dict:
        return {"A": self.A, "C": self.C, "G": self.G}

    @staticmethod
    def from_dict(d: dict) -> "Weights":
        return Weights(A=float(d["A"]), C=float(d["C"]), G=float(d["G"]))
