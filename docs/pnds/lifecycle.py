"""PNDS persistent-object lifecycle with a checkable registration guard."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import FrozenSet

class Status(str, Enum):
    EXPERIMENTAL = "E"
    PROVISIONAL = "P"
    REGISTERED = "R"
    QUARANTINED = "Q"
    RETIRED = "T"

@dataclass(frozen=True)
class Gates:
    utility: bool
    stability: bool
    integrity: bool
    cost: bool

    @property
    def all_pass(self) -> bool:
        return self.utility and self.stability and self.integrity and self.cost

class InvalidTransition(ValueError):
    pass

def transition(status: Status, target: Status, gates: Gates | None = None) -> Status:
    # The only transition into REGISTERED is guarded by all four gates.
    if target is Status.REGISTERED:
        if status is not Status.PROVISIONAL:
            raise InvalidTransition("only provisional objects may enter registered state")
        if gates is None or not gates.all_pass:
            raise InvalidTransition("registration requires all four gates")
        return target

    allowed = {
        Status.EXPERIMENTAL: {Status.PROVISIONAL, Status.QUARANTINED},
        Status.PROVISIONAL: {Status.QUARANTINED},
        Status.REGISTERED: {Status.RETIRED, Status.QUARANTINED},
        Status.QUARANTINED: {Status.EXPERIMENTAL},
        Status.RETIRED: set(),
    }
    if target not in allowed[status]:
        raise InvalidTransition(f"invalid transition {status.value}->{target.value}")
    return target
