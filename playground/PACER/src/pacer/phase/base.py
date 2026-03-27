"""
Phase Base
=======
Core structures for representing phase.
"""
# src/pacer/phase/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Self, TypeAlias

from typingkit.core import TypedDict, TypedList

from pacer.base import Demonstration, Demonstrations
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    Vector,
    npDType,
)

## ── Phase Alignment ──────────────────────────────────────────────────────────

Phase: TypeAlias = npDType  # tau \in [0, 1]


class Phases(TypedList[NumPoints, Phase]):
    def numpy(self) -> Vector[NumPoints]:
        return Vector[NumPoints](self)

    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.length
        return cls.full(T_i, Phase(0))


class PhasesCollection(TypedDict[NumDemos, DemoIndex, Phases[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demonstrations.demo_indices,
            lambda i: Phases[NumPoints].zeros_like(demonstrations[i]),
        )


## ─────────────────────────────────────────────────────────────────────────────
