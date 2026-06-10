"""
Vector Mode
=======
Mostly HKT workarounds
"""
# src/pacer/pacer/mode.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Any, Callable

from typingkit.core import RuntimeGeneric

from pacer.base import (
    Action,
    ActionsCollection,
    Demonstrations,
    Sample,
    State,
    StatesCollection,
)
from pacer.pacer.base import MetricValue
from pacer.pacer.binning import ConsensusStatistics, RibbonToken
from pacer.typings import (
    CollectionType,
    DemoIndex,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    TimeIndex,
    VectorType,
)

## ── Vector Mode ──────────────────────────────────────────────────────────────


# VectorType :: State / Action
# CollectionType :: StatesCollection / ActionsCollection


@dataclass(frozen=True)
class VectorMode(
    RuntimeGeneric[CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction]
):
    """Encapsulates operations for state or action processing."""

    # Field access
    vector_from_sample: Callable[[Sample[DimState, DimAction]], VectorType]
    anchor_from_stats: Callable[[ConsensusStatistics[DimState, DimAction]], VectorType]
    strength_from_token: Callable[[RibbonToken[DimState, DimAction]], MetricValue]

    # Construction
    wrap: Callable[[Any], VectorType]
    make_collection: Callable[
        [Demonstrations[NumDemos, NumPoints, DimState, DimAction]], CollectionType
    ]
    set_item: Callable[[CollectionType, DemoIndex, TimeIndex, VectorType], None]
    get_item: Callable[[CollectionType, DemoIndex, TimeIndex], VectorType]

    # Correction behaviour
    attenuation_requires_state_tangent: bool


def ACTION_MODE() -> VectorMode[
    ActionsCollection[NumDemos, NumPoints, DimAction],
    Action[DimAction],
    NumDemos,
    NumPoints,
    DimState,
    DimAction,
]:
    """`VectorMode` configuration for actions."""
    return VectorMode(
        vector_from_sample=lambda sample: sample.action,
        anchor_from_stats=lambda stats: stats.action_anchor,
        strength_from_token=lambda token: token.action_strength,
        wrap=Action[DimAction],
        make_collection=lambda demos: ActionsCollection[
            NumDemos, NumPoints, DimAction
        ].zeros_like(demos),
        set_item=lambda col, i, t, v: col[i].__setitem__(t, v),
        get_item=lambda col, i, t: col[i][t],
        attenuation_requires_state_tangent=True,
    )


def STATE_MODE() -> VectorMode[
    StatesCollection[NumDemos, NumPoints, DimState],
    State[DimState],
    NumDemos,
    NumPoints,
    DimState,
    DimAction,
]:
    """`VectorMode` configuration for states."""
    return VectorMode(
        vector_from_sample=lambda sample: sample.state,
        anchor_from_stats=lambda stats: stats.state_anchor,
        strength_from_token=lambda token: token.state_norm,
        wrap=State[DimState],
        make_collection=lambda demos: StatesCollection[
            NumDemos, NumPoints, DimState
        ].zeros_like(demos),
        set_item=lambda col, i, t, v: col[i].__setitem__(t, v),
        get_item=lambda col, i, t: col[i][t],
        attenuation_requires_state_tangent=False,
    )


## ─────────────────────────────────────────────────────────────────────────────
