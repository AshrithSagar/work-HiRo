"""
Learning Policies
=======
"""
# src/pacer/imitation/policy.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import cast, override

import torch.nn as nn
from torch import Tensor

from pacer.imitation.base import ILPolicy
from pacer.typings import DimAction, DimState

## ── Policies ─────────────────────────────────────────────────────────────────


class BCPolicy(ILPolicy[DimState, DimAction]):
    """Behavioral cloning policy that maps states to actions."""

    def __init__(
        self, state_dim: DimState, action_dim: DimAction, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.network: nn.Module = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @override
    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,)
        return cast(Tensor, self.network(states))


## ─────────────────────────────────────────────────────────────────────────────
