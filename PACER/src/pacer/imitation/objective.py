"""
Learning Objectives
=======
"""
# src/pacer/imitation/objective.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from pacer.imitation.base import ILDataset, ILObjective, ILPolicy
from pacer.typings import DimAction, DimState, NumDemos, NumPoints, torchDType

## ── Objectives ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class WeightedHuberObjective(ILObjective[NumDemos, NumPoints, DimState, DimAction]):
    delta: float = 1.0

    @override
    def __call__(
        self,
        *,
        policy: ILPolicy[DimState, DimAction],
        dataset: ILDataset[NumDemos, NumPoints, DimState, DimAction],
        device: torch.device,
    ) -> Tensor:
        loss = torch.zeros((), dtype=torchDType, device=device)
        total_weight = torch.zeros((), dtype=torchDType, device=device)
        for i in range(dataset.states.length):
            states = torch.as_tensor(
                dataset.states[i].numpy(), dtype=torchDType, device=device
            )
            targets = torch.as_tensor(
                dataset.actions[i].numpy(), dtype=torchDType, device=device
            )
            preds = policy(states)
            per_dim = F.huber_loss(preds, targets, delta=self.delta, reduction="none")
            per_sample = per_dim.mean(dim=1)
            if dataset.weights is None:
                loss += per_sample.sum()
                total_weight += per_sample.numel()
            else:
                weights = torch.as_tensor(
                    dataset.weights[i], dtype=torchDType, device=device
                )
                loss += (per_sample * weights).sum()
                total_weight += weights.sum()
        return loss / total_weight


## ─────────────────────────────────────────────────────────────────────────────
