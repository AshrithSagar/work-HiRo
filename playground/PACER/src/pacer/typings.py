"""
Typings
=======
Typing utils
"""
# src/pacer/typings.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import TypeAlias, TypeVar

import numpy as np
import torch

## ── Typings ──────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
NumBins = TypeVar("NumBins", bound=int, default=int)  # B

## ─────────────────────────────────────────────────────────────────────────────
