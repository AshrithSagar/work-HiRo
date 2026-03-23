"""
Utils
=======
"""
# src/pacer/utils.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from collections.abc import Sequence
from typing import Final, Literal, cast

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch
from torch._prims_common import DeviceLikeType

from pacer import console
from pacer.typings import npDType

## ── Utils ────────────────────────────────────────────────────────────────────

SEED: Final = 42
EPS: float = 1e-8
MAD_SCALE: float = 1.4826  # Gaussian consistency factor for MAD


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _get_torch_device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"Default device: [green]{device}[/green]")
    return device


TORCH_DEVICE: torch.device = _get_torch_device_auto()
_last_used_device: torch.device | None = None


def get_torch_device(device: DeviceLikeType = TORCH_DEVICE) -> torch.device:
    global _last_used_device
    device = torch.device(device)
    if device != _last_used_device:
        console.print(f"Using device: [green]{device}[/green]")
        _last_used_device = device
    return device


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return cast(np.ndarray, np.median(arr, axis=axis))


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> onp.Array1D[npDType]:
    _vec: onp.Array1D[npDType] = np.asarray(vec, dtype=npDType)
    match method:
        case "NORM":
            norm: npDType = la.norm(vec)
            return _vec / (norm + EPS)
        case "MINMAX" | "ZSCORE":
            _min = min(_vec)
            _max = max(_vec)
            return (_vec - _min) / (_max - _min + EPS)


## ─────────────────────────────────────────────────────────────────────────────
