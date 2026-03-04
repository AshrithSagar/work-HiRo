"""
Utils
=======
"""
# src/pacer/utils.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch

from pacer import console
from pacer.typings import npDType

## ── Utils ────────────────────────────────────────────────────────────────────

SEED = 42
EPS: float = 1e-8
MAD_SCALE: float = 1.4826  # Gaussian consistency factor for MAD


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_torch_device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        torch_device_auto = torch.device("mps")
    elif torch.cuda.is_available():
        torch_device_auto = torch.device("cuda")
    else:
        torch_device_auto = torch.device("cpu")
    console.print(f"Using device: [green]{torch_device_auto}[/]")
    return torch_device_auto


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return np.median(arr, axis=axis)  # type: ignore[no-any-return]  # ty: ignore[unused-ignore-comment]


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec, dtype=npDType)
    match method:
        case "NORM":
            norm = la.norm(vec)
            return vec / (norm + EPS)  # type: ignore[no-any-return]  # ty: ignore[unused-ignore-comment]
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)


## ─────────────────────────────────────────────────────────────────────────────
