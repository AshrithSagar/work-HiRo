"""
PACER
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/pacer/__init__.py

from pacer.pacer.pacer import PACER, PACERConfig, PACERResult

__all__ = [
    "PACER",
    "PACERConfig",
    "PACERResult",
]
