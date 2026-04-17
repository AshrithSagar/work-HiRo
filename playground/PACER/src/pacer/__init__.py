"""
PACER
=======
Progress-Aligned Curation for Error-Resilient Imitation Learning
"""
# src/pacer/__init__.py

from typing import Final

from rich.console import Console
from rich.traceback import install

install()

console: Final = Console()
