"""Repository paths shared by the package.

Runtime state lives under ``data/`` and trained artifacts under
``checkpoints/``.  Both are git-ignored: they contain machine-local
telemetry and models trained on private data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
EXPORT = ROOT / "export"
DOCS = ROOT / "docs"

for _p in (DATA, CHECKPOINTS, EXPORT):
    _p.mkdir(parents=True, exist_ok=True)
