"""File-backed state store with atomic writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


class State:
    def __init__(self, root: str | os.PathLike = "data"):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        target = (self.root / name).resolve()
        if self.root.resolve() not in target.parents:
            raise ValueError(f"state name escapes state dir: {name!r}")
        return target

    def write(self, name: str, df: pd.DataFrame) -> Path:
        """Write a frame as Parquet; readers never see a half-written file."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(name))
