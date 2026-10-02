"""File-backed state store with atomic writes."""

from __future__ import annotations

import json
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

    def write(self, name: str, obj: pd.DataFrame | dict | list) -> Path:
        """Write a frame (.parquet) or JSON value (.json) atomically.

        Readers never see a half-written file: data goes to a temp file in the
        same directory, which is then renamed over the target.
        """
        target = self.path(name)
        if target.suffix not in (".parquet", ".json"):
            raise ValueError(f"unsupported state file type: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        os.close(fd)
        try:
            if target.suffix == ".parquet":
                obj.to_parquet(tmp, index=False)
            else:
                with open(tmp, "w") as f:
                    json.dump(obj, f, indent=2)
                    f.write("\n")
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target

    def read(self, name: str) -> pd.DataFrame | dict | list:
        target = self.path(name)
        if target.suffix == ".json":
            return json.loads(target.read_text())
        return pd.read_parquet(target)

    def append(self, name: str, text: str) -> Path:
        """Append a line to a text log such as STATE.md (created if missing)."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as f:
            f.write(text.rstrip("\n") + "\n")
        return target
