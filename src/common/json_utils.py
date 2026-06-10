from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, should_write


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read a UTF-8 JSON file.

    Input:
        path: JSON file path.
        default: Value returned when the file does not exist.
    Output:
        Parsed JSON data.
    TODO:
        Add schema validation for each stage artifact.
    """
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, config: dict) -> None:
    """Write JSON through the shared project policy.

    Input:
        path: Destination JSON path.
        data: JSON-serializable object.
        config: Pipeline config with overwrite setting.
    Output:
        None.
    TODO:
        Add optional compact JSON output for production artifacts.
    """
    if not should_write(path, config):
        return
    ensure_parent(path)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
