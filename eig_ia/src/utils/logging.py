import json
import os
import time
from typing import Any, Dict


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_run_dir(base_dir: str) -> str:
    override = os.environ.get("EIG_IA_RUN_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    ts = timestamp()
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
