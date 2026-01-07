import hashlib
import json
import os
from typing import Any, Dict, Optional


def _hash_key(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_cache(cache_dir: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{_hash_key(key)}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_dir: str, key: Dict[str, Any], value: Dict[str, Any]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{_hash_key(key)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
    return path
