import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def timed(metrics: Dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = metrics.get(key, 0.0) + (time.perf_counter() - start)
