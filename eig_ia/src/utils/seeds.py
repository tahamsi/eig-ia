import os
import random
from typing import Optional

import numpy as np


def set_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_rng(seed: Optional[int] = None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng
