from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class LLMBase(ABC):
    def __init__(self, model_id: str, decoding_params: Dict[str, Any]):
        self.model_id = model_id
        self.decoding_params = decoding_params

    @abstractmethod
    def generate(self, prompt: str, n: int = 1) -> Tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def score(self, prompt: str, completions: List[str]) -> Tuple[List[float], Dict[str, Any]]:
        raise NotImplementedError
