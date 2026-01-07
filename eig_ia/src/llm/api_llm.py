import os
from typing import Any, Dict, List, Tuple

from .llm_base import LLMBase


class APILLM(LLMBase):
    def __init__(self, model_id: str, decoding_params: Dict[str, Any]):
        super().__init__(model_id, decoding_params)
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    def _check(self) -> None:
        if not self.api_key:
            raise RuntimeError("API key not found in OPENAI_API_KEY or ANTHROPIC_API_KEY")
        try:
            import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package not installed; install it to use APILLM") from exc

    def generate(self, prompt: str, n: int = 1) -> Tuple[List[str], Dict[str, Any]]:
        self._check()
        import openai  # type: ignore

        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=float(self.decoding_params.get("temperature", 0.7)),
            max_tokens=int(self.decoding_params.get("max_new_tokens", 128)),
        )
        texts = [choice.message["content"].strip() for choice in response.choices]
        usage = response.get("usage", {})
        return texts, {"latency": 0.0, "usage": usage}

    def score(self, prompt: str, completions: List[str]) -> Tuple[List[float], Dict[str, Any]]:
        self._check()
        import openai  # type: ignore

        scores = []
        for completion in completions:
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt + completion}],
                n=1,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
            )
            scores.append(float(response["choices"][0]["logprobs"]["token_logprobs"][0]))
        return scores, {"latency": 0.0, "usage": {}}
