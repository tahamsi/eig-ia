import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_base import LLMBase
from .tokenizer_utils import count_tokens, merge_usage


class HFLLM(LLMBase):
    def __init__(self, model_id: str, decoding_params: Dict[str, Any]):
        super().__init__(model_id, decoding_params)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def generate(self, prompt: str, n: int = 1) -> Tuple[List[str], Dict[str, Any]]:
        params = dict(self.decoding_params)
        max_new_tokens = int(params.pop("max_new_tokens", 64))
        temperature = float(params.pop("temperature", 0.7))
        top_p = float(params.pop("top_p", 0.95))
        do_sample = bool(params.pop("do_sample", True))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency = time.perf_counter() - start
        completions = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            completions.append(text[len(prompt) :].strip())
        usage = merge_usage(count_tokens(self.tokenizer, prompt), sum(count_tokens(self.tokenizer, c) for c in completions))
        return completions, {"latency": latency, "usage": usage}

    def score(self, prompt: str, completions: List[str]) -> Tuple[List[float], Dict[str, Any]]:
        scores = []
        start = time.perf_counter()
        for completion in completions:
            full_text = prompt + completion
            inputs = self.tokenizer(full_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            scores.append(-float(loss))
        latency = time.perf_counter() - start
        usage = merge_usage(sum(count_tokens(self.tokenizer, prompt) for _ in completions), sum(count_tokens(self.tokenizer, c) for c in completions))
        return scores, {"latency": latency, "usage": usage}
